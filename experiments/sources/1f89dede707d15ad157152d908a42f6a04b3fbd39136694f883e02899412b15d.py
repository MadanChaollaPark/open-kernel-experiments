# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _base_prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _best_moe(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    if num_tokens >= 128:
        block_size_m = 16
    else:
        for block_size_m in BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
                break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)

import triton as tr
import triton.language as tl

def prepare(weights):
    state=_base_prepare(weights)
    state['nv']={name:weights[name] for name in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}
    return state

@tr.jit
def _load_nv(W,SF,expert,n,k,N:tl.constexpr,K:tl.constexpr):
    offset=expert*N*(K//2)+n[:,None]*(K//2)+k[None,:]//2
    packed=tl.load(W+offset,(n[:,None]<N)&(k[None,:]<K),0).to(tl.int32)
    code=(packed>>((k[None,:]%2)*4))&15;mag=code&7
    bits=tl.where(mag<2,mag*0x3f00,0x3f00+(mag<<6))|((code&8)<<12)
    val=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)
    scale=tl.load(SF+expert*N*(K//16)+n[:,None]*(K//16)+k[None,:]//16,(n[:,None]<N)&(k[None,:]<K),0.0).to(tl.float32)
    return val*scale

@tr.jit
def _full_nv_mv(X,IDS,W,SF,GS,Y,N:tl.constexpr,K:tl.constexpr,TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GATED:tl.constexpr):
    slot=tl.program_id(1);expert=tl.load(IDS+slot)
    if expert>=0:
        n=tl.program_id(0)*BN+tl.arange(0,BN);k=tl.arange(0,BK)
        x=tl.load(X+(slot//TOPK)*K+k,k<K,0.0).to(tl.float32)
        gs=tl.load(GS+expert)
        if GATED:
            w1=_load_nv(W,SF,expert,n,k,2*N,K)
            w2=_load_nv(W,SF,expert,n+N,k,2*N,K)
            gate=(tl.sum(w1*x[None,:],1)*gs).to(tl.bfloat16).to(tl.float32)
            up=(tl.sum(w2*x[None,:],1)*gs).to(tl.bfloat16).to(tl.float32)
            out=gate/(1+tl.exp(-gate))*up
        else:
            w=_load_nv(W,SF,expert,n,k,N,K)
            out=tl.sum(w*x[None,:],1)*gs
        tl.store(Y+slot*N+n,out,n<N)

def kernel_fn(data,state):
    cfg=data['config'];m=cfg['bs'];h=cfg['d_hidden'];i=cfg['d_expert'];topk=cfg['top_k']
    if m>8:return _best_moe(data,state)
    bank=state['banks'][(cfg['local_start'],cfg['local_count'])]
    local=bank['expert_map'][data['topk_ids']].contiguous()
    x=data['hidden_states'];nv=state['nv']
    activated=torch.empty((m*topk,i),device=x.device,dtype=torch.bfloat16)
    y=torch.empty((m*topk,h),device=x.device,dtype=torch.bfloat16)
    _full_nv_mv[(tr.cdiv(i,1),m*topk)](x,local,nv['w13'],nv['w13_sf'],nv['w13_gs'],activated,i,h,topk,1,tr.next_power_of_2(h),True,num_warps=8)
    _full_nv_mv[(tr.cdiv(h,1),m*topk)](activated,local,nv['w2'],nv['w2_sf'],nv['w2_gs'],y,h,i,1,1,tr.next_power_of_2(i),False,num_warps=8)
    return fused_post_expert_norm_weighted_sum(y.view(m,topk,h),data['topk_weights'],norm_weight=state['norm_w'],eps=POST_NORM_EPS,weight_first=True,slot_expert_ids=local)
