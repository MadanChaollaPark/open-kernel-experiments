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
def _compact_slots(IDS,LOCAL,ACTIVE,COUNT,SLOTS:tl.constexpr,START:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
    s=tl.arange(0,B);expert=tl.load(IDS+s,s<SLOTS,-1)
    valid=(s<SLOTS)&(expert>=START)&(expert<START+E)
    rank=tl.cumsum(valid.to(tl.int32),0)-1
    tl.store(ACTIVE+rank,s,valid)
    tl.store(LOCAL+s,tl.where(valid,expert-START,-1),s<SLOTS)
    tl.store(COUNT,tl.sum(valid.to(tl.int32),0))

@tr.jit
def _fp4_pair(W,SF,e,n,k,N:tl.constexpr,K:tl.constexpr):
    mask=(n[:,None]<N)&(k[None,:]<K//2)
    packed=tl.load(W+e*N*(K//2)+n[:,None]*(K//2)+k[None,:],mask,0).to(tl.int32)
    a=packed&15;b=packed>>4
    ma=a&7;mb=b&7
    ba=tl.where(ma<2,ma*0x3f00,0x3f00+(ma<<6))|((a&8)<<12)
    bb=tl.where(mb<2,mb*0x3f00,0x3f00+(mb<<6))|((b&8)<<12)
    sf=tl.load(SF+e*N*(K//16)+n[:,None]*(K//16)+k[None,:]//8,mask,0).to(tl.float32)
    return ba.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)*sf,bb.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)*sf

@tr.jit
def _compact_mv(X,IDS,ACTIVE,COUNT,W,SF,GS,Y,N:tl.constexpr,K:tl.constexpr,TOPK:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GATED:tl.constexpr):
    blocks=tl.cdiv(N,BN);work=tl.program_id(0);end=tl.load(COUNT)*blocks
    while work<end:
        slot=tl.load(ACTIVE+work//blocks);expert=tl.load(IDS+slot)
        n=(work%blocks)*BN+tl.arange(0,BN)
        kk=tl.arange(0,BK)
        acc=tl.full((BN,BK),0,tl.float32)
        if GATED:upacc=tl.full((BN,BK),0,tl.float32)
        for start in range(tl.cdiv(K//2,BK)):
            k=start*BK+kk
            x0=tl.load(X+(slot//TOPK)*K+2*k,2*k<K,0).to(tl.float32)
            x1=tl.load(X+(slot//TOPK)*K+2*k+1,2*k+1<K,0).to(tl.float32)
            if GATED:
                a,b=_fp4_pair(W,SF,expert,n,k,2*N,K)
                c,d=_fp4_pair(W,SF,expert,n+N,k,2*N,K)
                acc=tl.fma(a,x0[None,:],acc);acc=tl.fma(b,x1[None,:],acc)
                upacc=tl.fma(c,x0[None,:],upacc);upacc=tl.fma(d,x1[None,:],upacc)
            else:
                a,b=_fp4_pair(W,SF,expert,n,k,N,K)
                acc=tl.fma(a,x0[None,:],acc);acc=tl.fma(b,x1[None,:],acc)
        gs=tl.load(GS+expert)
        if GATED:
            gate=(tl.sum(acc,1)*gs).to(tl.bfloat16).to(tl.float32)
            up=(tl.sum(upacc,1)*gs).to(tl.bfloat16).to(tl.float32)
            out=gate/(1+tl.exp(-gate))*up
        else:out=tl.sum(acc,1)*gs
        tl.store(Y+slot*N+n,out,n<N)
        work+=tl.num_programs(0)

def kernel_fn(data,state):
    cfg=data['config'];m=cfg['bs'];h=cfg['d_hidden'];i=cfg['d_expert'];topk=cfg['top_k']
    if m>32:return _best_moe(data,state)
    x=data['hidden_states'];nv=state['nv'];slots=m*topk
    local=torch.empty_like(data['topk_ids']);active=torch.empty((slots,),device=x.device,dtype=torch.int32);count=torch.empty((),device=x.device,dtype=torch.int32)
    _compact_slots[(1,)](data['topk_ids'],local,active,count,slots,cfg['local_start'],cfg['local_count'],tr.next_power_of_2(slots),num_warps=4)
    a=torch.empty((slots,i),device=x.device,dtype=torch.bfloat16);y=torch.empty((slots,h),device=x.device,dtype=torch.bfloat16)
    _compact_mv[(min(528,tr.cdiv(i,4)*slots),)](x,data['topk_ids'],active,count,nv['w13'],nv['w13_sf'],nv['w13_gs'],a,i,h,topk,4,128,True,num_warps=4)
    _compact_mv[(min(528,tr.cdiv(h,4)*slots),)](a,data['topk_ids'],active,count,nv['w2'],nv['w2_sf'],nv['w2_gs'],y,h,i,1,4,128,False,num_warps=4)
    return fused_post_expert_norm_weighted_sum(y.view(m,topk,h),data['topk_weights'],norm_weight=state['norm_w'],eps=POST_NORM_EPS,weight_first=True,slot_expert_ids=local)
