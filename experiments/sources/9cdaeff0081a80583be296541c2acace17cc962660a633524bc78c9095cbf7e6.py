# okc: problem=muse_decode_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _layer_prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = topk_ids
    topk_weights = topk_weights.to(torch.float32).contiguous()
    block_size_m=16 if num_tokens>=128 else 8
    if num_tokens<=32:
        slots=num_tokens*TOP_K
        maxpad=tr.cdiv(slots+LOCAL_EXPERTS*(block_size_m-1),block_size_m)*block_size_m
        maxexp=maxpad//block_size_m
        sorted_ids=torch.empty((maxpad,),device=x.device,dtype=torch.int32)
        block_experts=torch.empty((maxexp,),device=x.device,dtype=torch.int32)
        n_padded=torch.empty((1,),device=x.device,dtype=torch.int32)
        _compact_align[(1,)](local_ids,local_ids,sorted_ids,block_experts,n_padded,slots,0,LOCAL_EXPERTS,block_size_m,tr.next_power_of_2(slots),tr.next_power_of_2(LOCAL_EXPERTS),tr.next_power_of_2(maxpad),tr.next_power_of_2(tr.cdiv(slots,block_size_m)),maxpad,maxexp,False,num_warps=16 if slots>=512 else 8)
    elif num_tokens<=1024:
        slots=num_tokens*TOP_K
        maxpad=tr.cdiv(slots+LOCAL_EXPERTS*(block_size_m-1),block_size_m)*block_size_m
        maxexp=maxpad//block_size_m
        sorted_ids=torch.empty((maxpad,),device=x.device,dtype=torch.int32)
        block_experts=torch.empty((maxexp,),device=x.device,dtype=torch.int32)
        n_padded=torch.empty((1,),device=x.device,dtype=torch.int32)
        state['align_module'].align_routes(local_ids,local_ids,sorted_ids,block_experts,n_padded,0,LOCAL_EXPERTS,block_size_m,256)
    elif num_tokens==1:
        sorted_ids=state['single_sorted'];block_experts=local_ids.flatten();n_padded=state['single_count']
    else:
        sorted_ids,block_experts,n_padded=moe_align_block_size(local_ids,block_size_m,LOCAL_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
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
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv=_dense_op(data['attn_input'],'qkv_gate',state)
    adata=dict(data);adata['qkv']=qkv
    o=attention_kernel_fn(adata).reshape(tokens,hq*d)
    attn_output=_dense_op(o,'o',state)

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _dense_op(ffn_input,'pooler',state)
    with torch.cuda.stream(side_stream):
        if split_router:
            logits=_dense_op(ffn_input,'router',state)
        else:
            logits = _small_router(router_input,state)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    normalized,local_ids=_normalize_and_map(expert_input,topk_ids,data['local_start'],state['pxw'])
    # all-reduce: identity
    expert_output = _experts(normalized, topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _dense_op(expert_output,'unpooler',state,n=UNPOOL_SLICE)
        ffn_output = unpooled  # all-gather: identity
    else:
        ffn_output = _dense_op(expert_output,'unpooler',state)

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = _end_boundary(stream,ffn_output,state)
    return next_attn_input, stream, topk_ids

import torch
import triton as tr
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

@tr.jit
def dense__mm(X, W, L, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr, TRANS: tl.constexpr, NFULL: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.program_id(2) * BK + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(tl.cdiv(K, BK * S)):
        a = tl.load(X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0)
        off = k[:, None] * NFULL + n[None, :] if TRANS else n[None, :] * K + k[:, None]
        b = tl.load(W + off, (n[None, :] < N) & (k[:, None] < K), 0)
        acc = tl.dot(a, b, acc)
        if ROUTER:
            lo = tl.load(L + off, (n[None, :] < N) & (k[:, None] < K), 0)
            acc = tl.dot(a, lo, acc)
        k += BK * S
    tl.store(Y + tl.program_id(2) * M * N + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))

@tr.jit
def dense__reduce(P, Y, T: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for s in range(S):
        acc += tl.load(P + s * T + i, i < T, 0)
    tl.store(Y + i, acc, i < T)

def dense_prepare(weights):
    state = dict(weights)
    state['qkv_t'] = weights['qkv_gate'].t().contiguous()
    state['pooler_t'] = weights['pooler'].t().contiguous()
    hi = weights['router'].to(torch.bfloat16)
    lo = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['hi'] = hi
    state['lo'] = lo
    state['hi_t'] = hi.t().contiguous()
    state['lo_t'] = lo.t().contiguous()
    return state

def dense__hybrid(data, state):
    x = data['x']
    m, k = x.shape
    layer = data['config']['layer']
    n = data['config']['n']
    router = layer == 'router'
    if router:
        bm = 16 if m <= 16 else 32 if m <= 32 else 64
        bn = 32 if m <= 128 else 64
        bk = 128
        split = 8
        trans = 32 < m <= 128
        w = state['hi_t'] if trans else state['hi']
        lo = state['lo_t'] if trans else state['lo']
    elif layer == 'qkv_gate' and m <= 16:
        bm = 16
        bn = 64
        bk = 128
        split = 8
        trans = True
        w = state['qkv_t']
        lo = state['lo']
    elif layer == 'pooler' and 8 < m <= 16:
        bm = 16
        bn = 64
        bk = 128
        split = 4
        trans = False
        w = state[layer]
        lo = state['lo']
    elif layer in ('unpooler', 'pooler') and m <= 8:
        from flashinfer.gemm import mm_bf16
        return mm_bf16(x, state[layer][:n].t(), backend='tinygemm')
    elif layer == 'pooler':
        return torch.mm(x, state['pooler_t'])
    else:
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    y = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    p = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    full = w.shape[1] if trans else n
    dense__mm[tr.cdiv(m, bm), tr.cdiv(n, bn), split](x, w, lo, p, m, n, k, bm, bn, bk, split, router, trans, full, num_warps=4, num_stages=3)
    dense__reduce[tr.cdiv(m * n, 512),](p, y, m * n, split, 512)
    return y

@tr.jit
def dense__fullvec(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.program_id(2) * BK + tl.arange(0, BK)
    x = tl.load(X + m * K + k, k < K, 0).to(tl.float32)
    w = tl.load(W + n[:, None] * K + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    out = tl.sum(w * x[None, :], 1)
    tl.store(Y + tl.program_id(2) * M * N + m * N + n, out, n < N)

def dense_kernel_fn(data, state):
    x = data['x']
    m, k = x.shape
    layer = data['config']['layer']
    n = data['config']['n']
    if m != 1 or layer == 'o':
        return dense__hybrid(data, state)
    bn = 2 if layer == 'qkv_gate' else 1
    warps = 8 if layer in ('qkv_gate', 'unpooler') else 4
    y = torch.empty((m, n), device=x.device, dtype=torch.float32 if layer == 'router' else torch.bfloat16)
    dense__fullvec[m, tr.cdiv(n, bn), 1](x, state[layer], y, m, n, k, bn, tr.next_power_of_2(k), num_warps=warps)
    return y
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

@tr.jit
def attention__prepare(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    v = tl.load(QKV + t * stride + h * 64 + d).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / 64 + 1e-05)
    v = v * r
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h * 64 + (d ^ 1)).to(tl.float32) * r
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle) + tl.where(d % 2 == 0, -partner, partner) * libdevice.sin(angle)
    v = v.to(tl.bfloat16).to(tl.float32)
    if h < HQ:
        tl.store(Q + (t * HQ + h) * 64 + d, v)
    else:
        slot = tl.load(LOC + t)
        ks = tl.load(KS)
        vs = tl.load(VS)
        val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h - HQ) * 64 + d).to(tl.float32)
        off = (slot * HK + h - HQ) * 64 + d
        tl.store(KC + off, v / ks)
        tl.store(VC + off, val / vs)

@tr.jit
def attention__attention(Q, KC, VC, KS, VS, PT, SEQ, P, LSE, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, PAGE: tl.constexpr, MAXP: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    group: tl.constexpr = HQ // HK
    r = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nn = tl.arange(0, BN)
    token = r // group
    head = h * group + r % group
    seq = tl.load(SEQ + b)
    q = tl.load(Q + ((b * QL + token[:, None]) * HQ + head[:, None]) * 64 + d[None, :], r[:, None] < QL * group, 0.0)
    ks = tl.load(KS)
    vs = tl.load(VS)
    if WINDOW > 0:
        first = tl.maximum(0, seq - WINDOW + 1)
    else:
        first = 0
    begin = first + s * CHUNK
    acc = tl.full((BM, 64), 0, tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    peak = tl.full((BM,), -1000000.0, tl.float32)
    for offset in range(tl.cdiv(CHUNK, BN)):
        pos = begin + offset * BN + nn
        valid = (pos < seq + QL) & (pos < begin + CHUNK)
        page = tl.load(PT + b * MAXP + pos // PAGE, valid, 0)
        row = page * PAGE + pos % PAGE
        k = tl.load(KC + (row[None, :] * HK + h) * 64 + d[:, None], valid[None, :], 0.0)
        v = tl.load(VC + (row[:, None] * HK + h) * 64 + d[None, :], valid[:, None], 0.0).to(tl.bfloat16)
        mask = valid[None, :] & (r[:, None] < QL * group) & (pos[None, :] <= seq + token[:, None])
        if WINDOW > 0:
            mask = mask & (pos[None, :] >= seq + token[:, None] - WINDOW + 1)
        score = tl.dot(q, k) * SCALE * ks * 1.4426950408889634
        score = tl.where(mask, score, -1000000.0)
        next_peak = tl.maximum(peak, tl.max(score, 1))
        alpha = tl.exp2(peak - next_peak)
        prob = tl.where(mask, tl.exp2(score - next_peak[:, None]), 0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(prob.to(tl.bfloat16), v, acc)
        denom = denom * alpha + tl.sum(prob, 1)
        peak = next_peak
    normalized = acc * vs / tl.maximum(denom[:, None], 1e-30)
    logsum = tl.where(denom > 0, peak + tl.log2(denom), -1000000.0)
    base = ((b * HK + h) * SPLITS + s) * BM
    tl.store(P + (base + r[:, None]) * 64 + d[None, :], normalized)
    tl.store(LSE + base + r, logsum)

@tr.jit
def attention__finish(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h) * SPLITS + s) * BM + r
    lse = tl.load(LSE + base, s < SPLITS, -1000000.0)
    mx = tl.max(lse, 0)
    probability = tl.where(s < SPLITS, tl.exp2(lse - mx), 0.0)
    probability = probability / tl.sum(probability, 0)
    part = tl.load(P + base[:, None] * 64 + d[None, :], s[:, None] < SPLITS, 0)
    a = tl.sum(part * probability[:, None], 0).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 0) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head * 64 + d).to(tl.float32)
    out = a * norm / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head) * 64 + d, out)

def attention__hybrid(data):
    cfg = data['config']
    b = cfg['batch']
    ql = cfg['q_len']
    hq = cfg['num_q_heads']
    hk = cfg['num_kv_heads']
    qkv = data['qkv']
    q = torch.empty((b * ql, hq, 64), device=qkv.device, dtype=torch.float8_e4m3fn)
    if b * ql >= 16:
        attention__prepare_heads[b * ql, tr.cdiv(hq + hk, 4)](qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], q, hq, hk, cfg['rope_theta'], 4, num_warps=4)
    else:
        attention__prepare[b * ql, hq + hk](qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], q, hq, hk, cfg['rope_theta'], num_warps=4)
    maxp = data['page_table'].shape[1]
    page = cfg['page_size']
    window = cfg['window']
    if window == 0 and ql > 1 and (b >= 8):
        descale = {}
        if cfg['kv_scale'] != 1.0:
            descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(b, hk), v_descale=data['v_scale'].reshape(1, 1).expand(b, hk))
        out = flash_attn_with_kvcache(q=q, k_cache=data['k_cache'], v_cache=data['v_cache'], page_table=data['page_table'], cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=ql, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'], causal=True, scheduler_metadata=data['scheduler_metadata'], **descale)
        if isinstance(out, tuple):
            out = out[0]
        gate = qkv.reshape(b * ql, -1)[:, (hq + 2 * hk) * 64:]
        return rms_norm_gate(out.reshape(b * ql, hq * 64).to(torch.bfloat16), gate, None, head_dim=64, eps=1e-05, has_norm=True).reshape(b, ql, hq * 64)
    span = window + ql - 1 if window else maxp * page
    bn = 64 if window and b < 8 else 128
    blocks = 2048 if window and 8 <= b < 32 or (not window and ql > 1) else 512 if b >= 32 else 1024
    splits = min(128, max(4, tr.next_power_of_2(tr.cdiv(blocks, b * hk))))
    chunk = tr.cdiv(span, splits * bn) * bn
    splits = tr.cdiv(span, chunk)
    bm = max(16, tr.next_power_of_2(ql * (hq // hk)))
    p = torch.empty((b, hk, splits, bm, 64), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((b, hk, splits, bm), device=qkv.device, dtype=torch.float32)
    y = torch.empty((b, ql, hq * 64), device=qkv.device, dtype=torch.bfloat16)
    attention__attention[b, hk, splits](q, data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], data['page_table'], data['seq_lens'], p, lse, hq, hk, ql, page, maxp, window, cfg['sm_scale'] * cfg['qk_scale_factor'], splits, chunk, bm, bn, num_warps=4, num_stages=2)
    if b * ql >= 32 and splits <= 64:
        attention__finish_heads[b * ql, tr.cdiv(hq, 4)](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), 4, num_warps=4)
    else:
        attention__finish[b * ql, hq](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), num_warps=4)
    return y

@tr.jit
def attention__prepare_heads(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    mask = h[:, None] < HQ + HK
    v = tl.load(QKV + t * stride + h[:, None] * 64 + d[None, :], mask, 0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 1) / 64 + 1e-05)
    v = v * r[:, None]
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h[:, None] * 64 + (d[None, :] ^ 1), mask, 0.0).to(tl.float32) * r[:, None]
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle)[None, :] + tl.where(d[None, :] % 2 == 0, -partner, partner) * libdevice.sin(angle)[None, :]
    v = v.to(tl.bfloat16).to(tl.float32)
    tl.store(Q + (t * HQ + h[:, None]) * 64 + d[None, :], v, h[:, None] < HQ)
    slot = tl.load(LOC + t)
    ks = tl.load(KS)
    vs = tl.load(VS)
    kv_mask = (h[:, None] >= HQ) & (h[:, None] < HQ + HK)
    val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h[:, None] - HQ) * 64 + d[None, :], kv_mask, 0.0).to(tl.float32)
    off = (slot * HK + h[:, None] - HQ) * 64 + d[None, :]
    tl.store(KC + off, v / ks, kv_mask)
    tl.store(VC + off, val / vs, kv_mask)

@tr.jit
def attention__finish_heads(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h[:, None]) * SPLITS + s[None, :]) * BM + r[:, None]
    mask = (s[None, :] < SPLITS) & (head[:, None] < HQ)
    lse = tl.load(LSE + base, mask, -1000000.0)
    mx = tl.max(lse, 1)
    probability = tl.where(mask, tl.exp2(lse - mx[:, None]), 0.0)
    probability = probability / tl.sum(probability, 1)[:, None]
    part = tl.load(P + base[:, :, None] * 64 + d[None, None, :], mask[:, :, None], 0.0)
    a = tl.sum(part * probability[:, :, None], 1).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 1) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head[:, None] * 64 + d[None, :], head[:, None] < HQ, 0.0).to(tl.float32)
    out = a * norm[:, None] / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head[:, None]) * 64 + d[None, :], out, head[:, None] < HQ)
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

@tr.jit
def attention_inline__prepare(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    v = tl.load(QKV + t * stride + h * 64 + d).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / 64 + 1e-05)
    v = v * r
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h * 64 + (d ^ 1)).to(tl.float32) * r
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle) + tl.where(d % 2 == 0, -partner, partner) * libdevice.sin(angle)
    v = v.to(tl.bfloat16).to(tl.float32)
    if h < HQ:
        tl.store(Q + (t * HQ + h) * 64 + d, v)
    else:
        slot = tl.load(LOC + t)
        ks = tl.load(KS)
        vs = tl.load(VS)
        val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h - HQ) * 64 + d).to(tl.float32)
        off = (slot * HK + h - HQ) * 64 + d
        tl.store(KC + off, v / ks)
        tl.store(VC + off, val / vs)

@tr.jit
def attention_inline__attention_inline(QKV, POS, LOC, KC, VC, KS, VS, PT, SEQ, P, LSE, CS, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, PAGE: tl.constexpr, MAXP: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, THETA: tl.constexpr, BNEW: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    group: tl.constexpr = HQ // HK
    r = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nn = tl.arange(0, BN)
    token = r // group
    head = h * group + r % group
    seq = tl.load(SEQ + b)
    stride: tl.constexpr = (2 * HQ + 2 * HK) * 64
    tq = b * QL + token
    q = tl.load(QKV + tq[:, None] * stride + head[:, None] * 64 + d[None, :], r[:, None] < QL * group, 0.0).to(tl.float32)
    qnorm = tl.rsqrt(tl.sum(q * q, 1) / 64 + 1e-05)
    q = q * qnorm[:, None]
    if THETA != 0.0:
        qp = tl.load(QKV + tq[:, None] * stride + head[:, None] * 64 + (d[None, :] ^ 1), r[:, None] < QL * group, 0.0).to(tl.float32) * qnorm[:, None]
        position = tl.load(POS + tq, r < QL * group, 0)
        c = tl.load(CS + position[:, None] * 64 + d[None, :] // 2)
        sn = tl.load(CS + position[:, None] * 64 + 32 + d[None, :] // 2)
        q = q * c + tl.where(d[None, :] % 2 == 0, -qp, qp) * sn
    q = q.to(tl.bfloat16).to(tl.float8e4nv)
    ks = tl.load(KS)
    vs = tl.load(VS)
    if WINDOW > 0:
        first = tl.maximum(0, seq - WINDOW + 1)
    else:
        first = 0
    begin = first + s * CHUNK
    acc = tl.full((BM, 64), 0, tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    peak = tl.full((BM,), -1000000.0, tl.float32)
    for offset in range(tl.cdiv(CHUNK, BN)):
        pos = begin + offset * BN + nn
        valid = (pos < seq) & (pos < begin + CHUNK)
        page = tl.load(PT + b * MAXP + pos // PAGE, valid, 0)
        row = page * PAGE + pos % PAGE
        k = tl.load(KC + (row[None, :] * HK + h) * 64 + d[:, None], valid[None, :], 0.0)
        v = tl.load(VC + (row[:, None] * HK + h) * 64 + d[None, :], valid[:, None], 0.0).to(tl.bfloat16)
        mask = valid[None, :] & (r[:, None] < QL * group) & (pos[None, :] <= seq + token[:, None])
        if WINDOW > 0:
            mask = mask & (pos[None, :] >= seq + token[:, None] - WINDOW + 1)
        score = tl.dot(q, k) * SCALE * ks * 1.4426950408889634
        score = tl.where(mask, score, -1000000.0)
        next_peak = tl.maximum(peak, tl.max(score, 1))
        alpha = tl.exp2(peak - next_peak)
        prob = tl.where(mask, tl.exp2(score - next_peak[:, None]), 0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(prob.to(tl.bfloat16), v, acc)
        denom = denom * alpha + tl.sum(prob, 1)
        peak = next_peak
    if s == SPLITS - 1:
        j = tl.arange(0, BNEW)
        tnew = b * QL + j
        kn = tl.load(QKV + tnew[:, None] * stride + (HQ + h) * 64 + d[None, :], j[:, None] < QL, 0.0).to(tl.float32)
        kr = tl.rsqrt(tl.sum(kn * kn, 1) / 64 + 1e-05)
        kn = kn * kr[:, None]
        if THETA != 0.0:
            kp = tl.load(QKV + tnew[:, None] * stride + (HQ + h) * 64 + (d[None, :] ^ 1), j[:, None] < QL, 0.0).to(tl.float32) * kr[:, None]
            new_position = tl.load(POS + tnew, j < QL, 0)
            nc = tl.load(CS + new_position[:, None] * 64 + d[None, :] // 2)
            ns = tl.load(CS + new_position[:, None] * 64 + 32 + d[None, :] // 2)
            kn = kn * nc + tl.where(d[None, :] % 2 == 0, -kp, kp) * ns
        kn = (kn.to(tl.bfloat16).to(tl.float32) / ks).to(tl.float8e4nv)
        vn = tl.load(QKV + tnew[:, None] * stride + (HQ + HK + h) * 64 + d[None, :], j[:, None] < QL, 0.0).to(tl.float32)
        vn = (vn / vs).to(tl.float8e4nv)
        slot = tl.load(LOC + tnew, j < QL, 0)
        tl.store(KC + (slot[:, None] * HK + h) * 64 + d[None, :], kn, j[:, None] < QL)
        tl.store(VC + (slot[:, None] * HK + h) * 64 + d[None, :], vn, j[:, None] < QL)
        new_mask = (r[:, None] < QL * group) & (j[None, :] < QL) & (j[None, :] <= token[:, None])
        if WINDOW > 0:
            new_mask = new_mask & (j[None, :] >= token[:, None] - WINDOW + 1)
        new_score = tl.dot(q, tl.trans(kn)) * SCALE * ks * 1.4426950408889634
        new_score = tl.where(new_mask, new_score, -1000000.0)
        new_next_peak = tl.maximum(peak, tl.max(new_score, 1))
        new_alpha = tl.exp2(peak - new_next_peak)
        new_prob = tl.where(new_mask, tl.exp2(new_score - new_next_peak[:, None]), 0.0)
        acc = acc * new_alpha[:, None]
        acc = tl.dot(new_prob.to(tl.bfloat16), vn.to(tl.bfloat16), acc)
        denom = denom * new_alpha + tl.sum(new_prob, 1)
        peak = new_next_peak
    normalized = acc * vs / tl.maximum(denom[:, None], 1e-30)
    logsum = tl.where(denom > 0, peak + tl.log2(denom), -1000000.0)
    base = ((b * HK + h) * SPLITS + s) * BM
    tl.store(P + (base + r[:, None]) * 64 + d[None, :], normalized)
    tl.store(LSE + base + r, logsum)

@tr.jit
def attention_inline__finish(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h) * SPLITS + s) * BM + r
    lse = tl.load(LSE + base, s < SPLITS, -1000000.0)
    mx = tl.max(lse, 0)
    probability = tl.where(s < SPLITS, tl.exp2(lse - mx), 0.0)
    probability = probability / tl.sum(probability, 0)
    part = tl.load(P + base[:, None] * 64 + d[None, :], s[:, None] < SPLITS, 0)
    a = tl.sum(part * probability[:, None], 0).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 0) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head * 64 + d).to(tl.float32)
    out = a * norm / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head) * 64 + d, out)

def attention_inline_kernel_fn(data):
    cfg = data['config']
    b = cfg['batch']
    ql = cfg['q_len']
    hq = cfg['num_q_heads']
    hk = cfg['num_kv_heads']
    qkv = data['qkv']
    cs = attention_inline__cos_sin(cfg['rope_theta'], qkv.device) if cfg['rope_theta'] else qkv
    maxp = data['page_table'].shape[1]
    page = cfg['page_size']
    window = cfg['window']
    span = window + ql - 1 if window else maxp * page
    bn = 128 if not window or b >= 8 else 64
    blocks = 2048 if not window or b >= 8 else 512 if ql > 1 else 1024
    splits = min(128, max(4, tr.next_power_of_2(tr.cdiv(blocks, b * hk))))
    chunk = tr.cdiv(span, splits * bn) * bn
    splits = tr.cdiv(span, chunk)
    bm = max(16, tr.next_power_of_2(ql * (hq // hk)))
    p = torch.empty((b, hk, splits, bm, 64), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((b, hk, splits, bm), device=qkv.device, dtype=torch.float32)
    y = torch.empty((b, ql, hq * 64), device=qkv.device, dtype=torch.bfloat16)
    attention_inline__attention_inline[b, hk, splits](qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], data['page_table'], data['seq_lens'], p, lse, cs, hq, hk, ql, page, maxp, window, cfg['sm_scale'] * cfg['qk_scale_factor'], splits, chunk, bm, bn, cfg['rope_theta'], max(16, tr.next_power_of_2(ql)), num_warps=4, num_stages=2)
    if b * ql >= 32 and splits <= 64:
        attention_inline__finish_heads[b * ql, tr.cdiv(hq, 4)](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), 4, num_warps=4)
    else:
        attention_inline__finish[b * ql, hq](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), num_warps=4)
    return y

@tr.jit
def attention_inline__prepare_heads(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    mask = h[:, None] < HQ + HK
    v = tl.load(QKV + t * stride + h[:, None] * 64 + d[None, :], mask, 0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 1) / 64 + 1e-05)
    v = v * r[:, None]
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h[:, None] * 64 + (d[None, :] ^ 1), mask, 0.0).to(tl.float32) * r[:, None]
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle)[None, :] + tl.where(d[None, :] % 2 == 0, -partner, partner) * libdevice.sin(angle)[None, :]
    v = v.to(tl.bfloat16).to(tl.float32)
    tl.store(Q + (t * HQ + h[:, None]) * 64 + d[None, :], v, h[:, None] < HQ)
    slot = tl.load(LOC + t)
    ks = tl.load(KS)
    vs = tl.load(VS)
    kv_mask = (h[:, None] >= HQ) & (h[:, None] < HQ + HK)
    val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h[:, None] - HQ) * 64 + d[None, :], kv_mask, 0.0).to(tl.float32)
    off = (slot * HK + h[:, None] - HQ) * 64 + d[None, :]
    tl.store(KC + off, v / ks, kv_mask)
    tl.store(VC + off, val / vs, kv_mask)

@tr.jit
def attention_inline__finish_heads(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h[:, None]) * SPLITS + s[None, :]) * BM + r[:, None]
    mask = (s[None, :] < SPLITS) & (head[:, None] < HQ)
    lse = tl.load(LSE + base, mask, -1000000.0)
    mx = tl.max(lse, 1)
    probability = tl.where(mask, tl.exp2(lse - mx[:, None]), 0.0)
    probability = probability / tl.sum(probability, 1)[:, None]
    part = tl.load(P + base[:, :, None] * 64 + d[None, None, :], mask[:, :, None], 0.0)
    a = tl.sum(part * probability[:, :, None], 1).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 1) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head[:, None] * 64 + d[None, :], head[:, None] < HQ, 0.0).to(tl.float32)
    out = a * norm[:, None] / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head[:, None]) * 64 + d[None, :], out, head[:, None] < HQ)
_ROPE = {}

def attention_inline__cos_sin(theta, device):
    key = (theta, str(device))
    if key not in _ROPE:
        inv = 1.0 / theta ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
        frequencies = torch.outer(torch.arange(1 << 21, device=device, dtype=torch.float32), inv)
        _ROPE[key] = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    return _ROPE[key]

def attention_kernel_fn(data):
    cfg = data['config']
    b = cfg['batch']
    ql = cfg['q_len']
    if b < 8 or (cfg['window'] and b < 32 and (ql == 1)):
        return attention_inline_kernel_fn(data)
    return attention__hybrid(data)

def _compact_prepare(weights):
    state=_layer_prepare(weights)
    device=weights['qkv_gate'].device
    single_sorted=torch.full((TOP_K,8),TOP_K,device=device,dtype=torch.int32)
    single_sorted[:,0]=torch.arange(TOP_K,device=device,dtype=torch.int32)
    state['single_sorted']=single_sorted.flatten()
    state['single_count']=torch.tensor([TOP_K*8],device=device,dtype=torch.int32)
    state['pxw']=state['pre_expert_norm']._get_effective_weight()
    fb=state['ffn_boundary']
    state['fbw']=(fb._post_norm._get_effective_weight(),fb._pre_norm._get_effective_weight(),fb._residual_add._alpha,fb._residual_add._beta)
    state['dopt']=dense_prepare({name:weights[name] for name in ('qkv_gate','o','pooler','unpooler','router')})
    return state

def _dense_op(x,name,state,n=None):
    if n is None:n=state['dopt'][name].shape[0]
    return dense_kernel_fn({'x':x,'config':{'layer':name,'n':n}},state['dopt'])


@tr.jit
def _px_norm_map(X,W,IDS,START,Y,LOCAL,H:tl.constexpr):
    t=tl.program_id(0);k=tl.arange(0,H);j=tl.arange(0,8)
    x=tl.load(X+t*H+k).to(tl.float32)
    w=tl.load(W+k).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,0)/H+1.e-5)
    tl.store(Y+t*H+k,x*r*w)
    ids=tl.load(IDS+t*8+j).to(tl.int32)
    local=(ids-tl.load(START))&255
    tl.store(LOCAL+t*8+j,tl.where(local<32,local,-1))

def _normalize_and_map(x,ids,start,w):
    y=torch.empty_like(x);local=torch.empty_like(ids)
    _px_norm_map[(x.shape[0],)](x,w,ids,start,y,local,x.shape[1],num_warps=4)
    return y,local

def _small_router(x,state):
    m,k=x.shape;y=torch.empty((m,256),device=x.device,dtype=torch.float32)
    dense__fullvec[(m,256,1)](x,state['router'],y,m,256,k,1,tr.next_power_of_2(k),num_warps=4)
    return y

@tr.jit
def _sliced_boundary(BRANCH,STREAM,POST,PRE,ALPHA,BETA,Y,NS,S:tl.constexpr,H:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,H)
    branch=tl.load(BRANCH+row*S+i,i<S,0.).to(tl.float32)
    post=tl.load(POST+i).to(tl.float32)
    branch=(branch*post).to(tl.bfloat16).to(tl.float32)
    r=tl.rsqrt(tl.sum(branch*branch,0)/H+1.e-8)
    branch=(branch*r).to(tl.bfloat16).to(tl.float32)
    stream=tl.load(STREAM+row*H+i).to(tl.float32)
    alpha=tl.load(ALPHA+i);beta=tl.load(BETA+i)
    newstream=alpha*stream+beta*branch
    tl.store(NS+row*H+i,newstream)
    prein=newstream.to(tl.bfloat16).to(tl.float32)
    r2=tl.rsqrt(tl.sum(prein*prein,0)/H+1.e-5)
    pre=tl.load(PRE+i).to(tl.float32)
    tl.store(Y+row*H+i,prein*r2*pre)

def _end_boundary(stream,branch,state):
    if branch.shape[1]==8192:return state['ffn_boundary'](stream,branch,branch_dtype=torch.bfloat16)
    y=torch.empty_like(stream,dtype=torch.bfloat16);ns=torch.empty_like(stream)
    post,pre,alpha,beta=state['fbw']
    _sliced_boundary[(stream.shape[0],)](branch,stream,post,pre,alpha,beta,y,ns,branch.shape[1],8192,num_warps=8,enable_fp_fusion=False)
    return y,ns,None

@tr.jit
def _compact_align(IDS,LOCAL,SORT,EXP,PAD,SLOTS:tl.constexpr,START:tl.constexpr,COUNT:tl.constexpr,BM:tl.constexpr,S:tl.constexpr,R:tl.constexpr,P:tl.constexpr,NB:tl.constexpr,MAXPAD:tl.constexpr,MAXEXP:tl.constexpr,WRITE_LOCAL:tl.constexpr=True):
 slot=tl.arange(0,S)
 ids=tl.load(IDS+slot,slot<SLOTS,-1)
 local=tl.where((ids>=START)&(ids<START+COUNT),ids-START,-1)
 if WRITE_LOCAL:
  tl.store(LOCAL+slot,local,slot<SLOTS)
 expert=tl.arange(0,R)
 match=(expert[:,None]==local[None,:])&(slot[None,:]<SLOTS)&(expert[:,None]<COUNT)
 counts=tl.sum(match.to(tl.int32),1)
 blocks=tl.cdiv(counts,BM)
 offsets=(tl.cumsum(blocks,0)-blocks)*BM
 within=tl.cumsum(match.to(tl.int32),1)-1
 padding=tl.arange(0,P)
 tl.store(SORT+padding,SLOTS,padding<MAXPAD)
 tl.store(EXP+padding,-1,padding<MAXEXP)
 tl.debug_barrier()
 tl.store(SORT+offsets[:,None]+within,tl.broadcast_to(slot[None,:],(R,S)),match)
 piece=tl.arange(0,NB)
 tl.store(EXP+offsets[:,None]//BM+piece[None,:],tl.broadcast_to(expert[:,None],(R,NB)),piece[None,:]<blocks[:,None])
 tl.store(PAD,tl.sum(blocks,0)*BM)

from functools import cache
_ALIGN_SRC='#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\ntemplate<int THREADS>\n__global__ void group_routes(const int* ids,int* local,int* sorted,int* experts,int* padded,int slots,int start,int count,int bm,int maxpad,int maxexp){\n __shared__ int counts[32],offsets[32],heads[32];\n extern __shared__ int route[];\n int t=threadIdx.x;\n if(t<32){counts[t]=0;heads[t]=0;}\n for(int j=t;j<maxpad;j+=THREADS)sorted[j]=slots;\n for(int j=t;j<maxexp;j+=THREADS)experts[j]=-1;\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){\n  int e=ids[j]-start;e=(e>=0&&e<count)?e:-1;\n  route[j]=e;local[j]=e;\n  if(e>=0)atomicAdd(counts+e,1);\n }\n __syncthreads();\n if(t<32){\n  int blocks=(counts[t]+bm-1)/bm,scan=blocks;\n  #pragma unroll\n  for(int delta=1;delta<32;delta*=2){int prev=__shfl_up_sync(0xffffffff,scan,delta);if(t>=delta)scan+=prev;}\n  offsets[t]=(scan-blocks)*bm;\n  if(t==31)padded[0]=scan*bm;\n  for(int j=0;j<blocks;j++)experts[scan-blocks+j]=t;\n }\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){int e=route[j];if(e>=0){int pos=atomicAdd(heads+e,1);sorted[offsets[e]+pos]=j;}}\n}\nvoid align_routes(TensorView ids,TensorView local,TensorView sorted,TensorView experts,TensorView padded,int64_t start,int64_t count,int64_t bm,int64_t threads){\n ffi::CUDADeviceGuard guard(ids.device().device_id);\n TVM_FFI_ICHECK(count<=32&&count>0&&ids.dtype().bits==32&&ids.numel()<=8192);\n int slots=ids.numel();auto stream=get_stream(ids.device());\n #define RUN(T) group_routes<T><<<1,T,slots*sizeof(int),stream>>>(static_cast<const int*>(ids.data_ptr()),static_cast<int*>(local.data_ptr()),static_cast<int*>(sorted.data_ptr()),static_cast<int*>(experts.data_ptr()),static_cast<int*>(padded.data_ptr()),slots,start,count,bm,sorted.numel(),experts.numel())\n if(threads==128){RUN(128);}else{RUN(256);}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(align_routes,align_routes);\n'
@cache
def _align_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_moe_cuda_align_v1',cuda_sources=_ALIGN_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])

def prepare(weights):
 state=_compact_prepare(weights)
 state['align_module']=_align_module()
 return state
