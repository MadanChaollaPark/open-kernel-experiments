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


def prepare(weights):
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


def kernel_fn(data, state):
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

    block_size_m = 8
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
    cache1 = _tuned_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = _tuned_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)

# Marlin launch wrapper adapted from SGLang, Apache-2.0.
# Copyright (C) Marlin.2024 Elias Frantar; modifications by Neural Magic.
# https://www.apache.org/licenses/LICENSE-2.0
from sglang.kernels.jit.utils import cache_once, load_jit
_MAX_THREAD_N=256
_CUDA_SOURCE='#include <tvm/ffi/function.h>\n#include "gemm/marlin_moe/moe_wna16_marlin.cuh"\nnamespace sglang {\ntemplate<typename A,typename B> auto okc_div_ceil(A a,B b){return (a+b-1)/b;}\nvoid tuned_marlin(\n    tvm::ffi::TensorView a,\n    tvm::ffi::TensorView c,\n    tvm::ffi::TensorView b_q_weight,\n    tvm::ffi::TensorView b_bias,\n    tvm::ffi::TensorView b_scales,\n    tvm::ffi::TensorView global_scale,\n    tvm::ffi::TensorView b_zeros,\n    tvm::ffi::TensorView g_idx,\n    tvm::ffi::TensorView perm,\n    tvm::ffi::TensorView workspace,\n    tvm::ffi::TensorView sorted_token_ids,\n    tvm::ffi::TensorView expert_ids,\n    tvm::ffi::TensorView num_tokens_post_padded,\n    tvm::ffi::TensorView topk_weights,\n    tvm::ffi::TensorView a_tmp,\n    tvm::ffi::TensorView c_tmp,\n    int64_t moe_block_size,\n    int64_t top_k,\n    bool mul_topk_weights,\n    bool is_ep,\n    int64_t b_q_type_id,\n    int64_t size_m,\n    int64_t size_n,\n    int64_t size_k,\n    bool has_act_order,\n    bool has_bias,\n    bool is_k_full,\n    bool has_zp,\n    int64_t num_groups,\n    int64_t group_size,\n    bool use_atomic_add,\n    bool use_fp32_reduce,\n    bool is_zp_float, int64_t requested_k, int64_t requested_n, int64_t requested_threads, int64_t requested_blocks) {\n  using scalar_t=__nv_bfloat16;\n  constexpr bool kIsEP=true;\n  constexpr bool kHasBias=false;\n  auto tuned_mm=[&](\n    const void* A,\n    const void* B,\n    void* C,\n    void* C_tmp,\n    void* b_bias,\n    void* s,\n    void* s2,\n    void* zp,\n    void* g_idx,\n    void* perm,\n    void* a_tmp,\n    void* sorted_token_ids,\n    void* expert_ids,\n    void* num_tokens_past_padded,\n    void* topk_weights,\n    int moe_block_size,\n    int top_k,\n    bool mul_topk_weights,\n    bool is_ep,\n    int prob_m,\n    int prob_n,\n    int prob_k,\n    void* workspace,\n    host::ScalarType const& q_type,\n    bool has_bias,\n    bool has_act_order,\n    bool is_k_full,\n    bool has_zp,\n    int num_groups,\n    int group_size,\n    int dev,\n    cudaStream_t stream,\n    int thread_k,\n    int thread_n,\n    int sms,\n    bool use_atomic_add,\n    bool use_fp32_reduce,\n    bool is_zp_float) {\n  using namespace device::marlin_moe;\n  int thread_m_blocks = okc_div_ceil(moe_block_size, 16);\n  bool m_block_size_8 = moe_block_size == 8;\n\n  if (has_zp) {\n    host::RuntimeCheck(\n        q_type == host::kU4 || q_type == host::kU8, "q_type must be u4 or u8 when has_zp = True. Got = ", q_type.str());\n  } else {\n    host::RuntimeCheck(\n        q_type == host::kU4B8 || q_type == host::kU8B128 || q_type == host::kFE4M3fn || q_type == host::kFE2M1f,\n        "q_type must be uint4b8, uint8b128, float8_e4m3fn or float4_e2m1f when "\n        "has_zp = False. Got = ",\n        q_type.str());\n  }\n\n  host::RuntimeCheck(\n      prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");\n\n  int group_blocks = 0;\n  if (has_act_order) {\n    if (is_k_full) {\n      host::RuntimeCheck(group_size != -1);\n      group_blocks = group_size / 16;\n      host::RuntimeCheck(\n          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);\n    } else {\n      host::RuntimeCheck(group_size == 0);\n      group_blocks = 0;\n    }\n  } else {\n    if (group_size == -1) {\n      group_blocks = -1;\n    } else {\n      group_blocks = group_size / 16;\n      host::RuntimeCheck(\n          prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);\n    }\n  }\n\n  int num_bits = q_type.size_bits();\n  const int4* A_ptr = (const int4*)A;\n  const int4* B_ptr = (const int4*)B;\n  int4* C_ptr = (int4*)C;\n  int4* C_tmp_ptr = (int4*)C_tmp;\n  const int4* bias_ptr = (const int4*)b_bias;\n  const int4* s_ptr = (const int4*)s;\n  const uint16_t* s2_ptr = (const uint16_t*)s2;\n  const int4* zp_ptr = (const int4*)zp;\n  const int* g_idx_ptr = (const int*)g_idx;\n  const int* perm_ptr = (const int*)perm;\n  int4* a_tmp_ptr = (int4*)a_tmp;\n  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;\n  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;\n  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;\n  const float* topk_weights_ptr = (const float*)topk_weights;\n  int* locks = (int*)workspace;\n\n  if (has_act_order) {\n    // Permute A columns\n    auto perm_kernel = permute_cols_kernel<8>;\n    if (moe_block_size == 8) {\n    } else if (moe_block_size == 16)\n      perm_kernel = permute_cols_kernel<16>;\n    else if (moe_block_size == 32)\n      perm_kernel = permute_cols_kernel<32>;\n    else if (moe_block_size == 48)\n      perm_kernel = permute_cols_kernel<48>;\n    else if (moe_block_size == 64)\n      perm_kernel = permute_cols_kernel<64>;\n    else\n      host::Panic("unsupported moe_block_size ", moe_block_size);\n\n    // clang-format off\n    perm_kernel<<<sms, default_threads, 0, stream>>>(\n        A_ptr, perm_ptr, a_tmp_ptr, sorted_token_ids_ptr, expert_ids_ptr,\n        num_tokens_past_padded_ptr, prob_m, prob_k, top_k);\n    // clang-format on\n    A_ptr = a_tmp_ptr;\n    prob_m = prob_m * top_k;\n    top_k = 1;\n\n    // If we have a full K, then we can run the non-act-order version of Marlin\n    // (since the weight rows are reordered by increasing group ids, and by\n    // having a full K, we have full original groups)\n    if (is_k_full) has_act_order = false;\n  }\n\n  int max_shared_mem = 0;\n  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));\n  host::RuntimeCheck(max_shared_mem > 0);\n\n  exec_config_t exec_cfg{static_cast<int>(requested_blocks),thread_config_t{static_cast<int>(requested_k),static_cast<int>(requested_n),static_cast<int>(requested_threads)}};\n  thread_config_t thread_tfg=exec_cfg.tb_cfg;\n  int num_threads = thread_tfg.num_threads;\n  thread_k = thread_tfg.thread_k;\n  thread_n = thread_tfg.thread_n;\n  int blocks = sms * exec_cfg.blocks_per_sm;\n  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - kSharedMemoryLaunchReserve;\n\n  int thread_k_blocks = thread_k / 16;\n  int thread_n_blocks = thread_n / 16;\n\n  host::RuntimeCheck(\n      is_valid_config(\n          thread_tfg,\n          m_block_size_8,\n          thread_m_blocks,\n          prob_m,\n          prob_n,\n          prob_k,\n          num_bits,\n          group_size,\n          has_act_order,\n          is_k_full,\n          has_zp,\n          is_zp_float,\n          max_shared_mem),\n      "Invalid thread config: thread_m_blocks = ",\n      thread_m_blocks,\n      ", thread_k = ",\n      thread_tfg.thread_k,\n      ", thread_n = ",\n      thread_tfg.thread_n,\n      ", num_threads = ",\n      thread_tfg.num_threads,\n      " for MKN = [",\n      prob_m,\n      ", ",\n      prob_k,\n      ", ",\n      prob_n,\n      "] and num_bits = ",\n      num_bits,\n      ", group_size = ",\n      group_size,\n      ", has_act_order = ",\n      has_act_order,\n      ", is_k_full = ",\n      is_k_full,\n      ", has_zp = ",\n      has_zp,\n      ", is_zp_float = ",\n      is_zp_float,\n      ", max_shared_mem = ",\n      max_shared_mem);\n\n  host::RuntimeCheck(q_type==host::kFE2M1f && group_blocks==1 && !has_act_order && !has_zp && thread_m_blocks==1,"Unsupported tuned NVFP4 configuration");\n  auto kernel=MarlinDefault;\n  if (m_block_size_8==true && thread_k==128 && thread_n==64 && num_threads==128)\n    kernel=Marlin<scalar_t,host::kFE2M1f.id(),host::kFE4M3fn.id(),128,1,4,8,true,pipe_stages,1,false,kIsEP,kHasBias>;\n  if (m_block_size_8==true && thread_k==64 && thread_n==256 && num_threads==256)\n    kernel=Marlin<scalar_t,host::kFE2M1f.id(),host::kFE4M3fn.id(),256,1,16,4,true,pipe_stages,1,false,kIsEP,kHasBias>;\n  if (m_block_size_8==false && thread_k==128 && thread_n==64 && num_threads==128)\n    kernel=Marlin<scalar_t,host::kFE2M1f.id(),host::kFE4M3fn.id(),128,1,4,8,false,pipe_stages,1,false,kIsEP,kHasBias>;\n  if (m_block_size_8==false && thread_k==64 && thread_n==256 && num_threads==256)\n    kernel=Marlin<scalar_t,host::kFE2M1f.id(),host::kFE4M3fn.id(),256,1,16,4,false,pipe_stages,1,false,kIsEP,kHasBias>;\n\n  if (kernel == MarlinDefault) {\n    host::Panic(\n        "Unsupported shapes: MNK = [",\n        prob_m,\n        ", ",\n        prob_n,\n        ", ",\n        prob_k,\n        "]",\n        ", has_act_order = ",\n        has_act_order,\n        ", num_groups = ",\n        num_groups,\n        ", group_size = ",\n        group_size,\n        ", thread_m_blocks = ",\n        thread_m_blocks,\n        ", thread_n_blocks = ",\n        thread_n_blocks,\n        ", thread_k_blocks = ",\n        thread_k_blocks,\n        ", num_bits = ",\n        num_bits);\n  }\n\n  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));\n  // clang-format off\n  kernel<<<blocks, num_threads, max_shared_mem, stream>>>(\n      A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, s_ptr, s2_ptr, zp_ptr, g_idx_ptr,\n      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr,\n      topk_weights_ptr, top_k, mul_topk_weights, is_ep, num_groups, prob_m,\n      prob_n, prob_k, locks, has_bias, use_atomic_add, use_fp32_reduce, max_shared_mem);\n  // clang-format on\n};\n  using namespace host;\n\n  RuntimeCheck(is_ep == kIsEP, "is_ep does not match the compiled Marlin MoE specialization");\n  RuntimeCheck(has_bias == kHasBias, "has_bias does not match the compiled Marlin MoE specialization");\n\n  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);\n  int pack_factor = 32 / b_q_type.size_bits();\n\n  if (moe_block_size != 8) {\n    RuntimeCheck(moe_block_size % 16 == 0, "unsupported moe_block_size=", moe_block_size);\n    RuntimeCheck(moe_block_size >= 16 && moe_block_size <= 64, "unsupported moe_block_size=", moe_block_size);\n  }\n\n  // Verify A\n  RuntimeCheck(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0), ", size_m = ", size_m);\n  RuntimeCheck(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1), ", size_k = ", size_k);\n\n  // Verify B\n  RuntimeCheck(\n      size_k % device::marlin::tile_size == 0,\n      "size_k = ",\n      size_k,\n      " is not divisible by tile_size = ",\n      device::marlin::tile_size);\n  RuntimeCheck(\n      (size_k / device::marlin::tile_size) == b_q_weight.size(1),\n      "Shape mismatch: b_q_weight.size(1) = ",\n      b_q_weight.size(1),\n      ", size_k = ",\n      size_k,\n      ", tile_size = ",\n      device::marlin::tile_size);\n  RuntimeCheck(\n      b_q_weight.size(2) % device::marlin::tile_size == 0,\n      "b_q_weight.size(2) = ",\n      b_q_weight.size(2),\n      " is not divisible by tile_size = ",\n      device::marlin::tile_size);\n  int64_t actual_size_n = (b_q_weight.size(2) / device::marlin::tile_size) * pack_factor;\n  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);\n\n  // Verify device and strides\n  auto device = SymbolicDevice{};\n  device.set_options<kDLCUDA>();\n  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);\n\n  device.verify(b_q_weight.device());\n  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");\n\n  device.verify(b_scales.device());\n  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");\n\n  // thread_k, thread_n, sms\n  int thread_k = -1;\n  int thread_n = -1;\n  int sms = -1;\n  DLDevice dl_device = device.unwrap();\n  int dev = dl_device.device_id;\n  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);\n  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));\n\n  // Verify c (allocation done in Python)\n  device.verify(c.device());\n  RuntimeCheck(c.is_contiguous(), "c is not contiguous");\n  RuntimeCheck(\n      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);\n  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);\n\n  // Alloc c_tmp: SKIP, done in Python\n\n  // Detect groupsize: b_scales rank and dims\n  RuntimeCheck(b_scales.dim() == 3, "b_scales rank = ", b_scales.dim(), " is not 3");\n  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);\n  RuntimeCheck(\n      b_scales.size(1) == num_groups, "b_scales dim 1 = ", b_scales.size(1), " is not num_groups = ", num_groups);\n\n  // Validate g_idx, perm (Optional unwrap done in Python; empty tensors when absent)\n  if (g_idx.size(g_idx.dim() - 1) > 0 && perm.size(perm.dim() - 1) > 0) {\n    device.verify(g_idx.device());\n    RuntimeCheck(g_idx.is_contiguous(), "g_idx is not contiguous");\n    device.verify(perm.device());\n    RuntimeCheck(perm.is_contiguous(), "perm is not contiguous");\n\n    int64_t g_idx_last = g_idx.size(g_idx.dim() - 1);\n    int64_t perm_last = perm.size(perm.dim() - 1);\n    RuntimeCheck(\n        (g_idx_last == 0 && perm_last == 0) || (g_idx_last == size_k && perm_last == size_k),\n        "Unexpected g_idx.size(-1) = ",\n        g_idx_last,\n        " and perm.size(-1) = ",\n        perm_last,\n        ", where size_k = ",\n        size_k);\n  }\n  // has_act_order derivation: SKIP (passed as param)\n\n  // Verify group_size consistency\n  if (has_act_order) {\n    // SKIP: a_tmp allocation done in Python\n    if (is_k_full) {\n      RuntimeCheck(num_groups > 1, "For act_order, num_groups must be > 1");\n      RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);\n    }\n  } else {\n    if (num_groups > 1) {\n      RuntimeCheck(\n          size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);\n    }\n  }\n\n  // Verify global_scale (Optional unwrap done in Python)\n  int64_t global_scale_size = global_scale.size(0);\n  if (global_scale_size > 0) {\n    RuntimeCheck(b_q_type == kFE2M1f && group_size == 16, "global_scale can only be used for nvfp4 format.");\n  } else {\n    RuntimeCheck(\n        !(b_q_type == kFE2M1f && group_size == 16), "the global_scale parameter must be passed for nvfp4 format.");\n  }\n\n  // Verify b_bias (Optional unwrap done in Python)\n  if (has_bias) {\n    device.verify(b_bias.device());\n    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");\n    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(0) != size_n");\n    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");\n  }\n\n  // b_zeros Optional unwrap + has_zp derivation: SKIP (done in Python)\n\n  // Verify b_q_type vs has_zp\n  if (has_zp) {\n    device.verify(b_zeros.device());\n    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");\n    RuntimeCheck(\n        b_q_type == kU4 || b_q_type == kU8, "b_q_type must be u4 or u8 when has_zp = True. Got = ", b_q_type.str());\n  } else {\n    RuntimeCheck(\n        b_q_type == kU4B8 || b_q_type == kU8B128 || b_q_type == kFE4M3fn || b_q_type == kFE2M1f,\n        "b_q_type must be uint4b8, uint8b128, float8_e4m3fn or "\n        "float4_e2m1f when "\n        "has_zp = False. Got = ",\n        b_q_type.str());\n  }\n\n  if (has_zp && is_zp_float) {\n    RuntimeCheck(\n        std::is_same<scalar_t, fp16_t>::value,\n        "Computation type must be float16 (half) when using float zero "\n        "points.");\n  }\n\n  if (b_q_type == kFE2M1f) {\n    RuntimeCheck(\n        group_size == 16 || group_size == 32,\n        "float4_e2m1f only supports group_size == 16 (NVFP4) or group_size == 32 (MXFP4). Got group_size = ",\n        group_size);\n    RuntimeCheck(\n        group_size != 32 || std::is_same<scalar_t, nv_bfloat16>::value,\n        "MXFP4 Marlin with E8M0 scales is only instantiated for bfloat16 activations.");\n  }\n\n  // Verify b_zeros\n  if (has_zp) {\n    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");\n    if (is_zp_float) {\n      RuntimeCheck(b_zeros.size(2) == size_n, "b_zeros dim 2 = ", b_zeros.size(2), " is not size_n = ", size_n);\n      RuntimeCheck(\n          num_groups == b_zeros.size(1), "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);\n      RuntimeCheck(num_groups != -1, "num_groups must be != -1");\n    } else {\n      RuntimeCheck(\n          b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);\n      RuntimeCheck(\n          b_zeros.size(2) == size_n / pack_factor,\n          "b_zeros dim 2 = ",\n          b_zeros.size(2),\n          " is not size_n / pack_factor = ",\n          size_n / pack_factor);\n    }\n  }\n\n  // Verify workspace size\n  RuntimeCheck(\n      size_n % device::marlin::min_thread_n == 0,\n      "size_n = ",\n      size_n,\n      ", is not divisible by min_thread_n = ",\n      device::marlin::min_thread_n);\n\n  int64_t max_n_tiles = size_n / device::marlin::min_thread_n;\n  int64_t min_workspace_size =\n      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);\n  RuntimeCheck(\n      workspace.size(0) >= min_workspace_size,\n      "workspace.numel = ",\n      workspace.size(0),\n      " is below min_workspace_size = ",\n      min_workspace_size);\n\n  // Early return for zero-size M (moved after all validation)\n  if (size_m == 0) return;\n\n  tuned_mm(\n      a.data_ptr(),\n      b_q_weight.data_ptr(),\n      c.data_ptr(),\n      c_tmp.data_ptr(),\n      b_bias.data_ptr(),\n      b_scales.data_ptr(),\n      global_scale.data_ptr(),\n      b_zeros.data_ptr(),\n      g_idx.data_ptr(),\n      perm.data_ptr(),\n      a_tmp.data_ptr(),\n      sorted_token_ids.data_ptr(),\n      expert_ids.data_ptr(),\n      num_tokens_post_padded.data_ptr(),\n      topk_weights.data_ptr(),\n      static_cast<int>(moe_block_size),\n      static_cast<int>(top_k),\n      mul_topk_weights,\n      is_ep,\n      static_cast<int>(size_m),\n      static_cast<int>(size_n),\n      static_cast<int>(size_k),\n      workspace.data_ptr(),\n      b_q_type,\n      has_bias,\n      has_act_order,\n      is_k_full,\n      has_zp,\n      static_cast<int>(num_groups),\n      static_cast<int>(group_size),\n      dev,\n      stream,\n      thread_k,\n      thread_n,\n      sms,\n      use_atomic_add,\n      use_fp32_reduce,\n      is_zp_float);\n}\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(tuned_marlin,sglang::tuned_marlin);\n'
@cache_once
def _jit_tuned():
    from tvm_ffi.cpp import load_inline
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    return load_inline('okc_moe_launch_tune_v3',cuda_sources=_CUDA_SOURCE,
        extra_cuda_cflags=['-std=c++20','-O3','--expt-relaxed-constexpr','-gencode=arch=compute_90a,code=sm_90a','-DSGL_CUDA_ARCH=900'],
        extra_include_paths=[str(KERNEL_PATH/'include'),str(KERNEL_PATH/'csrc')])

def _or_empty(t, device, dtype):
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)
def _tuned_gemm(a, c_or_none, b_q_weight, b_bias_or_none, b_scales, global_scale_or_none, b_zeros_or_none, g_idx_or_none, perm_or_none, workspace, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, moe_block_size, top_k, mul_topk_weights, is_ep, b_q_type, size_m, size_n, size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False):
    device = a.device
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)
    if size_m == 0:
        return c
    has_act_order = g_idx_or_none is not None and perm_or_none is not None and (g_idx_or_none.numel() > 0) and (perm_or_none.numel() > 0) and (g_idx_or_none.size(-1) > 0) and (perm_or_none.size(-1) > 0)
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0
    has_bias = b_bias_or_none is not None
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    elif num_groups > 1:
        group_size = size_k // num_groups
    else:
        group_size = -1
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)
    if use_fp32_reduce and (not use_atomic_add):
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_c_tmp_size = min(size_n * sorted_token_ids.size(0), sms * 4 * moe_block_size * _MAX_THREAD_N)
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)
    module = _jit_tuned()
    module.tuned_marlin(a, c, b_q_weight, b_bias_t, b_scales, global_scale_t, b_zeros_t, g_idx_t, perm_t, workspace, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, a_tmp, c_tmp, moe_block_size, top_k, mul_topk_weights, is_ep, b_q_type.id, size_m, size_n, size_k, has_act_order, has_bias, is_k_full, has_zp, num_groups, group_size, use_atomic_add, use_fp32_reduce, is_zp_float, _TILE_K, _TILE_N, _THREADS, _BLOCKS)
    return c

_TILE_K=64
_TILE_N=256
_THREADS=256
_BLOCKS=4
