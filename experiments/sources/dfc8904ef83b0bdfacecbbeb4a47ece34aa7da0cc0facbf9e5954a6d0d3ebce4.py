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



import triton as tr
import triton.language as tl

@tr.jit
def _compact_align(IDS,LOCAL,SORT,EXP,PAD,SLOTS:tl.constexpr,START:tl.constexpr,COUNT:tl.constexpr,BM:tl.constexpr,S:tl.constexpr,R:tl.constexpr,P:tl.constexpr,NB:tl.constexpr,MAXPAD:tl.constexpr,MAXEXP:tl.constexpr):
 slot=tl.arange(0,S)
 ids=tl.load(IDS+slot,slot<SLOTS,-1)
 local=tl.where((ids>=START)&(ids<START+COUNT),ids-START,-1)
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
    single_sorted=torch.full((TOP_K,8),TOP_K,device=device,dtype=torch.int32)
    single_sorted[:,0]=torch.arange(TOP_K,device=device,dtype=torch.int32)
    single_count=torch.tensor([TOP_K*8],device=device,dtype=torch.int32)
    torch.cuda.synchronize()
    return {
        'single_sorted':single_sorted.flatten(),'single_count':single_count,
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _best_moe(data,state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    topk_weights=data['topk_weights'].to(torch.float32).contiguous()
    if num_tokens<=1024:
        block_size_m=16 if num_tokens>=128 else 8
        slots=num_tokens*TOP_K
        maxpad=tr.cdiv(slots+bank['count']*(block_size_m-1),block_size_m)*block_size_m
        maxexp=maxpad//block_size_m
        local_ids=torch.empty_like(data['topk_ids'],dtype=torch.int32)
        sorted_ids=torch.empty((maxpad,),device=x.device,dtype=torch.int32)
        block_experts=torch.empty((maxexp,),device=x.device,dtype=torch.int32)
        n_padded=torch.empty((1,),device=x.device,dtype=torch.int32)
        if True and num_tokens<=32:
            _compact_align[(1,)](data['topk_ids'],local_ids,sorted_ids,block_experts,n_padded,slots,cfg['local_start'],cfg['local_count'],block_size_m,tr.next_power_of_2(slots),tr.next_power_of_2(bank['count']),tr.next_power_of_2(maxpad),tr.next_power_of_2(tr.cdiv(slots,block_size_m)),maxpad,maxexp,num_warps=16 if slots>=512 else 8)
        else:
            state['align_module'].align_routes(data['topk_ids'],local_ids,sorted_ids,block_experts,n_padded,cfg['local_start'],cfg['local_count'],block_size_m,256)
    elif num_tokens==1:
        block_size_m=8
        local_ids=expert_map[data['topk_ids']].contiguous()
        sorted_ids=state['single_sorted'];block_experts=local_ids.flatten();n_padded=state['single_count']
    else:
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
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=(num_tokens > 1024) and expert_map is not None,
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

from functools import cache
_ALIGN_SRC='#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\ntemplate<int THREADS>\n__global__ void group_routes(const int* ids,int* local,int* sorted,int* experts,int* padded,int slots,int start,int count,int bm,int maxpad,int maxexp){\n __shared__ int counts[32],offsets[32],heads[32];\n extern __shared__ int route[];\n int t=threadIdx.x;\n if(t<32){counts[t]=0;heads[t]=0;}\n for(int j=t;j<maxpad;j+=THREADS)sorted[j]=slots;\n for(int j=t;j<maxexp;j+=THREADS)experts[j]=-1;\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){\n  int e=ids[j]-start;e=(e>=0&&e<count)?e:-1;\n  route[j]=e;local[j]=e;\n  if(e>=0)atomicAdd(counts+e,1);\n }\n __syncthreads();\n if(t<32){\n  int blocks=(counts[t]+bm-1)/bm,scan=blocks;\n  #pragma unroll\n  for(int delta=1;delta<32;delta*=2){int prev=__shfl_up_sync(0xffffffff,scan,delta);if(t>=delta)scan+=prev;}\n  offsets[t]=(scan-blocks)*bm;\n  if(t==31)padded[0]=scan*bm;\n  for(int j=0;j<blocks;j++)experts[scan-blocks+j]=t;\n }\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){int e=route[j];if(e>=0){int pos=atomicAdd(heads+e,1);sorted[offsets[e]+pos]=j;}}\n}\nvoid align_routes(TensorView ids,TensorView local,TensorView sorted,TensorView experts,TensorView padded,int64_t start,int64_t count,int64_t bm,int64_t threads){\n ffi::CUDADeviceGuard guard(ids.device().device_id);\n TVM_FFI_ICHECK(count<=32&&count>0&&ids.dtype().bits==32&&ids.numel()<=8192);\n int slots=ids.numel();auto stream=get_stream(ids.device());\n #define RUN(T) group_routes<T><<<1,T,slots*sizeof(int),stream>>>(static_cast<const int*>(ids.data_ptr()),static_cast<int*>(local.data_ptr()),static_cast<int*>(sorted.data_ptr()),static_cast<int*>(experts.data_ptr()),static_cast<int*>(padded.data_ptr()),slots,start,count,bm,sorted.numel(),experts.numel())\n if(threads==128){RUN(128);}else{RUN(256);}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(align_routes,align_routes);\n'
@cache
def _align_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_moe_cuda_align_v1',cuda_sources=_ALIGN_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])

def _prmt_base_prepare(weights):
 state=_base_prepare(weights)
 state['align_module']=_align_module()
 return state

from functools import cache
_WARP_SRC='#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\n__device__ __forceinline__ float fp4_value(unsigned code){\n unsigned mag=code&7;\n unsigned bits=(mag<2?mag*0x3f00u:0x3f00u+(mag<<6))|((code&8)<<12);\n return __uint_as_float(bits<<16);\n}\n__device__ __forceinline__ float fp8_value(unsigned char bits){\n __nv_fp8_e4m3 f;f.__x=bits;return static_cast<float>(f);\n}\n// Exact FP4 E2M1 to BF16 register lookup. NVFP4 storage is unchanged.\n__device__ __forceinline__ uint32_t okc_prmt(uint32_t a,uint32_t b,uint32_t select){\n uint32_t out;asm("prmt.b32 %0, %1, %2, %3;":"=r"(out):"r"(a),"r"(b),"r"(select));return out;\n}\n__device__ __forceinline__ void fp4x4_bf16(uint32_t q,uint32_t& a,uint32_t& b){\n uint32_t sel=q&0x7777u;\n uint32_t hi=okc_prmt(0x3f3f3f00u,0x40404040u,sel);\n uint32_t lo=okc_prmt(0xc0800000u,0xc0804000u,sel);\n a=okc_prmt(lo,hi,0x5140u)|((q&8u)<<12)|((q&0x80u)<<24);\n b=okc_prmt(lo,hi,0x7362u)|((q&0x800u)<<4)|((q&0x8000u)<<16);\n}\n__device__ __forceinline__ void fp4x8_bf16(uint32_t q,uint32_t& a,uint32_t& b,uint32_t& c,uint32_t& d){\n fp4x4_bf16(q,a,b);fp4x4_bf16(q>>16,c,d);\n}\n\ntemplate<int WARPS,bool GATED>\n__global__ void warp_nv_mv(const __nv_bfloat16* __restrict__ X,const int* __restrict__ IDS,const unsigned char* __restrict__ W,const unsigned char* __restrict__ SF,const float* __restrict__ GS,__nv_bfloat16* __restrict__ Y,int N,int K,int topk,int start,int count){\n int slot=blockIdx.y,expert=IDS[slot];\n if(expert<start || expert>=start+count)return;\n int n=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x%32;\n if(n>=N)return;\n int weight_n=GATED?2*N:N;\n const unsigned char* wr=W+((int64_t)expert*weight_n+n)*(K/2);\n const unsigned char* sr=SF+((int64_t)expert*weight_n+n)*(K/16);\n const __nv_bfloat16* xr=X+(slot/topk)*K;\n float gate=0.f,up=0.f;\n\n #pragma unroll 2\n for(int k=lane*16;k<K;k+=512){\n  uint2 packed=*reinterpret_cast<const uint2*>(wr+k/2);\n  float sf=fp8_value(sr[k/16]);\n  uint2 packed_up={0,0};float sf_up=0;\n  if constexpr(GATED){packed_up=*reinterpret_cast<const uint2*>(wr+(int64_t)N*(K/2)+k/2);sf_up=fp8_value(sr[(int64_t)N*(K/16)+k/16]);}\n  uint32_t gw[8],uw[8];\n  fp4x8_bf16(packed.x,gw[0],gw[1],gw[2],gw[3]);\n  fp4x8_bf16(packed.y,gw[4],gw[5],gw[6],gw[7]);\n  if constexpr(GATED){\n   fp4x8_bf16(packed_up.x,uw[0],uw[1],uw[2],uw[3]);\n   fp4x8_bf16(packed_up.y,uw[4],uw[5],uw[6],uw[7]);\n  }\n  float group_g=0.f,group_u=0.f;\n  #pragma unroll\n  for(int j=0;j<8;j++){\n   float2 xv=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xr+k+j*2));\n   float2 gv=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gw[j]));\n   group_g=fmaf(xv.x,gv.x,group_g);group_g=fmaf(xv.y,gv.y,group_g);\n   if constexpr(GATED){float2 uv=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&uw[j]));group_u=fmaf(xv.x,uv.x,group_u);group_u=fmaf(xv.y,uv.y,group_u);}\n  }\n  gate=fmaf(group_g,sf,gate);\n  if constexpr(GATED)up=fmaf(group_u,sf_up,up);\n\n }\n #pragma unroll\n for(int delta=16;delta;delta/=2){gate+=__shfl_down_sync(0xffffffff,gate,delta);if constexpr(GATED)up+=__shfl_down_sync(0xffffffff,up,delta);}\n if(lane==0){\n  float gs=GS[expert];\n  float value=gate*gs;\n  if constexpr(GATED){float g=__bfloat162float(__float2bfloat16_rn(value));float u=__bfloat162float(__float2bfloat16_rn(up*gs));value=g/(1.f+expf(-g))*u;}\n  Y[slot*N+n]=__float2bfloat16_rn(value);\n }\n}\nvoid warp_nv_run(TensorView x,TensorView ids,TensorView w,TensorView sf,TensorView gs,TensorView y,int64_t topk,int64_t start,int64_t count,int64_t warps,bool gated){\n ffi::CUDADeviceGuard guard(x.device().device_id);\n int N=y.size(1),K=x.size(1),slots=y.size(0);\n TVM_FFI_ICHECK(K%256==0 && N%16==0);\n auto X=static_cast<const __nv_bfloat16*>(x.data_ptr());auto IDS=static_cast<const int*>(ids.data_ptr());auto W=static_cast<const unsigned char*>(w.data_ptr());auto SF=static_cast<const unsigned char*>(sf.data_ptr());auto GS=static_cast<const float*>(gs.data_ptr());auto Y=static_cast<__nv_bfloat16*>(y.data_ptr());auto stream=get_stream(x.device());\n #define LAUNCH(B,G) warp_nv_mv<B,G><<<dim3((N+B-1)/B,slots),B*32,0,stream>>>(X,IDS,W,SF,GS,Y,N,K,topk,start,count)\n if(gated){if(warps==4){LAUNCH(4,true);}else if(warps==8){LAUNCH(8,true);}else{LAUNCH(16,true);}}\n else{if(warps==4){LAUNCH(4,false);}else if(warps==8){LAUNCH(8,false);}else{LAUNCH(16,false);}}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(warp_nv_run,warp_nv_run);\n\n__global__ void probe_fp4_prmt(uint32_t* out){\n uint32_t q=blockIdx.x*blockDim.x+threadIdx.x;\n if(q<65536){uint32_t a,b;fp4x4_bf16(q,a,b);out[q*2]=a;out[q*2+1]=b;}\n}\nvoid prmt_probe(TensorView out){ffi::CUDADeviceGuard guard(out.device().device_id);probe_fp4_prmt<<<256,256,0,get_stream(out.device())>>>(static_cast<uint32_t*>(out.data_ptr()));TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(prmt_probe,prmt_probe);\n'
@cache
def _warp_module():
    import pathlib, importlib.util
    from tvm_ffi.cpp import load_inline
    root = pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations))) / 'data'
    return load_inline('okc_moe_prmt_gemv_v2', cuda_sources=_WARP_SRC, backend='cuda', extra_cuda_cflags=['-std=c++17', '-O3', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-gencode=arch=compute_90a,code=sm_90a'], extra_include_paths=[str(root / 'include'), str(root / 'csrc'), str(root / 'cutlass/include'), str(root / 'cutlass/tools/util/include')])

def prepare(weights):
    state = _prmt_base_prepare(weights)
    state['nv'] = {name: weights[name] for name in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs')}
    return state

def kernel_fn(data, state):
    cfg = data['config']
    m = cfg['bs']
    h = cfg['d_hidden']
    i = cfg['d_expert']
    topk = cfg['top_k']
    if m > 8:
        return _best_moe(data, state)
    bank = state['banks'][cfg['local_start'], cfg['local_count']]
    local = bank['expert_map'][data['topk_ids']].contiguous()
    x = data['hidden_states']
    nv = state['nv']
    activated = torch.empty((m * topk, i), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((m * topk, h), device=x.device, dtype=torch.bfloat16)
    mod = _warp_module()
    mod.warp_nv_run(x, data['topk_ids'], nv['w13'], nv['w13_sf'], nv['w13_gs'], activated, topk, cfg['local_start'], cfg['local_count'], 4, True)
    mod.warp_nv_run(activated, data['topk_ids'], nv['w2'], nv['w2_sf'], nv['w2_gs'], y, 1, cfg['local_start'], cfg['local_count'], 4, False)
    return fused_post_expert_norm_weighted_sum(y.view(m, topk, h), data['topk_weights'], norm_weight=state['norm_w'], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local)
