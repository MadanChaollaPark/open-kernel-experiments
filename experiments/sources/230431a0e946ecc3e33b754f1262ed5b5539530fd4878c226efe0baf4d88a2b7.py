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

def _align_base_prepare(weights):
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


def _best_moe(data, state):
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

from functools import cache
_ALIGN_SRC='#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\ntemplate<int THREADS>\n__global__ void group_routes(const int* ids,int* local,int* sorted,int* experts,int* padded,int slots,int start,int count,int bm,int maxpad,int maxexp){\n __shared__ int counts[32],offsets[32],heads[32];\n extern __shared__ int route[];\n int t=threadIdx.x;\n if(t<32){counts[t]=0;heads[t]=0;}\n for(int j=t;j<maxpad;j+=THREADS)sorted[j]=slots;\n for(int j=t;j<maxexp;j+=THREADS)experts[j]=-1;\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){\n  int e=ids[j]-start;e=(e>=0&&e<count)?e:-1;\n  route[j]=e;local[j]=e;\n  if(e>=0)atomicAdd(counts+e,1);\n }\n __syncthreads();\n if(t<32){\n  int blocks=(counts[t]+bm-1)/bm,scan=blocks;\n  #pragma unroll\n  for(int delta=1;delta<32;delta*=2){int prev=__shfl_up_sync(0xffffffff,scan,delta);if(t>=delta)scan+=prev;}\n  offsets[t]=(scan-blocks)*bm;\n  if(t==31)padded[0]=scan*bm;\n  for(int j=0;j<blocks;j++)experts[scan-blocks+j]=t;\n }\n __syncthreads();\n for(int j=t;j<slots;j+=THREADS){int e=route[j];if(e>=0){int pos=atomicAdd(heads+e,1);sorted[offsets[e]+pos]=j;}}\n}\nvoid align_routes(TensorView ids,TensorView local,TensorView sorted,TensorView experts,TensorView padded,int64_t start,int64_t count,int64_t bm,int64_t threads){\n ffi::CUDADeviceGuard guard(ids.device().device_id);\n TVM_FFI_ICHECK(count<=32&&count>0&&ids.dtype().bits==32&&ids.numel()<=8192);\n int slots=ids.numel();auto stream=get_stream(ids.device());\n #define RUN(T) group_routes<T><<<1,T,slots*sizeof(int),stream>>>(static_cast<const int*>(ids.data_ptr()),static_cast<int*>(local.data_ptr()),static_cast<int*>(sorted.data_ptr()),static_cast<int*>(experts.data_ptr()),static_cast<int*>(padded.data_ptr()),slots,start,count,bm,sorted.numel(),experts.numel())\n if(threads==128){RUN(128);}else{RUN(256);}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(align_routes,align_routes);\n'
@cache
def _align_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_moe_cuda_align_v1',cuda_sources=_ALIGN_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])

def _base_prepare(weights):
 state=_align_base_prepare(weights)
 state['align_module']=_align_module()
 return state

from functools import cache
_WARP_SRC='#include <cuda_bf16.h>\n#include <cuda_fp8.h>\n#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\n__device__ __forceinline__ float fp4_value(unsigned code){\n unsigned mag=code&7;unsigned bits=(mag<2?mag*0x3f00u:0x3f00u+(mag<<6))|((code&8)<<12);\n return __uint_as_float(bits<<16);\n}\n__device__ __forceinline__ float fp8_value(unsigned char bits){__nv_fp8_e4m3 f;f.__x=bits;return static_cast<float>(f);}\n__device__ __forceinline__ uint16_t fp8_to_bf16_bits(unsigned char bits){\n unsigned sign=(bits&128u)<<8,mag=bits&127u;\n if(mag>=8)return sign|((mag<<4)+0x3c00u);\n if(mag==0)return sign;\n __nv_bfloat16 value=__float2bfloat16_rn(fp8_value(bits));\n return *reinterpret_cast<uint16_t*>(&value);\n}\n__device__ __forceinline__ uint32_t dequant_pair(unsigned char bits,uint16_t sf,uint32_t lut){\n uint32_t lo=__shfl_sync(0xffffffff,lut,bits&15),hi=__shfl_sync(0xffffffff,lut,bits>>4);\n uint32_t packed=lo|(hi<<16),scales=(uint32_t)sf|((uint32_t)sf<<16);\n __nv_bfloat162 values=*reinterpret_cast<__nv_bfloat162*>(&packed);\n __nv_bfloat162 result=__hmul2(values,*reinterpret_cast<__nv_bfloat162*>(&scales));\n return *reinterpret_cast<uint32_t*>(&result);\n}\n__device__ __forceinline__ void mma(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1,float& c0,float& c1,float& c2,float& c3){\n asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\n\n__device__ __forceinline__ void copy16(void* dst,const void* src){\n unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));\n asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(src));\n}\ntemplate<bool GATED>\n__device__ __forceinline__ void prefetch_tile(unsigned char* dst,const unsigned char* wt,const unsigned char* st,int k16,int N){\n int t=threadIdx.x;\n if(t<32)copy16(dst+t*16,wt+k16*512+t*16);\n else if(t<36)copy16(dst+512+(t-32)*16,st+k16*64+(t-32)*16);\n if constexpr(GATED){\n  if(t>=36&&t<68)copy16(dst+576+(t-36)*16,wt+(int64_t)(N/64)*256*512+k16*512+(t-36)*16);\n  else if(t>=68&&t<72)copy16(dst+1088+(t-68)*16,st+(int64_t)(N/64)*256*64+k16*64+(t-68)*16);\n }\n asm volatile("cp.async.commit_group;");\n}\n\ntemplate<bool GATED,int SPLITS>\n__global__ void warp_mma(const __nv_bfloat16* __restrict__ X,const int* __restrict__ SORT,const int* __restrict__ EXP,const int* __restrict__ PAD,\n const unsigned char* __restrict__ W,const unsigned char* __restrict__ SF,const float* __restrict__ GS,float* __restrict__ PART,__nv_bfloat16* __restrict__ Y,int N,int slots,int topk){\n constexpr int K=4096;int block=blockIdx.y,split=blockIdx.z;\n int expert=EXP[block];if(expert<0)return;\n int rows=GATED?2*N:N,feature=blockIdx.x*64,lane=threadIdx.x%32;\n int m0=(threadIdx.x/32)*16+lane/4,kk=(lane%4)*2;\n int xslot=lane/4==0?block:slots;\n const unsigned char* wt=W+((int64_t)expert*(rows/64)+blockIdx.x)*(K/16)*512;\n const unsigned char* st=SF+((int64_t)expert*(rows/64)+blockIdx.x)*(K/16)*64;\n float c0=0,c1=0,c2=0,c3=0,u0=0,u1=0,u2=0,u3=0;\n uint32_t lut=__float_as_uint(fp4_value(lane&15))>>16;\n constexpr int PIPE=8,STEPS=256/SPLITS;\n __shared__ __align__(16) unsigned char ring[PIPE][GATED?1152:576];\n #pragma unroll\n for(int j=0;j<PIPE&&j<STEPS;j++)prefetch_tile<GATED>(ring[j],wt,st,split*STEPS+j,N);\n for(int step=0;step<STEPS;step++){\n  if(step<=STEPS-PIPE){\n   if constexpr(PIPE==2)asm volatile("cp.async.wait_group 1;");\n   else if constexpr(PIPE==4)asm volatile("cp.async.wait_group 3;");\n   else asm volatile("cp.async.wait_group 7;");\n  }else asm volatile("cp.async.wait_group 0;");\n  __syncthreads();\n  unsigned char* tile=ring[step%PIPE];\n  uint32_t packed0=*reinterpret_cast<const uint32_t*>(tile+threadIdx.x*4),packed1=0;\n  uint16_t scales0=*reinterpret_cast<const uint16_t*>(tile+512+(threadIdx.x/32)*16+(lane/4)*2),scales1=0;\n  if constexpr(GATED){packed1=*reinterpret_cast<const uint32_t*>(tile+576+threadIdx.x*4);scales1=*reinterpret_cast<const uint16_t*>(tile+1088+(threadIdx.x/32)*16+(lane/4)*2);}\n  uint32_t b0=0,b1=0;\n  if(xslot<slots){const __nv_bfloat16* x=X+(xslot/topk)*K+(split*STEPS+step)*16+kk;b0=*reinterpret_cast<const uint32_t*>(x);b1=*reinterpret_cast<const uint32_t*>(x+8);}\n  __syncthreads();\n  if(step+PIPE<STEPS)prefetch_tile<GATED>(ring[step%PIPE],wt,st,split*STEPS+step+PIPE,N);\n  uint16_t sf0=fp8_to_bf16_bits(scales0&255),sf1=fp8_to_bf16_bits(scales0>>8);\n  uint32_t a0=dequant_pair(packed0&255,sf0,lut),a1=dequant_pair((packed0>>8)&255,sf1,lut);\n  uint32_t a2=dequant_pair((packed0>>16)&255,sf0,lut),a3=dequant_pair(packed0>>24,sf1,lut);\n  mma(a0,a1,a2,a3,b0,b1,c0,c1,c2,c3);\n  if constexpr(GATED){\n   sf0=fp8_to_bf16_bits(scales1&255);sf1=fp8_to_bf16_bits(scales1>>8);\n   a0=dequant_pair(packed1&255,sf0,lut);a1=dequant_pair((packed1>>8)&255,sf1,lut);\n   a2=dequant_pair((packed1>>16)&255,sf0,lut);a3=dequant_pair(packed1>>24,sf1,lut);\n   mma(a0,a1,a2,a3,b0,b1,u0,u1,u2,u3);\n  }\n }\n float cv[4]={c0,c1,c2,c3},uv[4]={u0,u1,u2,u3};\n #pragma unroll\n for(int j=0;j<4;j++){\n  int slot=((lane%4)*2+j%2)==0?block:slots,n=feature+m0+(j/2)*8;\n  if(slot<slots){\n   if constexpr(SPLITS==1){\n    float value=cv[j]*GS[expert];\n    if constexpr(GATED){float g=__bfloat162float(__float2bfloat16_rn(value)),u=__bfloat162float(__float2bfloat16_rn(uv[j]*GS[expert]));value=g/(1.f+expf(-g))*u;}\n    Y[slot*N+n]=__float2bfloat16_rn(value);\n   }else{\n    PART[((int64_t)split*slots+slot)*rows+n]=cv[j];\n    if constexpr(GATED)PART[((int64_t)split*slots+slot)*rows+N+n]=uv[j];\n   }\n  }\n }\n}\ntemplate<bool GATED,int SPLITS>\n__global__ void finish_mma(const float* __restrict__ P,const int* __restrict__ IDS,const float* __restrict__ GS,__nv_bfloat16* __restrict__ Y,int N,int slots){\n int slot=blockIdx.y,n=blockIdx.x*256+threadIdx.x,expert=IDS[slot];if(expert<0||n>=N)return;\n constexpr int factor=GATED?2:1;float g=0,u=0;\n #pragma unroll\n for(int s=0;s<SPLITS;s++){g+=P[((int64_t)s*slots+slot)*(N*factor)+n];if constexpr(GATED)u+=P[((int64_t)s*slots+slot)*(N*factor)+N+n];}\n float value=g*GS[expert];\n if constexpr(GATED){g=__bfloat162float(__float2bfloat16_rn(value));u=__bfloat162float(__float2bfloat16_rn(u*GS[expert]));value=g/(1.f+expf(-g))*u;}\n Y[slot*N+n]=__float2bfloat16_rn(value);\n}\nvoid warp_mma_run(TensorView x,TensorView sorted,TensorView experts,TensorView padded,TensorView ids,TensorView w,TensorView sf,TensorView gs,TensorView partial,TensorView y,int64_t topk,bool gated,int64_t splits){\n ffi::CUDADeviceGuard guard(x.device().device_id);TVM_FFI_ICHECK(x.size(1)==4096&&y.size(1)%64==0);\n auto X=static_cast<const __nv_bfloat16*>(x.data_ptr());auto SORT=static_cast<const int*>(sorted.data_ptr());auto EXP=static_cast<const int*>(experts.data_ptr());auto PAD=static_cast<const int*>(padded.data_ptr());auto IDS=static_cast<const int*>(ids.data_ptr());\n auto W=static_cast<const unsigned char*>(w.data_ptr());auto SF=static_cast<const unsigned char*>(sf.data_ptr());auto GS=static_cast<const float*>(gs.data_ptr());auto PART=static_cast<float*>(partial.data_ptr());auto Y=static_cast<__nv_bfloat16*>(y.data_ptr());\n int N=y.size(1),slots=y.size(0);auto stream=get_stream(x.device());\n #define LAUNCH(G,S) {warp_mma<G,S><<<dim3(N/64,slots,S),128,0,stream>>>(X,SORT,EXP,PAD,W,SF,GS,PART,Y,N,slots,topk);if constexpr(S>1)finish_mma<G,S><<<dim3((N+255)/256,slots),256,0,stream>>>(PART,IDS,GS,Y,N,slots);}\n if(gated){if(splits==8){LAUNCH(true,8);}else if(splits==16){LAUNCH(true,16);}else if(splits==32){LAUNCH(true,32);}else{LAUNCH(true,64);}}\n else{if(splits==8){LAUNCH(false,8);}else if(splits==16){LAUNCH(false,16);}else if(splits==32){LAUNCH(false,32);}else{LAUNCH(false,64);}}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(warp_mma_run,warp_mma_run);\n'

@cache
def _warp_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_moe_warp_mma_async_p8_v1',cuda_sources=_WARP_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])


def prepare(weights):
 state=_base_prepare(weights)
 state['wg']={}
 for (start,count) in state['banks']:
  bank={}
  for name in ('w13','w2'):
   w=weights[name][start:start+count];n=w.shape[1];k=w.shape[2]*2
   blocked=w.reshape(count,n//64,64,k//16,8).permute(0,1,3,2,4).contiguous()
   bank[name]=blocked.reshape(count,n//64,k//16,4,2,8,2,4).permute(0,1,2,3,5,7,6,4).contiguous()
   scales=weights[name+'_sf'][start:start+count].view(torch.uint8).reshape(count,n//64,64,k//16).permute(0,1,3,2).contiguous()
   bank[name+'_sf']=scales.reshape(count,n//64,k//16,4,2,8).permute(0,1,2,3,5,4).contiguous()
   bank[name+'_gs']=weights[name+'_gs'][start:start+count].contiguous()
  state['wg'][(start,count)]=bank
 state['warp_module']=_warp_module()
 return state

def kernel_fn(data,state):
 cfg=data['config'];m=cfg['bs'];h=cfg['d_hidden'];i=cfg['d_expert'];topk=cfg['top_k']
 if m!=1:return _best_moe(data,state)
 bank=state['banks'][(cfg['local_start'],cfg['local_count'])]
 wg=state['wg'][(cfg['local_start'],cfg['local_count'])]
 local=bank['expert_map'][data['topk_ids']].contiguous()
 sorted_ids=block_experts=n_padded=local.view(-1)
 x=data['hidden_states'];activated=torch.empty((m*topk,i),device=x.device,dtype=torch.bfloat16)
 y=torch.empty((m*topk,h),device=x.device,dtype=torch.bfloat16)
 mod=state['warp_module']
 partial13=torch.empty((8,m*topk,2*i) if 8>1 else (0,),device=x.device,dtype=torch.float32)
 partial2=torch.empty((8,m*topk,h) if 8>1 else (0,),device=x.device,dtype=torch.float32)
 common=dict(moe_block_size=8,mul_topk_weights=False,is_ep=True,b_q_type=scalar_types.float4_e2m1f,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
 if True:
  mod.warp_mma_run(x,sorted_ids,block_experts,n_padded,local,wg['w13'],wg['w13_sf'],wg['w13_gs'],partial13,activated,topk,True,8)
 else:
  cache1=torch.empty((m*topk,2*i),device=x.device,dtype=torch.bfloat16)
  cache1=moe_wna16_marlin_gemm(x,cache1,bank['w13'],None,bank['w13_s'],bank['w13_gs'],None,None,None,state['workspace'],sorted_ids,block_experts,n_padded,data['topk_weights'],top_k=topk,size_m=m,size_n=2*i,size_k=h,**common)
  silu_and_mul(cache1,activated,expert_ids=local.view(-1),expert_step=1)
 if True:
  mod.warp_mma_run(activated,sorted_ids,block_experts,n_padded,local,wg['w2'],wg['w2_sf'],wg['w2_gs'],partial2,y,1,False,8)
 else:
  y=moe_wna16_marlin_gemm(activated,y,bank['w2'],None,bank['w2_s'],bank['w2_gs'],None,None,None,state['workspace'],sorted_ids,block_experts,n_padded,data['topk_weights'],top_k=1,size_m=m*topk,size_n=h,size_k=i,**common)
 return fused_post_expert_norm_weighted_sum(y.view(m,topk,h),data['topk_weights'],norm_weight=state['norm_w'],eps=POST_NORM_EPS,weight_first=True,slot_expert_ids=local)
