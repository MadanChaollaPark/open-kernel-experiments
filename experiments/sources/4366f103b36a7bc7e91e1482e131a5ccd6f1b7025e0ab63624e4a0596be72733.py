# okc: problem=muse_bf16_proj
import torch
import triton as tr
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

@tr.jit
def _mm(X,W,L,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,ROUTER:tl.constexpr,TRANS:tl.constexpr,NFULL:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.program_id(2)*BK+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for _ in range(tl.cdiv(K,BK*S)):
        a=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        off=k[:,None]*NFULL+n[None,:] if TRANS else n[None,:]*K+k[:,None]
        b=tl.load(W+off,(n[None,:]<N)&(k[:,None]<K),0)
        acc=tl.dot(a,b,acc)
        if ROUTER:
            lo=tl.load(L+off,(n[None,:]<N)&(k[:,None]<K),0)
            acc=tl.dot(a,lo,acc)
        k+=BK*S
    tl.store(Y+tl.program_id(2)*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

@tr.jit
def _reduce(P,Y,T:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    acc=tl.full((B,),0,tl.float32)
    for s in range(S): acc+=tl.load(P+s*T+i,i<T,0)
    tl.store(Y+i,acc,i<T)

def prepare(weights):
    state=dict(weights)
    state['qkv_t']=weights['qkv_gate'].t().contiguous()
    state['pooler_t']=weights['pooler'].t().contiguous()
    hi=weights['router'].to(torch.bfloat16)
    lo=(weights['router']-hi.float()).to(torch.bfloat16)
    state['hi']=hi;state['lo']=lo
    state['hi_t']=hi.t().contiguous();state['lo_t']=lo.t().contiguous()
    return state

def _hybrid(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    router=layer=='router'
    if router:
        bm=16 if m<=16 else (32 if m<=32 else 64)
        bn=32 if m<=128 else 64
        bk=128;split=8;trans=32<m<=128
        w=state['hi_t'] if trans else state['hi']
        lo=state['lo_t'] if trans else state['lo']
    elif layer=='qkv_gate' and m<=16:
        bm=16;bn=64;bk=128;split=8;trans=True
        w=state['qkv_t'];lo=state['lo']
    elif layer=='pooler' and 8<m<=16:
        bm=16;bn=64;bk=128;split=4;trans=False
        w=state[layer];lo=state['lo']
    elif layer in ('unpooler','pooler') and m<=8:
        from flashinfer.gemm import mm_bf16
        return mm_bf16(x,state[layer][:n].t(),backend='tinygemm')
    elif layer=='pooler':
        return torch.mm(x,state['pooler_t'])
    else:
        return _bf16_gemm_dispatch_impl(x,state[layer][:n],None)
    y=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    p=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
    full=w.shape[1] if trans else n
    _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn),split)](x,w,lo,p,m,n,k,bm,bn,bk,split,router,trans,full,num_warps=4,num_stages=3)
    _reduce[(tr.cdiv(m*n,512),)](p,y,m*n,split,512)
    return y

@tr.jit
def _fullvec(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    m=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.program_id(2)*BK+tl.arange(0,BK)
    x=tl.load(X+m*K+k,k<K,0).to(tl.float32)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0).to(tl.float32)
    out=tl.sum(w*x[None,:],1)
    tl.store(Y+tl.program_id(2)*M*N+m*N+n,out,n<N)

def _best_dense(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    if m!=1 or layer=='o':return _hybrid(data,state)
    bn=2 if layer=='qkv_gate' else 1
    warps=8 if layer in ('qkv_gate','unpooler') else 4
    y=torch.empty((m,n),device=x.device,dtype=torch.float32 if layer=='router' else torch.bfloat16)
    _fullvec[(m,tr.cdiv(n,bn),1)](x,state[layer],y,m,n,k,bn,tr.next_power_of_2(k),num_warps=warps)
    return y

from functools import cache
_WMMA_SRC='#include <cuda_bf16.h>\n#include <mma.h>\n#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\nnamespace wmma=nvcuda::wmma;\n__device__ __forceinline__ void copy8(__nv_bfloat16* dst,const __nv_bfloat16* src){\n unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));\n asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(address),"l"(src));\n}\ntemplate<int WARPS>\n__global__ void okc_wmma(const __nv_bfloat16* __restrict__ X,const __nv_bfloat16* __restrict__ W,__nv_bfloat16* __restrict__ Y,int M,int N,int K){\n int warp=threadIdx.x/32;\n int mr=blockIdx.y*16,nr=blockIdx.x*16;\n wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;\n wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;\n wmma::fragment<wmma::accumulator,16,16,16,float> c;\n wmma::fill_fragment(c,0.0f);\n constexpr int BK=256;\n __shared__ __align__(32) __nv_bfloat16 tiles[2][2][16*BK];\n for(int j=threadIdx.x*8;j<16*BK;j+=WARPS*32*8){\n  copy8(&tiles[0][0][j],X+(mr+j/BK)*K+j%BK);\n  copy8(&tiles[0][1][j],W+(nr+j/BK)*K+j%BK);\n }\n asm volatile("cp.async.commit_group;");\n asm volatile("cp.async.wait_group 0;");\n __syncthreads();\n for(int start=0;start<K;start+=BK){\n  int buf=(start/BK)&1;\n  if(start+BK<K){\n   for(int j=threadIdx.x*8;j<16*BK;j+=WARPS*32*8){\n    copy8(&tiles[buf^1][0][j],X+(mr+j/BK)*K+start+BK+j%BK);\n    copy8(&tiles[buf^1][1][j],W+(nr+j/BK)*K+start+BK+j%BK);\n   }\n   asm volatile("cp.async.commit_group;");\n  }\n  #pragma unroll\n  for(int k=warp*16;k<BK;k+=WARPS*16){\n   wmma::load_matrix_sync(a,&tiles[buf][0][k],BK);\n   wmma::load_matrix_sync(b,&tiles[buf][1][k],BK);\n   wmma::mma_sync(c,a,b,c);\n  }\n  __syncthreads();\n  asm volatile("cp.async.wait_group 0;");\n  __syncthreads();\n }\n __shared__ __align__(32) float partial[WARPS][256];\n wmma::store_matrix_sync(partial[warp],c,16,wmma::mem_row_major);\n __syncthreads();\n for(int i=threadIdx.x;i<256;i+=WARPS*32){\n  float sum=0;\n  #pragma unroll\n  for(int w=0;w<WARPS;w++)sum+=partial[w][i];\n  Y[(mr+i/16)*N+nr+i%16]=__float2bfloat16_rn(sum);\n }\n}\nvoid okc_wmma_run(TensorView x,TensorView w,TensorView y,int64_t warps){\n ffi::CUDADeviceGuard guard(x.device().device_id);\n int M=x.size(0),K=x.size(1),N=y.size(1);\n TVM_FFI_ICHECK(M%16==0 && N%16==0 && K%256==0);\n auto X=static_cast<const __nv_bfloat16*>(x.data_ptr());auto W=static_cast<const __nv_bfloat16*>(w.data_ptr());auto Y=static_cast<__nv_bfloat16*>(y.data_ptr());\n dim3 grid(N/16,M/16);auto stream=get_stream(x.device());\n if(warps==4)okc_wmma<4><<<grid,128,0,stream>>>(X,W,Y,M,N,K);\n else if(warps==8)okc_wmma<8><<<grid,256,0,stream>>>(X,W,Y,M,N,K);\n else okc_wmma<16><<<grid,512,0,stream>>>(X,W,Y,M,N,K);\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(okc_wmma_run,okc_wmma_run);\n'

@cache
def _wmma_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_dense_wmma_shared_v1',cuda_sources=_WMMA_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])

def kernel_fn(data,state):
 x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
 if layer=='router' or m>32 or m%16 or n%16 or k%256:return _best_dense(data,state)
 y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 _wmma_module().okc_wmma_run(x,state[layer],y,16)
 return y
