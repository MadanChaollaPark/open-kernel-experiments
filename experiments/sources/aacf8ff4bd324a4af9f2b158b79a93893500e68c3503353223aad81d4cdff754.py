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
_WARP_SRC='#include <cuda_bf16.h>\n#include <flashinfer/utils.cuh>\n#include "tvm_ffi_utils.h"\nusing namespace flashinfer;\ntemplate<int WARPS,int WPR,bool ROUTER>\n__global__ void dense_warp_mv(const __nv_bfloat16* __restrict__ X,const void* __restrict__ W,void* __restrict__ Y,int N,int K){\n constexpr int ROWS=WARPS/WPR;\n int warp=threadIdx.x/32,lane=threadIdx.x%32,row=warp/WPR,part=warp%WPR,n=blockIdx.x*ROWS+row;\n float sums[4]={0.f,0.f,0.f,0.f};\n if(n<N){\n  #pragma unroll 2\n  for(int k=part*256+lane*8;k<K;k+=WPR*256){\n   uint4 xv=*reinterpret_cast<const uint4*>(X+blockIdx.y*K+k);\n   const __nv_bfloat16* xf=reinterpret_cast<const __nv_bfloat16*>(&xv);\n   if constexpr(ROUTER){\n    auto wr=static_cast<const float*>(W)+(int64_t)n*K+k;\n    float4 w0=*reinterpret_cast<const float4*>(wr),w1=*reinterpret_cast<const float4*>(wr+4);\n    const float* wf0=reinterpret_cast<const float*>(&w0);const float* wf1=reinterpret_cast<const float*>(&w1);\n    #pragma unroll\n    for(int j=0;j<4;j++){sums[j]=fmaf(__bfloat162float(xf[j]),wf0[j],sums[j]);sums[j]=fmaf(__bfloat162float(xf[j+4]),wf1[j],sums[j]);}\n   }else{\n    uint4 wv=*reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(W)+(int64_t)n*K+k);\n    const __nv_bfloat16* wf=reinterpret_cast<const __nv_bfloat16*>(&wv);\n    #pragma unroll\n    for(int j=0;j<8;j++)sums[j%4]=fmaf(__bfloat162float(xf[j]),__bfloat162float(wf[j]),sums[j%4]);\n   }\n  }\n }\n float sum=(sums[0]+sums[1])+(sums[2]+sums[3]);\n #pragma unroll\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);\n if constexpr(WPR>1){\n  __shared__ float partial[WARPS];\n  if(lane==0)partial[warp]=sum;\n  __syncthreads();\n  if(part==0 && lane==0){sum=0;\n   #pragma unroll\n   for(int p=0;p<WPR;p++)sum+=partial[row*WPR+p];\n  }\n }\n if(part==0 && lane==0 && n<N){\n  if constexpr(ROUTER)static_cast<float*>(Y)[blockIdx.y*N+n]=sum;\n  else static_cast<__nv_bfloat16*>(Y)[blockIdx.y*N+n]=__float2bfloat16_rn(sum);\n }\n}\nvoid dense_warp_run(TensorView x,TensorView w,TensorView y,int64_t warps,int64_t wpr,bool router){\n ffi::CUDADeviceGuard guard(x.device().device_id);\n int M=x.size(0),K=x.size(1),N=y.size(1);\n TVM_FFI_ICHECK(K%256==0);\n auto X=static_cast<const __nv_bfloat16*>(x.data_ptr());auto W=w.data_ptr();auto Y=y.data_ptr();auto stream=get_stream(x.device());\n #define LAUNCH(B,P,R) dense_warp_mv<B,P,R><<<dim3((N+B/P-1)/(B/P),M),B*32,0,stream>>>(X,W,Y,N,K)\n #define CHOOSE(R) if(wpr==16){LAUNCH(16,16,R);}else if(wpr==8 && warps==16){LAUNCH(16,8,R);}else if(wpr==8){LAUNCH(8,8,R);}else if(warps==4){LAUNCH(4,1,R);}else if(warps==16){LAUNCH(16,4,R);}else if(wpr==1){LAUNCH(8,1,R);}else if(wpr==2){LAUNCH(8,2,R);}else{LAUNCH(8,4,R);}\n if(router){CHOOSE(true);}else{CHOOSE(false);}\n TVM_FFI_ICHECK(cudaGetLastError()==cudaSuccess);\n}\nTVM_FFI_DLL_EXPORT_TYPED_FUNC(dense_warp_run,dense_warp_run);\n'

@cache
def _warp_module():
 import pathlib,importlib.util
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 return load_inline('okc_dense_warp_wide_v1',cuda_sources=_WARP_SRC,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include')])

def kernel_fn(data,state):
 x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
 if m!=1 or k%256:return _best_dense(data,state)
 y=torch.empty((m,n),device=x.device,dtype=torch.float32 if layer=='router' else torch.bfloat16)
 _warp_module().dense_warp_run(x,state[layer],y,16,16,layer=='router')
 return y
