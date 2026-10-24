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

def _hybrid_prepare(weights):
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

from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

def prepare(weights):
    state=_hybrid_prepare(weights)
    state['runner']=get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
    state['cublas_workspace']=torch.empty(32*1024*1024,device=weights['router'].device,dtype=torch.uint8)
    return state

def kernel_fn(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    if layer=='router':return _hybrid(data,state)
    out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    args=[x,state[layer][:n].t(),None,False,out,state['cublas_workspace']]
    # This is an algorithm-metadata query; it never reads activation values.
    count=state['runner']._get_algos(args)[1]
    if 67>=count:return _hybrid(data,state)
    return state['runner'].forward(args,tactic=67)
