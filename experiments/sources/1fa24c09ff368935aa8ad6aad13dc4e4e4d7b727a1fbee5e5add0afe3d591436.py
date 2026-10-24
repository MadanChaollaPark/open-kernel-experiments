# okc: problem=muse_bf16_proj
import torch
import triton as tr
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

@tr.jit
def _mv(X,W,P,N:tl.constexpr,K:tl.constexpr,FULL:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    split=tl.program_id(1)
    kk=tl.arange(0,BK)
    acc=tl.full((BK,BN),0,tl.float32)
    for it in range(tl.cdiv(K,S*BK)):
        k=split*tl.cdiv(K,S)+it*BK+kk
        mask=k<tl.minimum((split+1)*tl.cdiv(K,S),K)
        x=tl.load(X+k,mask,0).to(tl.float32)
        w=tl.load(W+k[:,None]*FULL+n[None,:],mask[:,None]&(n[None,:]<N),0).to(tl.float32)
        acc+=x[:,None]*w
    out=tl.sum(acc,0)
    tl.store(P+split*N+n,out,n<N)

@tr.jit
def _sum(P,Y,N:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    n=tl.program_id(0)*B+tl.arange(0,B)
    a=tl.full((B,),0,tl.float32)
    for s in range(S):a+=tl.load(P+s*N+n,n<N,0)
    tl.store(Y+n,a,n<N)

def prepare(weights):
    state=dict(weights)
    state.update({'t_'+name:w.t().contiguous() for name,w in weights.items()})
    w=weights['router'];hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    state['pair']=torch.cat([hi,lo],dim=0).t().contiguous()
    return state

def kernel_fn(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    if m!=1:
        if layer!='router':return _bf16_gemm_dispatch_impl(x,state[layer][:n],None)
        both=torch.mm(x,state['pair'],out_dtype=torch.float32)
        return both[:,:n]+both[:,n:]
    bn,bk,split=64,32,32
    p=torch.empty((split,n),device=x.device,dtype=torch.float32)
    y=torch.empty((1,n),device=x.device,dtype=state[layer].dtype)
    w=state['t_'+layer]
    _mv[(tr.cdiv(n,bn),split)](x,w,p,n,k,w.shape[1],bn,bk,split,num_warps=4)
    _sum[(tr.cdiv(n,256),)](p,y,n,split,256)
    return y
