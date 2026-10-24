# okc: problem=muse_bf16_proj
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.full((BN,BK),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        kk=start*BK+k
        x=tl.load(X+kk,kk<K,0).to(tl.float32)
        w=tl.load(W+n[:,None]*K+kk[None,:],(n[:,None]<N)&(kk[None,:]<K),0).to(tl.float32)
        acc+=w*x[None,:]
    y=tl.sum(acc,1)
    tl.store(Y+n,y,n<N)

@tr.jit
def _mm(X,W,L,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,ROUTER:tl.constexpr,NFULL:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.program_id(2)*BK+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for _ in range(tl.cdiv(K,BK*S)):
        a=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        b=tl.load(W+k[:,None]*NFULL+n[None,:],(n[None,:]<N)&(k[:,None]<K),0)
        acc=tl.dot(a,b,acc)
        if ROUTER:
            lo=tl.load(L+k[:,None]*NFULL+n[None,:],(n[None,:]<N)&(k[:,None]<K),0)
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
    state['hi']=weights['router'].to(torch.bfloat16)
    state['lo']=(weights['router']-state['hi'].float()).to(torch.bfloat16)
    return {name:w.t().contiguous() for name,w in state.items()}

def kernel_fn(data,state):
    x=data['x']; m,k=x.shape; n=data['config']['n']; layer=data['config']['layer']
    router=layer=='router'
    y=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if False:
        _gemv[(tr.cdiv(n,4),)](x,state[layer],y,n,k,4,1024,num_warps=4)
        return y
    bm=16 if m<=16 else (32 if m<=32 else 64)
    bn=32; bk=128; split=8
    p=torch.empty((split,m,n),device=x.device,dtype=torch.float32) if split>1 else y
    _mm[(tr.cdiv(m,bm),tr.cdiv(n,bn),split)](x,state['hi'] if router else state[layer],state['lo'],p,m,n,k,bm,bn,bk,split,router,state[layer].shape[1],num_warps=4,num_stages=3)
    if split>1: _reduce[(tr.cdiv(m*n,512),)](p,y,m*n,split,512)
    return y
