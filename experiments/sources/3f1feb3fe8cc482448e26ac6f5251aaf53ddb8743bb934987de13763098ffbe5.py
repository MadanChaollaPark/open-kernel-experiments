# okc: problem=muse_bf16_proj
import torch
import triton as tr
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

@tr.jit
def _compressed_mm(X,W,LOW,HIGH,BASE,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,S:tl.constexpr,ROUTER:tl.constexpr,TRANS:tl.constexpr,NFULL:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.program_id(2)*BK+tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for _ in range(tl.cdiv(K,BK*S)):
        a=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        off=k[:,None]*NFULL+n[None,:]
        mask=(n[None,:]<N)&(k[:,None]<K)
        base=tl.load(BASE+k[:,None]*(NFULL//32)+n[None,:]//32,mask,0).to(tl.int32)
        low=tl.load(LOW+off,mask,0).to(tl.int32)
        high=tl.load(HIGH+k[:,None]*(NFULL//2)+n[None,:]//2,mask,0).to(tl.int32)
        nibble=(high>>((n[None,:]%2)*4))&15
        bits=((nibble&8)<<12)|(((base&127)-(nibble&7))<<8)|low
        decoded=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
        original=tl.load(W+off,mask&((base&128)!=0),0)
        w=tl.where((base&128)!=0,original,decoded)
        acc=tl.dot(a,w,acc)
        k+=BK*S
    tl.store(Y+tl.program_id(2)*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))

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

def _original_prepare(weights):
    state=dict(weights)
    state['qkv_t']=weights['qkv_gate'].t().contiguous()
    state['pooler_t']=weights['pooler'].t().contiguous()
    hi=weights['router'].to(torch.bfloat16)
    lo=(weights['router']-hi.float()).to(torch.bfloat16)
    state['hi']=hi;state['lo']=lo
    state['hi_t']=hi.t().contiguous();state['lo_t']=lo.t().contiguous()
    return state

def _original_kernel_fn(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    router=layer=='router'
    if router:
        bm=16 if m<=16 else (32 if m<=32 else 64)
        bn=32 if m<=128 else 64
        bk=128;split=8;trans=32<m<=128
        w=state['hi_t'] if trans else state['hi']
        lo=state['lo_t'] if trans else state['lo']
    elif layer=='qkv_gate' and m<=16:
        bm=16;bn=32;bk=128;split=4;trans=True
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


def prepare(weights):
    state=_original_prepare(weights)
    for name,w in weights.items():
        if w.dtype!=torch.bfloat16: continue
        w=w.t().contiguous()
        state[name+'_packed']=w
        n,k=w.shape
        # A 32-value block retains all BF16 bits. Rare wide-exponent blocks
        # use their original BF16 storage, selected by the header flag.
        bits=w.view(torch.int16).to(torch.int32)&65535
        exphi=(bits>>8)&127
        groups=exphi.view(n,k//32,32)
        mx=groups.amax(-1);mn=groups.amin(-1)
        header=(mx|((mn<mx-7).to(torch.int32)<<7)).to(torch.uint8)
        delta=(mx[:,:,None]-groups).reshape(n,k)
        nibble=(((bits>>12)&8)|(delta&7)).to(torch.uint8)
        packed=nibble[:,0::2]|(nibble[:,1::2]<<4)
        state[name+'_compressed']=((bits&255).to(torch.uint8).contiguous(),packed.contiguous(),header.contiguous())
    return state

def kernel_fn(data,state):
    x=data['x'];m,k=x.shape;layer=data['config']['layer'];n=data['config']['n']
    if layer=='router':return _original_kernel_fn(data,state)
    bm=16 if m<=16 else (32 if m<=32 else 64)
    bn=32;bk=128;split=4
    low,high,header=state[layer+'_compressed']
    y=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    p=torch.empty((split,m,n),device=x.device,dtype=torch.float32)
    _compressed_mm[(tr.cdiv(m,bm),tr.cdiv(n,bn),split)](x,state[layer+'_packed'],low,high,header,p,m,n,k,bm,bn,bk,split,False,True,state[layer].shape[0],num_warps=4,num_stages=2)
    _reduce[(tr.cdiv(m*n,512),)](p,y,m*n,split,512)
    return y
