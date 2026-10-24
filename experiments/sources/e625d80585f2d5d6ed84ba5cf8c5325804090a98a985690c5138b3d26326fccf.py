# okc: problem=muse_attention_decode
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

@tr.jit
def _prepare(QKV,POS,LOC,KC,VC,KS,VS,Q,HQ:tl.constexpr,HK:tl.constexpr,THETA:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)
    d=tl.arange(0,64)
    stride=(2*HQ+2*HK)*64
    v=tl.load(QKV+t*stride+h*64+d).to(tl.float32)
    r=tl.rsqrt(tl.sum(v*v,0)/64+1e-5)
    v=v*r
    if THETA != 0.0:
        partner=tl.load(QKV+t*stride+h*64+(d^1)).to(tl.float32)*r
        pos=tl.load(POS+t).to(tl.float32)
        inv_freq=tl.div_rn(1.0,libdevice.pow(THETA,(2*(d//2)).to(tl.float32)/64))
        angle=pos*inv_freq
        v=v*libdevice.cos(angle)+tl.where(d%2==0,-partner,partner)*libdevice.sin(angle)
    v=v.to(tl.bfloat16).to(tl.float32)
    if h<HQ:
        tl.store(Q+(t*HQ+h)*64+d,v)
    else:
        slot=tl.load(LOC+t)
        ks=tl.load(KS);vs=tl.load(VS)
        val=tl.load(QKV+t*stride+(HQ+HK)*64+(h-HQ)*64+d).to(tl.float32)
        off=(slot*HK+h-HQ)*64+d
        tl.store(KC+off,v/ks)
        tl.store(VC+off,val/vs)


@tr.jit
def _prepare_heads(QKV,POS,LOC,KC,VC,KS,VS,Q,HQ:tl.constexpr,HK:tl.constexpr,THETA:tl.constexpr,BH:tl.constexpr):
    t=tl.program_id(0);h=tl.program_id(1)*BH+tl.arange(0,BH)
    d=tl.arange(0,64)
    stride=(2*HQ+2*HK)*64
    mask=h[:,None]<HQ+HK
    v=tl.load(QKV+t*stride+h[:,None]*64+d[None,:],mask,0.0).to(tl.float32)
    r=tl.rsqrt(tl.sum(v*v,1)/64+1e-5)
    v=v*r[:,None]
    if THETA != 0.0:
        partner=tl.load(QKV+t*stride+h[:,None]*64+(d[None,:]^1),mask,0.0).to(tl.float32)*r[:,None]
        pos=tl.load(POS+t).to(tl.float32)
        inv_freq=tl.div_rn(1.0,libdevice.pow(THETA,(2*(d//2)).to(tl.float32)/64))
        angle=pos*inv_freq
        v=v*libdevice.cos(angle)[None,:]+tl.where(d[None,:]%2==0,-partner,partner)*libdevice.sin(angle)[None,:]
    v=v.to(tl.bfloat16).to(tl.float32)
    tl.store(Q+(t*HQ+h[:,None])*64+d[None,:],v,h[:,None]<HQ)
    slot=tl.load(LOC+t);ks=tl.load(KS);vs=tl.load(VS)
    kv_mask=(h[:,None]>=HQ)&(h[:,None]<HQ+HK)
    val=tl.load(QKV+t*stride+(HQ+HK)*64+(h[:,None]-HQ)*64+d[None,:],kv_mask,0.0).to(tl.float32)
    off=(slot*HK+h[:,None]-HQ)*64+d[None,:]
    tl.store(KC+off,v/ks,kv_mask)
    tl.store(VC+off,val/vs,kv_mask)


from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
_PLANS={}

@tr.jit
def _metadata(PT,SEQ,IND,IDX,LAST,B:tl.constexpr,QL:tl.constexpr,PAGE:tl.constexpr,MAXP:tl.constexpr,BB:tl.constexpr,TILE:tl.constexpr):
    b=tl.program_id(0);block=tl.program_id(1)
    rows=tl.arange(0,BB)
    lengths=tl.load(SEQ+rows,rows<B,0)+QL
    counts=tl.where(rows<B,tl.cdiv(lengths,PAGE),0)
    start=tl.sum(tl.where(rows<b,counts,0),0)
    length=tl.load(SEQ+b)+QL
    count=tl.cdiv(length,PAGE)
    i=block*TILE+tl.arange(0,TILE)
    values=tl.load(PT+b*MAXP+i,i<count,0)
    tl.store(IDX+start+i,values,i<count)
    if block==0:
        tl.store(IND+b,start)
        if b==B-1:tl.store(IND+B,start+count)
        tl.store(LAST+b,(length-1)%PAGE+1)

def _plan(data,tc):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len'];hq=cfg['num_q_heads'];hk=cfg['num_kv_heads'];page=cfg['page_size'];maxp=data['page_table'].shape[1]
    key=(b,ql,hq,hk,page,maxp,cfg['window'],cfg['sm_scale'],cfg['qk_scale_factor'],tc,str(data['qkv'].device))
    if key not in _PLANS:
        device=data['qkv'].device
        workspace=torch.zeros(128*1024*1024,device=device,dtype=torch.uint8)
        ind=torch.empty((b+1,),device=device,dtype=torch.int32)
        idx=torch.empty((b*maxp,),device=device,dtype=torch.int32)
        last=torch.empty((b,),device=device,dtype=torch.int32)
        wrapper=BatchDecodeWithPagedKVCacheWrapper(workspace,kv_layout='NHD',use_cuda_graph=True,use_tensor_cores=tc,paged_kv_indptr_buffer=ind,paged_kv_indices_buffer=idx,paged_kv_last_page_len_buffer=last,backend='fa2')
        # A shape-only upper-bound schedule. Runtime CSR metadata is rebuilt
        # from this call's GPU lengths and page table before every attention.
        host_ind=torch.arange(b+1,dtype=torch.int32)*maxp
        host_idx=torch.zeros(b*maxp,dtype=torch.int32)
        host_last=torch.full((b,),page,dtype=torch.int32)
        wrapper.plan(host_ind,host_idx,host_last,hq,hk,64,page,q_data_type=torch.bfloat16,kv_data_type=torch.float8_e4m3fn,sm_scale=cfg['sm_scale']*cfg['qk_scale_factor'],window_left=cfg['window']-1 if cfg['window'] else -1,q_len_per_req=ql)
        _PLANS[key]=(wrapper,ind,idx,last)
    return _PLANS[key]

def kernel_fn(data):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len'];hq=cfg['num_q_heads'];hk=cfg['num_kv_heads']
    wrapper,ind,idx,last=_plan(data,False or ql>1)
    qkv=data['qkv'];q=torch.empty((b*ql,hq,64),device=qkv.device,dtype=torch.bfloat16)
    _prepare_heads[(b*ql,tr.cdiv(hq+hk,4))](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],4,num_warps=4)
    maxp=data['page_table'].shape[1]
    _metadata[(b,tr.cdiv(maxp,256))](data['page_table'],data['seq_lens'],ind,idx,last,b,ql,cfg['page_size'],maxp,tr.next_power_of_2(b),256)
    out=wrapper.run(q,(data['k_cache'],data['v_cache']),k_scale=cfg['kv_scale'],v_scale=cfg['kv_scale'])
    gate=qkv.reshape(b*ql,-1)[:,(hq+2*hk)*64:]
    return rms_norm_gate(out.reshape(b*ql,hq*64).to(torch.bfloat16),gate,None,head_dim=64,eps=1e-5,has_norm=True).reshape(b,ql,hq*64)
