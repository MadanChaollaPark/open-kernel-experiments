# okc: problem=muse_attention_decode
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

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

def kernel_fn(data):
    cfg=data['config']; b=cfg['batch']; length=cfg['q_len']
    hq=cfg['num_q_heads']; hk=cfg['num_kv_heads']; tokens=b*length
    qkv=data['qkv']; q=torch.empty((tokens,hq,64),device=qkv.device,dtype=torch.float8_e4m3fn)
    if tokens>=16:
        _prepare_heads[(tokens,tr.cdiv(hq+hk,4))](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],4,num_warps=4)
    else:
        _prepare[(tokens,hq+hk)](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],num_warps=4)
    descale={}
    if cfg['kv_scale']!=1.0:
        descale=dict(k_descale=data['k_scale'].reshape(1,1).expand(b,hk),v_descale=data['v_scale'].reshape(1,1).expand(b,hk))
    window=cfg['window']
    out=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],
        page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],
        max_seqlen_q=length,softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'],causal=True,
        window_size=(window-1,0) if window else (-1,-1),scheduler_metadata=None,num_splits=min(32,max(1,tr.cdiv(window or data['page_table'].shape[1]*cfg['page_size'],256))),pack_gqa=True,**descale)
    if isinstance(out,tuple):out=out[0]
    gate=qkv.view(tokens,-1)[:,(hq+2*hk)*64:]
    return rms_norm_gate(out.reshape(tokens,hq*64).to(torch.bfloat16),gate,None,head_dim=64,eps=1e-5,has_norm=True).view(b,length,hq*64)

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

