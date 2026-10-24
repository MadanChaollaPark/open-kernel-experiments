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
def _attention_inline(QKV,POS,LOC,KC,VC,KS,VS,PT,SEQ,P,LSE,
               HQ:tl.constexpr,HK:tl.constexpr,QL:tl.constexpr,PAGE:tl.constexpr,MAXP:tl.constexpr,
               WINDOW:tl.constexpr,SCALE:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,THETA:tl.constexpr,BNEW:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);s=tl.program_id(2)
    group:tl.constexpr=HQ//HK
    r=tl.arange(0,BM);d=tl.arange(0,64);nn=tl.arange(0,BN)
    token=r//group;head=h*group+r%group
    seq=tl.load(SEQ+b)
    stride:tl.constexpr=(2*HQ+2*HK)*64
    tq=b*QL+token
    q=tl.load(QKV+tq[:,None]*stride+head[:,None]*64+d[None,:],r[:,None]<QL*group,0.0).to(tl.float32)
    qnorm=tl.rsqrt(tl.sum(q*q,1)/64+1e-5)
    q=q*qnorm[:,None]
    if THETA!=0.0:
        qp=tl.load(QKV+tq[:,None]*stride+head[:,None]*64+(d[None,:]^1),r[:,None]<QL*group,0.0).to(tl.float32)*qnorm[:,None]
        position=tl.load(POS+tq,r<QL*group,0).to(tl.float32)
        inv=tl.div_rn(1.0,libdevice.pow(THETA,(2*(d//2)).to(tl.float32)/64))
        angle=position[:,None]*inv[None,:]
        q=q*libdevice.cos(angle)+tl.where(d[None,:]%2==0,-qp,qp)*libdevice.sin(angle)
    q=q.to(tl.bfloat16).to(tl.float8e4nv)
    ks=tl.load(KS);vs=tl.load(VS)
    if WINDOW>0:
        first=tl.maximum(0,seq-WINDOW+1)
    else:
        first=0
    begin=first+s*CHUNK
    acc=tl.full((BM,64),0,tl.float32)
    denom=tl.full((BM,),0,tl.float32)
    peak=tl.full((BM,),-1.0e6,tl.float32)
    for offset in range(tl.cdiv(CHUNK,BN)):
        pos=begin+offset*BN+nn
        valid=(pos<seq)&(pos<begin+CHUNK)
        page=tl.load(PT+b*MAXP+pos//PAGE,valid,0)
        row=page*PAGE+pos%PAGE
        k=tl.load(KC+(row[None,:]*HK+h)*64+d[:,None],valid[None,:],0.0)
        v=tl.load(VC+(row[:,None]*HK+h)*64+d[None,:],valid[:,None],0.0).to(tl.bfloat16)
        mask=valid[None,:]&(r[:,None]<QL*group)&(pos[None,:]<=seq+token[:,None])
        if WINDOW>0:
            mask=mask&(pos[None,:]>=seq+token[:,None]-WINDOW+1)
        score=tl.dot(q,k)*SCALE*ks*1.4426950408889634
        score=tl.where(mask,score,-1.0e6)
        next_peak=tl.maximum(peak,tl.max(score,1))
        alpha=tl.exp2(peak-next_peak)
        prob=tl.where(mask,tl.exp2(score-next_peak[:,None]),0.0)
        acc=acc*alpha[:,None]
        acc=tl.dot(prob.to(tl.bfloat16),v,acc)
        denom=denom*alpha+tl.sum(prob,1)
        peak=next_peak
    # New keys are read directly from this call's projections. Only the final
    # split writes their cache rows, so no block reads a concurrent cache write.
    if s==SPLITS-1:
        j=tl.arange(0,BNEW);tnew=b*QL+j
        kn=tl.load(QKV+tnew[:,None]*stride+(HQ+h)*64+d[None,:],j[:,None]<QL,0.0).to(tl.float32)
        kr=tl.rsqrt(tl.sum(kn*kn,1)/64+1e-5)
        kn=kn*kr[:,None]
        if THETA!=0.0:
            kp=tl.load(QKV+tnew[:,None]*stride+(HQ+h)*64+(d[None,:]^1),j[:,None]<QL,0.0).to(tl.float32)*kr[:,None]
            new_position=tl.load(POS+tnew,j<QL,0).to(tl.float32)
            new_inv=tl.div_rn(1.0,libdevice.pow(THETA,(2*(d//2)).to(tl.float32)/64))
            new_angle=new_position[:,None]*new_inv[None,:]
            kn=kn*libdevice.cos(new_angle)+tl.where(d[None,:]%2==0,-kp,kp)*libdevice.sin(new_angle)
        kn=(kn.to(tl.bfloat16).to(tl.float32)/ks).to(tl.float8e4nv)
        vn=tl.load(QKV+tnew[:,None]*stride+(HQ+HK+h)*64+d[None,:],j[:,None]<QL,0.0).to(tl.float32)
        vn=(vn/vs).to(tl.float8e4nv)
        slot=tl.load(LOC+tnew,j<QL,0)
        tl.store(KC+(slot[:,None]*HK+h)*64+d[None,:],kn,j[:,None]<QL)
        tl.store(VC+(slot[:,None]*HK+h)*64+d[None,:],vn,j[:,None]<QL)
        new_mask=(r[:,None]<QL*group)&(j[None,:]<QL)&(j[None,:]<=token[:,None])
        if WINDOW>0:new_mask=new_mask&(j[None,:]>=token[:,None]-WINDOW+1)
        new_score=tl.dot(q,tl.trans(kn))*SCALE*ks*1.4426950408889634
        new_score=tl.where(new_mask,new_score,-1.0e6)
        new_next_peak=tl.maximum(peak,tl.max(new_score,1))
        new_alpha=tl.exp2(peak-new_next_peak)
        new_prob=tl.where(new_mask,tl.exp2(new_score-new_next_peak[:,None]),0.0)
        acc=acc*new_alpha[:,None]
        acc=tl.dot(new_prob.to(tl.bfloat16),vn.to(tl.bfloat16),acc)
        denom=denom*new_alpha+tl.sum(new_prob,1)
        peak=new_next_peak
    normalized=acc*vs/tl.maximum(denom[:,None],1.0e-30)
    logsum=tl.where(denom>0,peak+tl.log2(denom),-1.0e6)
    base=((b*HK+h)*SPLITS+s)*BM
    tl.store(P+(base+r[:,None])*64+d[None,:],normalized)
    tl.store(LSE+base+r,logsum)

@tr.jit
def _finish(P,LSE,QKV,Y,HQ:tl.constexpr,HK:tl.constexpr,QL:tl.constexpr,
            SPLITS:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr):
    t=tl.program_id(0);head=tl.program_id(1)
    b=t//QL;token=t%QL
    group:tl.constexpr=HQ//HK
    h=head//group;r=token*group+head%group
    s=tl.arange(0,BS);d=tl.arange(0,64)
    base=((b*HK+h)*SPLITS+s)*BM+r
    lse=tl.load(LSE+base,s<SPLITS,-1.0e6)
    mx=tl.max(lse,0)
    probability=tl.where(s<SPLITS,tl.exp2(lse-mx),0.0)
    probability=probability/tl.sum(probability,0)
    part=tl.load(P+base[:,None]*64+d[None,:],s[:,None]<SPLITS,0)
    a=tl.sum(part*probability[:,None],0).to(tl.bfloat16).to(tl.float32)
    norm=tl.rsqrt(tl.sum(a*a,0)/64+1e-5)
    gate=tl.load(QKV+t*(2*HQ+2*HK)*64+(HQ+2*HK)*64+head*64+d).to(tl.float32)
    out=a*norm/(1.0+tl.exp(-gate))
    tl.store(Y+(t*HQ+head)*64+d,out)

def kernel_fn(data):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len'];hq=cfg['num_q_heads'];hk=cfg['num_kv_heads']
    qkv=data['qkv']
    maxp=data['page_table'].shape[1];page=cfg['page_size'];window=cfg['window']
    span=window+ql-1 if window else maxp*page
    splits=min(128,max(4,tr.next_power_of_2(tr.cdiv(2048,b*hk))))
    chunk=tr.cdiv(span,splits*128)*128
    splits=tr.cdiv(span,chunk)
    bm=max(16,tr.next_power_of_2(ql*(hq//hk)))
    p=torch.empty((b,hk,splits,bm,64),device=qkv.device,dtype=torch.float32)
    lse=torch.empty((b,hk,splits,bm),device=qkv.device,dtype=torch.float32)
    y=torch.empty((b,ql,hq*64),device=qkv.device,dtype=torch.bfloat16)
    _attention_inline[(b,hk,splits)](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],data['page_table'],data['seq_lens'],p,lse,hq,hk,ql,page,maxp,window,cfg['sm_scale']*cfg['qk_scale_factor'],splits,chunk,bm,128,cfg['rope_theta'],max(16,tr.next_power_of_2(ql)),num_warps=4,num_stages=2)
    if b*ql>=32 and splits<=64:
        _finish_heads[(b*ql,tr.cdiv(hq,4))](p,lse,qkv,y,hq,hk,ql,splits,bm,tr.next_power_of_2(splits),4,num_warps=4)
    else:
        _finish[(b*ql,hq)](p,lse,qkv,y,hq,hk,ql,splits,bm,tr.next_power_of_2(splits),num_warps=4)
    return y

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

@tr.jit
def _finish_heads(P,LSE,QKV,Y,HQ:tl.constexpr,HK:tl.constexpr,QL:tl.constexpr,
                  SPLITS:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr,BH:tl.constexpr):
    t=tl.program_id(0);head=tl.program_id(1)*BH+tl.arange(0,BH)
    b=t//QL;token=t%QL
    group:tl.constexpr=HQ//HK
    h=head//group;r=token*group+head%group
    s=tl.arange(0,BS);d=tl.arange(0,64)
    base=((b*HK+h[:,None])*SPLITS+s[None,:])*BM+r[:,None]
    mask=(s[None,:]<SPLITS)&(head[:,None]<HQ)
    lse=tl.load(LSE+base,mask,-1.0e6)
    mx=tl.max(lse,1)
    probability=tl.where(mask,tl.exp2(lse-mx[:,None]),0.0)
    probability=probability/tl.sum(probability,1)[:,None]
    part=tl.load(P+base[:,:,None]*64+d[None,None,:],mask[:,:,None],0.0)
    a=tl.sum(part*probability[:,:,None],1).to(tl.bfloat16).to(tl.float32)
    norm=tl.rsqrt(tl.sum(a*a,1)/64+1e-5)
    gate=tl.load(QKV+t*(2*HQ+2*HK)*64+(HQ+2*HK)*64+head[:,None]*64+d[None,:],head[:,None]<HQ,0.0).to(tl.float32)
    out=a*norm[:,None]/(1.0+tl.exp(-gate))
    tl.store(Y+(t*HQ+head[:,None])*64+d[None,:],out,head[:,None]<HQ)
