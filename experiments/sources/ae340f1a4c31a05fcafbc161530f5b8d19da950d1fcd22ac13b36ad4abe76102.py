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


@tr.jit
def _attention(Q,KC,VC,KS,VS,PT,SEQ,P,LSE,
               HQ:tl.constexpr,HK:tl.constexpr,QL:tl.constexpr,PAGE:tl.constexpr,MAXP:tl.constexpr,
               WINDOW:tl.constexpr,SCALE:tl.constexpr,SPLITS:tl.constexpr,CHUNK:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr):
    b=tl.program_id(0);h=tl.program_id(1);s=tl.program_id(2)
    group:tl.constexpr=HQ//HK
    r=tl.arange(0,BM);d=tl.arange(0,64);nn=tl.arange(0,BN)
    token=r//group;head=h*group+r%group
    seq=tl.load(SEQ+b)
    q=tl.load(Q+((b*QL+token[:,None])*HQ+head[:,None])*64+d[None,:],r[:,None]<QL*group,0.0)
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
        valid=(pos<seq+QL)&(pos<begin+CHUNK)
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

def _hybrid(data):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len'];hq=cfg['num_q_heads'];hk=cfg['num_kv_heads']
    qkv=data['qkv'];q=torch.empty((b*ql,hq,64),device=qkv.device,dtype=torch.float8_e4m3fn)
    if b*ql>=16:
        _prepare_heads[(b*ql,tr.cdiv(hq+hk,4))](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],4,num_warps=4)
    else:
        _prepare[(b*ql,hq+hk)](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],num_warps=4)
    maxp=data['page_table'].shape[1];page=cfg['page_size'];window=cfg['window']
    if window==0 and ql>1 and b>=8:
        descale={}
        if cfg['kv_scale']!=1.0:
            descale=dict(k_descale=data['k_scale'].reshape(1,1).expand(b,hk),v_descale=data['v_scale'].reshape(1,1).expand(b,hk))
        out=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=ql,softmax_scale=cfg['sm_scale']*cfg['qk_scale_factor'],causal=True,scheduler_metadata=data['scheduler_metadata'],**descale)
        if isinstance(out,tuple):out=out[0]
        gate=qkv.reshape(b*ql,-1)[:,(hq+2*hk)*64:]
        return rms_norm_gate(out.reshape(b*ql,hq*64).to(torch.bfloat16),gate,None,head_dim=64,eps=1e-5,has_norm=True).reshape(b,ql,hq*64)
    span=window+ql-1 if window else maxp*page
    bn=64 if window and b<8 else 128
    blocks=2048 if (window and 8<=b<32) or (not window and ql>1) else (512 if b>=32 else 1024)
    splits=min(128,max(4,tr.next_power_of_2(tr.cdiv(blocks,b*hk))))
    chunk=tr.cdiv(span,splits*bn)*bn
    splits=tr.cdiv(span,chunk)
    bm=max(16,tr.next_power_of_2(ql*(hq//hk)))
    p=torch.empty((b,hk,splits,bm,64),device=qkv.device,dtype=torch.float32)
    lse=torch.empty((b,hk,splits,bm),device=qkv.device,dtype=torch.float32)
    y=torch.empty((b,ql,hq*64),device=qkv.device,dtype=torch.bfloat16)
    _attention[(b,hk,splits)](q,data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],data['page_table'],data['seq_lens'],p,lse,hq,hk,ql,page,maxp,window,cfg['sm_scale']*cfg['qk_scale_factor'],splits,chunk,bm,bn,num_warps=4,num_stages=2)
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

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

@tr.jit
def inline__prepare(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    v = tl.load(QKV + t * stride + h * 64 + d).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 0) / 64 + 1e-05)
    v = v * r
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h * 64 + (d ^ 1)).to(tl.float32) * r
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle) + tl.where(d % 2 == 0, -partner, partner) * libdevice.sin(angle)
    v = v.to(tl.bfloat16).to(tl.float32)
    if h < HQ:
        tl.store(Q + (t * HQ + h) * 64 + d, v)
    else:
        slot = tl.load(LOC + t)
        ks = tl.load(KS)
        vs = tl.load(VS)
        val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h - HQ) * 64 + d).to(tl.float32)
        off = (slot * HK + h - HQ) * 64 + d
        tl.store(KC + off, v / ks)
        tl.store(VC + off, val / vs)

@tr.jit
def inline__attention_inline(QKV, POS, LOC, KC, VC, KS, VS, PT, SEQ, P, LSE, CS, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, PAGE: tl.constexpr, MAXP: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, THETA: tl.constexpr, BNEW: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    group: tl.constexpr = HQ // HK
    r = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nn = tl.arange(0, BN)
    token = r // group
    head = h * group + r % group
    seq = tl.load(SEQ + b)
    stride: tl.constexpr = (2 * HQ + 2 * HK) * 64
    tq = b * QL + token
    q = tl.load(QKV + tq[:, None] * stride + head[:, None] * 64 + d[None, :], r[:, None] < QL * group, 0.0).to(tl.float32)
    qnorm = tl.rsqrt(tl.sum(q * q, 1) / 64 + 1e-05)
    q = q * qnorm[:, None]
    if THETA != 0.0:
        qp = tl.load(QKV + tq[:, None] * stride + head[:, None] * 64 + (d[None, :] ^ 1), r[:, None] < QL * group, 0.0).to(tl.float32) * qnorm[:, None]
        position = tl.load(POS + tq, r < QL * group, 0)
        c = tl.load(CS + position[:, None] * 64 + d[None, :] // 2)
        sn = tl.load(CS + position[:, None] * 64 + 32 + d[None, :] // 2)
        q = q * c + tl.where(d[None, :] % 2 == 0, -qp, qp) * sn
    q = q.to(tl.bfloat16).to(tl.float8e4nv)
    ks = tl.load(KS)
    vs = tl.load(VS)
    if WINDOW > 0:
        first = tl.maximum(0, seq - WINDOW + 1)
    else:
        first = 0
    begin = first + s * CHUNK
    acc = tl.full((BM, 64), 0, tl.float32)
    denom = tl.full((BM,), 0, tl.float32)
    peak = tl.full((BM,), -1000000.0, tl.float32)
    for offset in range(tl.cdiv(CHUNK, BN)):
        pos = begin + offset * BN + nn
        valid = (pos < seq) & (pos < begin + CHUNK)
        page = tl.load(PT + b * MAXP + pos // PAGE, valid, 0)
        row = page * PAGE + pos % PAGE
        k = tl.load(KC + (row[None, :] * HK + h) * 64 + d[:, None], valid[None, :], 0.0)
        v = tl.load(VC + (row[:, None] * HK + h) * 64 + d[None, :], valid[:, None], 0.0).to(tl.bfloat16)
        mask = valid[None, :] & (r[:, None] < QL * group) & (pos[None, :] <= seq + token[:, None])
        if WINDOW > 0:
            mask = mask & (pos[None, :] >= seq + token[:, None] - WINDOW + 1)
        score = tl.dot(q, k) * SCALE * ks * 1.4426950408889634
        score = tl.where(mask, score, -1000000.0)
        next_peak = tl.maximum(peak, tl.max(score, 1))
        alpha = tl.exp2(peak - next_peak)
        prob = tl.where(mask, tl.exp2(score - next_peak[:, None]), 0.0)
        acc = acc * alpha[:, None]
        acc = tl.dot(prob.to(tl.bfloat16), v, acc)
        denom = denom * alpha + tl.sum(prob, 1)
        peak = next_peak
    if s == SPLITS - 1:
        j = tl.arange(0, BNEW)
        tnew = b * QL + j
        kn = tl.load(QKV + tnew[:, None] * stride + (HQ + h) * 64 + d[None, :], j[:, None] < QL, 0.0).to(tl.float32)
        kr = tl.rsqrt(tl.sum(kn * kn, 1) / 64 + 1e-05)
        kn = kn * kr[:, None]
        if THETA != 0.0:
            kp = tl.load(QKV + tnew[:, None] * stride + (HQ + h) * 64 + (d[None, :] ^ 1), j[:, None] < QL, 0.0).to(tl.float32) * kr[:, None]
            new_position = tl.load(POS + tnew, j < QL, 0)
            nc = tl.load(CS + new_position[:, None] * 64 + d[None, :] // 2)
            ns = tl.load(CS + new_position[:, None] * 64 + 32 + d[None, :] // 2)
            kn = kn * nc + tl.where(d[None, :] % 2 == 0, -kp, kp) * ns
        kn = (kn.to(tl.bfloat16).to(tl.float32) / ks).to(tl.float8e4nv)
        vn = tl.load(QKV + tnew[:, None] * stride + (HQ + HK + h) * 64 + d[None, :], j[:, None] < QL, 0.0).to(tl.float32)
        vn = (vn / vs).to(tl.float8e4nv)
        slot = tl.load(LOC + tnew, j < QL, 0)
        tl.store(KC + (slot[:, None] * HK + h) * 64 + d[None, :], kn, j[:, None] < QL)
        tl.store(VC + (slot[:, None] * HK + h) * 64 + d[None, :], vn, j[:, None] < QL)
        new_mask = (r[:, None] < QL * group) & (j[None, :] < QL) & (j[None, :] <= token[:, None])
        if WINDOW > 0:
            new_mask = new_mask & (j[None, :] >= token[:, None] - WINDOW + 1)
        new_score = tl.dot(q, tl.trans(kn)) * SCALE * ks * 1.4426950408889634
        new_score = tl.where(new_mask, new_score, -1000000.0)
        new_next_peak = tl.maximum(peak, tl.max(new_score, 1))
        new_alpha = tl.exp2(peak - new_next_peak)
        new_prob = tl.where(new_mask, tl.exp2(new_score - new_next_peak[:, None]), 0.0)
        acc = acc * new_alpha[:, None]
        acc = tl.dot(new_prob.to(tl.bfloat16), vn.to(tl.bfloat16), acc)
        denom = denom * new_alpha + tl.sum(new_prob, 1)
        peak = new_next_peak
    normalized = acc * vs / tl.maximum(denom[:, None], 1e-30)
    logsum = tl.where(denom > 0, peak + tl.log2(denom), -1000000.0)
    base = ((b * HK + h) * SPLITS + s) * BM
    tl.store(P + (base + r[:, None]) * 64 + d[None, :], normalized)
    tl.store(LSE + base + r, logsum)

@tr.jit
def inline__finish(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h) * SPLITS + s) * BM + r
    lse = tl.load(LSE + base, s < SPLITS, -1000000.0)
    mx = tl.max(lse, 0)
    probability = tl.where(s < SPLITS, tl.exp2(lse - mx), 0.0)
    probability = probability / tl.sum(probability, 0)
    part = tl.load(P + base[:, None] * 64 + d[None, :], s[:, None] < SPLITS, 0)
    a = tl.sum(part * probability[:, None], 0).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 0) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head * 64 + d).to(tl.float32)
    out = a * norm / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head) * 64 + d, out)

def inline_kernel_fn(data):
    cfg = data['config']
    b = cfg['batch']
    ql = cfg['q_len']
    hq = cfg['num_q_heads']
    hk = cfg['num_kv_heads']
    qkv = data['qkv']
    cs = inline__cos_sin(cfg['rope_theta'], qkv.device) if cfg['rope_theta'] else qkv
    maxp = data['page_table'].shape[1]
    page = cfg['page_size']
    window = cfg['window']
    span = window + ql - 1 if window else maxp * page
    bn = 128 if not window or b >= 8 else 64
    blocks = 2048 if not window or b >= 8 else 512 if ql > 1 else 1024
    splits = min(128, max(4, tr.next_power_of_2(tr.cdiv(blocks, b * hk))))
    chunk = tr.cdiv(span, splits * bn) * bn
    splits = tr.cdiv(span, chunk)
    bm = max(16, tr.next_power_of_2(ql * (hq // hk)))
    p = torch.empty((b, hk, splits, bm, 64), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((b, hk, splits, bm), device=qkv.device, dtype=torch.float32)
    y = torch.empty((b, ql, hq * 64), device=qkv.device, dtype=torch.bfloat16)
    inline__attention_inline[b, hk, splits](qkv, data['positions'], data['cache_loc'], data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], data['page_table'], data['seq_lens'], p, lse, cs, hq, hk, ql, page, maxp, window, cfg['sm_scale'] * cfg['qk_scale_factor'], splits, chunk, bm, bn, cfg['rope_theta'], max(16, tr.next_power_of_2(ql)), num_warps=4, num_stages=2)
    if b * ql >= 32 and splits <= 64:
        inline__finish_heads[b * ql, tr.cdiv(hq, 4)](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), 4, num_warps=4)
    else:
        inline__finish[b * ql, hq](p, lse, qkv, y, hq, hk, ql, splits, bm, tr.next_power_of_2(splits), num_warps=4)
    return y

@tr.jit
def inline__prepare_heads(QKV, POS, LOC, KC, VC, KS, VS, Q, HQ: tl.constexpr, HK: tl.constexpr, THETA: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    d = tl.arange(0, 64)
    stride = (2 * HQ + 2 * HK) * 64
    mask = h[:, None] < HQ + HK
    v = tl.load(QKV + t * stride + h[:, None] * 64 + d[None, :], mask, 0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(v * v, 1) / 64 + 1e-05)
    v = v * r[:, None]
    if THETA != 0.0:
        partner = tl.load(QKV + t * stride + h[:, None] * 64 + (d[None, :] ^ 1), mask, 0.0).to(tl.float32) * r[:, None]
        pos = tl.load(POS + t).to(tl.float32)
        inv_freq = tl.div_rn(1.0, libdevice.pow(THETA, (2 * (d // 2)).to(tl.float32) / 64))
        angle = pos * inv_freq
        v = v * libdevice.cos(angle)[None, :] + tl.where(d[None, :] % 2 == 0, -partner, partner) * libdevice.sin(angle)[None, :]
    v = v.to(tl.bfloat16).to(tl.float32)
    tl.store(Q + (t * HQ + h[:, None]) * 64 + d[None, :], v, h[:, None] < HQ)
    slot = tl.load(LOC + t)
    ks = tl.load(KS)
    vs = tl.load(VS)
    kv_mask = (h[:, None] >= HQ) & (h[:, None] < HQ + HK)
    val = tl.load(QKV + t * stride + (HQ + HK) * 64 + (h[:, None] - HQ) * 64 + d[None, :], kv_mask, 0.0).to(tl.float32)
    off = (slot * HK + h[:, None] - HQ) * 64 + d[None, :]
    tl.store(KC + off, v / ks, kv_mask)
    tl.store(VC + off, val / vs, kv_mask)

@tr.jit
def inline__finish_heads(P, LSE, QKV, Y, HQ: tl.constexpr, HK: tl.constexpr, QL: tl.constexpr, SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    b = t // QL
    token = t % QL
    group: tl.constexpr = HQ // HK
    h = head // group
    r = token * group + head % group
    s = tl.arange(0, BS)
    d = tl.arange(0, 64)
    base = ((b * HK + h[:, None]) * SPLITS + s[None, :]) * BM + r[:, None]
    mask = (s[None, :] < SPLITS) & (head[:, None] < HQ)
    lse = tl.load(LSE + base, mask, -1000000.0)
    mx = tl.max(lse, 1)
    probability = tl.where(mask, tl.exp2(lse - mx[:, None]), 0.0)
    probability = probability / tl.sum(probability, 1)[:, None]
    part = tl.load(P + base[:, :, None] * 64 + d[None, None, :], mask[:, :, None], 0.0)
    a = tl.sum(part * probability[:, :, None], 1).to(tl.bfloat16).to(tl.float32)
    norm = tl.rsqrt(tl.sum(a * a, 1) / 64 + 1e-05)
    gate = tl.load(QKV + t * (2 * HQ + 2 * HK) * 64 + (HQ + 2 * HK) * 64 + head[:, None] * 64 + d[None, :], head[:, None] < HQ, 0.0).to(tl.float32)
    out = a * norm[:, None] / (1.0 + tl.exp(-gate))
    tl.store(Y + (t * HQ + head[:, None]) * 64 + d[None, :], out, head[:, None] < HQ)
_ROPE = {}

def inline__cos_sin(theta, device):
    key = (theta, str(device))
    if key not in _ROPE:
        inv = 1.0 / theta ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
        frequencies = torch.outer(torch.arange(1 << 21, device=device, dtype=torch.float32), inv)
        _ROPE[key] = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    return _ROPE[key]

def _best_attention(data):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len']
    if b<8 or (cfg['window'] and b<32 and ql==1):
        return inline_kernel_fn(data)
    return _hybrid(data)

@tr.jit
def _one_chunk(Q,KC,VC,KS,VS,PT,SEQ,P,LSE,HQ:tl.constexpr,HK:tl.constexpr,QL:tl.constexpr,PAGE:tl.constexpr,MAXP:tl.constexpr,SCALE:tl.constexpr,SPLITS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,TM:tl.constexpr,B:tl.constexpr,QT:tl.constexpr):
    pid=tl.program_id(0);qt=pid%QT;pid=pid//QT
    b=pid%B;h=(pid//B)%HK;s=pid//(B*HK)
    group:tl.constexpr=HQ//HK
    r=qt*BM+tl.arange(0,BM);d=tl.arange(0,64);nn=tl.arange(0,BN)
    token=r//group;head=h*group+r%group
    seq=tl.load(SEQ+b);pos=s*BN+nn
    valid=pos<seq+QL
    page=tl.load(PT+b*MAXP+pos//PAGE,valid,0);row=page*PAGE+pos%PAGE
    q=tl.load(Q+((b*QL+token[:,None])*HQ+head[:,None])*64+d[None,:],r[:,None]<QL*group,0.0)
    k=tl.load(KC+(row[None,:]*HK+h)*64+d[:,None],valid[None,:],0.0)
    v=tl.load(VC+(row[:,None]*HK+h)*64+d[None,:],valid[:,None],0.0).to(tl.bfloat16)
    mask=valid[None,:]&(r[:,None]<QL*group)&(pos[None,:]<=seq+token[:,None])
    score=tl.dot(q,k)*SCALE*tl.load(KS)*1.4426950408889634
    score=tl.where(mask,score,-1.0e6)
    peak=tl.max(score,1)
    prob=tl.where(mask,tl.exp2(score-peak[:,None]),0.0)
    denom=tl.sum(prob,1)
    acc=tl.dot(prob.to(tl.bfloat16),v)
    normalized=acc*tl.load(VS)/tl.maximum(denom[:,None],1.0e-30)
    logsum=tl.where(denom>0,peak+tl.log2(denom),-1.0e6)
    base=((b*HK+h)*SPLITS+s)*TM+r
    tl.store(P+base[:,None]*64+d[None,:],normalized)
    tl.store(LSE+base,logsum)

def kernel_fn(data):
    cfg=data['config'];b=cfg['batch'];ql=cfg['q_len'];hq=cfg['num_q_heads'];hk=cfg['num_kv_heads']
    if cfg['window'] or b<8 or ql<2:return _best_attention(data)
    qkv=data['qkv'];q=torch.empty((b*ql,hq,64),device=qkv.device,dtype=torch.float8_e4m3fn)
    _prepare_heads[(b*ql,tr.cdiv(hq+hk,4))](qkv,data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],q,hq,hk,cfg['rope_theta'],4,num_warps=4)
    maxp=data['page_table'].shape[1];page=cfg['page_size'];span=maxp*page
    bn=512;tile=64;tm=max(16,tr.next_power_of_2(ql*(hq//hk)));qt=tr.cdiv(tm,tile);splits=tr.cdiv(span,bn)
    p=torch.empty((b,hk,splits,tm,64),device=qkv.device,dtype=torch.float32);lse=torch.empty((b,hk,splits,tm),device=qkv.device,dtype=torch.float32)
    y=torch.empty((b,ql,hq*64),device=qkv.device,dtype=torch.bfloat16)
    _one_chunk[(b*hk*splits*qt,)](q,data['k_cache'],data['v_cache'],data['k_scale'],data['v_scale'],data['page_table'],data['seq_lens'],p,lse,hq,hk,ql,page,maxp,cfg['sm_scale']*cfg['qk_scale_factor'],splits,tile,bn,tm,b,qt,num_warps=8,num_stages=1)
    _finish[(b*ql,hq)](p,lse,qkv,y,hq,hk,ql,splits,tm,tr.next_power_of_2(splits),num_warps=8)
    return y
