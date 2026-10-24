# okc: problem=muse_bf16_proj
import torch
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

def prepare(weights):
    state=dict(weights)
    w=weights['router'];hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    state['pair']=torch.cat([hi,lo],dim=0).t().contiguous()
    return state

def kernel_fn(data,state):
    x=data['x'];name=data['config']['layer'];n=data['config']['n']
    if name!='router':
        if x.shape[0]<=32:
            from flashinfer.gemm import mm_bf16
            return mm_bf16(x,state[name][:n].t(),backend='tinygemm')
        return _bf16_gemm_dispatch_impl(x,state[name][:n],None)
    both=torch.mm(x,state['pair'],out_dtype=torch.float32)
    return both[:,:n]+both[:,n:]
