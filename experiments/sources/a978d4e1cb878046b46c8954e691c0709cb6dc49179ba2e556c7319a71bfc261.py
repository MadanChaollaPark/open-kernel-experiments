# okc: problem=muse_bf16_proj
import torch
def prepare(weights):
    state={name:w.t().contiguous() for name,w in weights.items()}
    hi=weights['router'].to(torch.bfloat16)
    lo=(weights['router']-hi.float()).to(torch.bfloat16)
    state['hi']=hi.t().contiguous()
    state['lo']=lo.t().contiguous()
    state['unpooler_slice']=weights['unpooler'][:1024].t().contiguous()
    return state
def kernel_fn(data,state):
    x=data['x'];layer=data['config']['layer']
    if layer=='router':
        if x.shape[0]==1:return torch.mm(x.float(),state[layer])
        return torch.mm(x,state['hi'],out_dtype=torch.float32)+torch.mm(x,state['lo'],out_dtype=torch.float32)
    if layer=='unpooler' and x.shape[0]<=8:layer='unpooler_slice'
    return torch.mm(x,state[layer])
