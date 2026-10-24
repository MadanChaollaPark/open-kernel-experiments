"""Independent diagnostic preprocessing; source layout follows pinned NVIDIA Apache-2.0 interleaver."""
def interleave_weights(weight):
 import torch
 e,n,kh=weight.shape;k=kh*2;rows=e*n
 b=torch.arange(rows//2,device=weight.device)[:,None,None]
 p=torch.arange(k//64,device=weight.device)[None,:,None]
 lane=torch.arange(16,device=weight.device)[None,None,:]
 row=b//8*16+b%8;mma=lane//4;col=p*32+mma*8+lane%4
 first=row*kh+col;second=(row+8)*kh+col
 w=weight.reshape(-1).to(torch.int64)
 raw=w[first]|(w[second]<<8)|(w[first+4]<<16)|(w[second+4]<<24)
 outword=torch.zeros_like(raw)
 # Bit permutation copied from pinned moe_gemm_mixed_utils.cu; no numeric conversion.
 for mask,shift in [(0x80,24),(0x70,18),(0x8,12),(0x7,6),(0x8000,13),(0x7000,7),(0x800,1),(0x700,-5),(0x800000,2),(0x700000,-4),(0x80000,-10),(0x70000,-16),(0x80000000,-1),(0x10000000,1),(0x60000000,-3),(0x8000000,-13),(0x1000000,-11),(0x6000000,-15)]:
  bits=raw&mask;outword|=bits<<shift if shift>=0 else bits>>(-shift)
 dstrow=row+(mma%2)*8;dstcol=p*32+lane//8*16+(lane%4)*4
 dst=dstrow*kh+dstcol
 out=torch.empty_like(weight).reshape(-1)
 for j in range(4):out[(dst+j).reshape(-1)]=((outword>>(8*j))&255).to(torch.uint8).reshape(-1)
 return out.reshape_as(weight)

def fold_scales(scale):
 import torch
 e,n,g=scale.shape
 m=torch.arange(n,device=scale.device)[:,None]
 kg=torch.arange(g,device=scale.device)[None,:]
 offset=((m//64)*(g//8)+kg//8)*512+((m%64)%32)*16+((m%64)//32)*8+kg%8
 out=torch.empty((e,n*g),device=scale.device,dtype=torch.uint8)
 out[:,offset.reshape(-1)]=scale.reshape(e,-1)
 return out.reshape_as(scale)
