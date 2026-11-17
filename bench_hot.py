#!/usr/bin/env python3
"""热/冷词表实测：树莓派上只加载热表时的速度。"""
import json, time, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0,"/srv/brainai")
from tok_pipe import Block
torch.set_num_threads(4)
ck=torch.load("/tmp/vlm_cont.pt",map_location="cpu",weights_only=False)
sd=ck["model"]; vocab=ck["vocab"]
d=384;H=8;LAYERS=6;P=4
class M(nn.Module):
    def __init__(self,Vn):
        super().__init__()
        self.proj=nn.Linear(512,d*P); self.tok=nn.Embedding(Vn,d)
        self.pos=nn.Embedding(64,d); self.imp=nn.Parameter(torch.zeros(Vn))
        self.blocks=nn.ModuleList([Block(d,H) for _ in range(LAYERS)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,Vn,bias=False); self.head.weight=self.tok.weight
    def forward(self,f,x):
        B,T=x.shape
        pre=self.proj(f).view(B,P,-1)
        h=torch.cat([pre,self.tok(x)],1)+self.pos(torch.arange(P+T)[None])
        impx=torch.cat([torch.ones(B,P),self.imp[x].clamp(min=1e-3)],1)
        for b in self.blocks: h=b(h,impx)
        return self.head(self.ln(h)[:,P:])
def load(Vn):
    m=M(Vn); cur=m.state_dict()
    for k,v in sd.items():
        if k in cur and cur[k].shape==v.shape: cur[k]=v
    # 截断到 Vn（热表）
    for k in ("tok.weight",):
        if Vn < cur[k].shape[0]: cur[k]=cur[k][:Vn]
    if Vn < cur["imp"].shape[0]: cur["imp"]=cur["imp"][:Vn]
    m.load_state_dict(cur, strict=False); return m.eval()
f=torch.randn(1,512)
for name,Vn in (("全词表 8895",8895),("热表 7306",7306),("热表 6000",6000)):
    m=load(Vn)
    with torch.no_grad():
        ids=[1]; t0=time.time()
        for _ in range(10):
            _=m(f,torch.tensor([ids])); ids.append(5)
        dt=(time.time()-t0)/10
    print(f"  {name:<12} → {1/dt:>5.1f} token/秒   一句话 {dt*20:.1f} 秒   参数 {sum(p.numel() for p in m.parameters())/1e6:.1f}M")
