#!/usr/bin/env python3
"""词表增长如何影响树莓派速度：Vn 从 5704 到 25000 实测。"""
import time, sys, torch, torch.nn as nn
sys.path.insert(0,"/srv/brainai")
from tok_pipe import Block
torch.set_num_threads(4)
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
f=torch.randn(1,512)
for Vn in (5704, 8895, 15000, 25000):
    m=M(Vn).eval()
    with torch.no_grad():
        ids=[1]
        t0=time.time()
        for _ in range(10):
            _=m(f,torch.tensor([ids])); ids.append(5)
        dt=(time.time()-t0)/10
    print(f"  词表 {Vn:>6} → {1/dt:>5.1f} token/秒   一句话 {dt*20:.1f} 秒")
    del m
