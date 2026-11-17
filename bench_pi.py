#!/usr/bin/env python3
"""在树莓派上实测：GPU 训好的 12M 模型能否跑得动。"""
import time, sys, math, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/srv/brainai")
from tok_pipe import Block
from pathlib import Path
ROOT=Path("/srv/brainai")
ck=torch.load("/tmp/vlm_gpu.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; Vn=len(vocab)
d=384;H=8;LAYERS=6;P=4;L=26
class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Linear(512,d*P)
        self.tok=nn.Embedding(Vn,d); self.pos=nn.Embedding(P+L+4,d)
        self.imp=nn.Parameter(torch.zeros(Vn))
        self.blocks=nn.ModuleList([Block(d,H) for _ in range(LAYERS)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,Vn,bias=False); self.head.weight=self.tok.weight
    def forward(self,f,x):
        B,T=x.shape
        pre=self.proj(f).view(B,P,-1)
        h=torch.cat([pre,self.tok(x)],1)+self.pos(torch.arange(P+T)[None])
        impx=torch.cat([torch.ones(B,P),self.imp[x].clamp(min=1e-3)],1)
        for b in self.blocks: h=b(h,impx)
        return self.head(self.ln(h)[:,P:])
m=VLM(); sd=ck["model"]; cur=m.state_dict()
for k,v in sd.items():
    if k in cur and cur[k].shape==v.shape: cur[k]=v
m.load_state_dict(cur); m.eval()
n_par=sum(p.numel() for p in m.parameters())
print(f"参数量 {n_par/1e6:.1f}M  词表 {Vn}")
torch.set_num_threads(4)
f=torch.randn(1,512)
with torch.no_grad():
    ids=[1]
    t0=time.time()
    for _ in range(20):
        x=torch.tensor([ids])
        _=m(f,x)
        ids.append(5)
    dt=(time.time()-t0)/20
print(f"每生成一个 token {dt*1000:.0f} ms → 约 {1/dt:.1f} token/秒")
print(f"生成 20 token 的句子需要 {dt*20:.1f} 秒")
