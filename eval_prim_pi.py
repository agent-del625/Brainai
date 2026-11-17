#!/usr/bin/env python3
"""在树莓派上评估绘图模型（规范化槽位顺序）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/prim_v8_test.npz",allow_pickle=True)
cfg=np.load(ROOT/"data/prim_v8_train.npz",allow_pickle=True)["cfg"]
N,K,G,RES,SB,RB=int(cfg[0]),int(cfg[5]),int(cfg[1]),int(cfg[2]),int(cfg[3]),int(cfg[4])
class Enc(nn.Module):
    def __init__(self,d=256):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                             nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.Conv2d(128,d,3,2,1),nn.GELU(),
                             nn.AdaptiveAvgPool2d(1))
    def forward(self,x): return self.c(x).flatten(1)
class SlotNet(nn.Module):
    def __init__(self,d=256):
        super().__init__()
        self.enc=Enc(d); self.q=nn.Parameter(torch.randn(N,512)*0.02)
        self.mlp=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.LayerNorm(512))
        self.after=nn.Sequential(nn.Linear(512,512),nn.GELU(),nn.LayerNorm(512))
        self.h_id=nn.Linear(512,K); self.h_cell=nn.Linear(512,G*G); self.h_sub=nn.Linear(512,RES*RES)
        self.h_sb=nn.Linear(512,SB); self.h_rb=nn.Linear(512,RB); self.h_pres=nn.Linear(512,1)
        self.h_map=nn.Linear(512,G*G)
    def forward(self,x):
        z=self.mlp(self.enc(x)); h=self.after(z.unsqueeze(1)+self.q.unsqueeze(0))
        return (self.h_id(h),self.h_cell(h),self.h_sub(h),self.h_sb(h),self.h_rb(h),
                self.h_pres(h).squeeze(-1),self.h_map(h))
m=SlotNet(); m.load_state_dict(torch.load("/tmp/prim_gpu.pt",map_location="cpu")["model"]); m.eval()
I=torch.from_numpy(d["imgs"]).float().unsqueeze(1)/255.0
S=torch.from_numpy(d["slots"]).long(); M=torch.from_numpy(d["mask"]).float()
acc=np.zeros(6); nb=0
with torch.no_grad():
    for s in range(0,len(I),200):
        x=I[s:s+200]; sl=S[s:s+200]; mk=M[s:s+200]
        p_id,p_cell,p_sub,p_sb,p_rb,p_pres,_=m(x)
        acc+=np.array([float((p_id.argmax(-1)==sl[:,:,0]).float().mean()),
                       float((p_cell.argmax(-1)==sl[:,:,1]).float().mean()),
                       float((p_sub.argmax(-1)==sl[:,:,2]).float().mean()),
                       float((p_sb.argmax(-1)==sl[:,:,3]).float().mean()),
                       float((p_rb.argmax(-1)==sl[:,:,4]).float().mean()),
                       float(((torch.sigmoid(p_pres)>0.5).float()==mk).float().mean())])
        nb+=1
acc/=nb
print(f"积木ID {acc[0]*100:.1f}%  网格 {acc[1]*100:.1f}%  子格 {acc[2]*100:.1f}%  缩放 {acc[3]*100:.1f}%  旋转 {acc[4]*100:.1f}%  存在 {acc[5]*100:.1f}%")
print(f"随机基线 ID {100/K:.1f}%  网格 {100/(G*G):.1f}%")
