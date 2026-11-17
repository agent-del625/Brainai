#!/usr/bin/env python3
"""A：概念→积木 改成概率采样（每次不一样），并输出对照图。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
tr=np.load(ROOT/"data/prim_v6_train.npz",allow_pickle=True)
cats=[str(c) for c in np.load(ROOT/"data/prim_v6_cats.npy")]
ck=torch.load(ROOT/"ckpt/concept_gen.pt",map_location="cpu",weights_only=False)
cent=torch.as_tensor(ck["cent"]).float(); N,G,RES,SB,RB,K=[int(x) for x in ck["cfg"]]
class Gen(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.emb=nn.Embedding(len(cats),h); self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.mlp=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def forward(self,y):
        e=self.emb(y).unsqueeze(1); h=self.mlp(e+self.q.unsqueeze(0))
        return self.h_id(h),self.h_cell(h),self.h_res(h),self.h_sc(h),self.h_rot(h)
gen=Gen(); gen.load_state_dict(ck["model"]); gen.eval()
def render(items):
    img=np.zeros((S,S),np.uint8)
    for pid,cell,res,sb,rb in items:
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th); sh=cent[int(pid)].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),255,1)
    return img
rows=[]
with torch.no_grad():
    for ci,c in enumerate(cats):
        outs=gen(torch.tensor([ci]))
        probs=[torch.softmax(o[0],-1) for o in outs]
        tiles=[]
        for _ in range(5):
            items=[]
            for t in range(N):
                vals=[int(torch.multinomial(probs[k][t],1)) for k in range(5)]
                items.append(tuple(vals))
            tiles.append(render(items))
        rows.append(np.hstack(tiles))
img=np.vstack(rows)
cv2.imwrite(str(ROOT/"logs/turtle/concept_sampled.png"), img)
print("图:", ROOT/"logs/turtle/concept_sampled.png", "行:", cats)
