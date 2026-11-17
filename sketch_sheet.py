#!/usr/bin/env python3
"""整理一版「模型画的画」：上行=真实简笔画，下行=模型画的（概念 → 积木）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=96
tr=np.load(ROOT/"data/prim_v6_train.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v6_test.npz",allow_pickle=True)
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
def render(items,size=S):
    img=np.zeros((size,size),np.uint8)
    for pid,cell,res,sb,rb in items:
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th); sh=cent[int(pid)].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*size/2-0.5,(sh[:,1]+1)*size/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),255,1)
    return img
Yte=te["y"]; It=te["imgs"]
top=[]; bot=[]
with torch.no_grad():
    for ci,c in enumerate(cats):
        j=int(np.where(Yte==ci)[0][0])
        gt=cv2.resize(255-It[j],(S,S),interpolation=cv2.INTER_NEAREST)
        outs=gen(torch.tensor([ci])); probs=[torch.softmax(o[0],-1) for o in outs]
        nsl=int(np.asarray(tr["mask"])[np.asarray(tr["y"])==ci].sum(1).mean().round())
        items=[tuple(int(torch.multinomial(probs[k][t],1)) for k in range(5)) for t in range(nsl)]
        mine=255-render(items,S)
        top.append(gt); bot.append(mine)
def row(tiles,gap=6):
    out=[]
    for t in tiles: out += [t, np.full((S,gap),80,np.uint8)]
    return np.hstack(out[:-1])
img=np.vstack([np.full((30,row(top).shape[1]),255,np.uint8), row(top), np.full((10,row(top).shape[1]),255,np.uint8), row(bot)])
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cv2.putText(img,"TOP: real QuickDraw   BOTTOM: model drew from the word",(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
for i,c in enumerate(cats):
    cv2.putText(img,c,(4+i*(S+6),45),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,128,0),1)
cv2.imwrite(str(ROOT/"logs/turtle/sketch_sheet.png"), img)
print("图:", ROOT/"logs/turtle/sketch_sheet.png", img.shape)
