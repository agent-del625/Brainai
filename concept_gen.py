#!/usr/bin/env python3
"""概念 → 积木配方：训练「类别 → 槽位序列」生成器，并画出来。

评测：用独立的「图 → 类别」分类器给生成的图打分（生成的图像不像该类别）。
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, os
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
tr=np.load(ROOT/"data/prim_v6_train.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v6_test.npz",allow_pickle=True)
cats=[str(c) for c in np.load(ROOT/"data/prim_v6_cats.npy")]
cent=torch.from_numpy(tr["cent"]); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
Xtr=torch.from_numpy(tr["slots"]); Mtr=torch.from_numpy(tr["mask"]).float(); Itr=torch.from_numpy(tr["imgs"]).float().unsqueeze(1)/255.; Ytr=torch.from_numpy(tr["y"])
Ite=torch.from_numpy(te["imgs"]).float().unsqueeze(1)/255.; Yte=torch.from_numpy(te["y"])
NC=len(cats); print("类别",NC,"训练",len(Itr))

# ---------- 1) 分类器：图 → 类别（用来给生成结果打分） ----------
class Cls(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,24,3,2,1),nn.GELU(),nn.Conv2d(24,48,3,2,1),nn.GELU(),
                               nn.Conv2d(48,96,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(96*8*8,256),nn.GELU(),nn.Linear(256,NC))
    def forward(self,x): return self.net(x)
cls=Cls(); oc=torch.optim.AdamW(cls.parameters(),lr=1e-3)
for step in range(1,1201):
    i=torch.randint(0,len(Itr),(64,))
    loss=F.cross_entropy(cls(Itr[i]),Ytr[i]); oc.zero_grad(); loss.backward(); oc.step()
with torch.no_grad():
    acc=(cls(Ite).argmax(1)==Yte).float().mean().item()
print(f"分类器（图→类别，11 类）测试集准确率 {acc*100:.0f}%")
cls.eval()

# ---------- 2) 生成器：类别 → 槽位 ----------
class Gen(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.emb=nn.Embedding(NC,h)
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.mlp=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def forward(self,y):
        e=self.emb(y).unsqueeze(1)
        h=self.mlp(e+self.q.unsqueeze(0))
        return self.h_id(h),self.h_cell(h),self.h_res(h),self.h_sc(h),self.h_rot(h)
gen=Gen(); og=torch.optim.AdamW(gen.parameters(),lr=1e-3)
for step in range(1,4001):
    i=torch.randint(0,len(Itr),(64,))
    outs=gen(Ytr[i]); M=Mtr[i]
    L=0.; w=0.
    for lg,idx in zip(outs,range(5)):
        ce=F.cross_entropy(lg.transpose(1,2), Xtr[i][:,:,idx], reduction='none')
        L=L+(ce*M).sum(); w+=M.sum()
    loss=L/w/5
    og.zero_grad(); loss.backward(); og.step()
    if step%800==0: print(f"[生成器] step {step} loss {loss.item():.3f}", flush=True)

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

gen.eval()
rows=[]; agree=0; tot=0
with torch.no_grad():
    for ci,c in enumerate(cats):
        tiles=[]
        for _ in range(4):
            outs=gen(torch.tensor([ci]))
            pr=[o[0].argmax(-1) for o in outs]
            # 用该类的平均槽位数裁剪长度（从训练集中统计）
            nsl=int(Mtr[Ytr==ci].sum(1).float().mean().round().item())
            items=[tuple(int(pr[k][t]) for k in range(5)) for t in range(nsl)]
            img=render(items)
            tiles.append(img)
            with torch.no_grad():
                pred=int(cls((torch.from_numpy(img[None,None]).float()/255.).clamp(0,1)).argmax())
            tot+=1; agree+= int(pred==ci)
        rows.append(np.hstack(tiles))
print(f"生成图被分类器认成目标类别的比例：{agree}/{tot} = {agree/tot*100:.0f}%")
out=np.vstack(rows)
cv2.imwrite(str(ROOT/"logs/turtle/concept_gen.png"), out)
torch.save({"model":gen.state_dict(),"cent":cent,"cfg":tr["cfg"]}, ROOT/"ckpt/concept_gen.pt")
print("图:", ROOT/"logs/turtle/concept_gen.png")
print("行顺序:", cats)
