#!/usr/bin/env python3
"""C：看图作画——CLIP 特征 → 积木配方 → 画出简笔画。

训练用 QuickDraw 简笔画的 CLIP 特征（图 ↔ 配方 成对）；
推理时喂真实照片的 CLIP 特征。
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, os
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/qd_clip.npz",allow_pickle=True)
tr=np.load(ROOT/"data/prim_v6_train.npz",allow_pickle=True)
cent=torch.as_tensor(tr["cent"]).float(); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
F_=torch.from_numpy(d["feats"]); SL=torch.from_numpy(d["slots"]); MK=torch.from_numpy(d["mask"]).float()
n=len(F_); p=np.random.permutation(n); a=int(n*0.9)
tr_i,va_i=p[:a],p[a:]
print("样本",n,"槽位",N,"K",K)

class P2S(nn.Module):
    def __init__(self,h=384):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h),nn.GELU())
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def forward(self,f):
        h=self.net(f).unsqueeze(1)+self.q.unsqueeze(0)
        return self.h_id(h),self.h_cell(h),self.h_res(h),self.h_sc(h),self.h_rot(h)

m=P2S(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
def loss_fn(outs,X,M):
    L=0.; w=0.
    for lg,idx in zip(outs,range(5)):
        ce=F.cross_entropy(lg.transpose(1,2), X[:,:,idx], reduction='none')
        L=L+(ce*M).sum(); w+=M.sum()
    return L/w/5
for step in range(1,4001):
    i=torch.from_numpy(np.random.choice(tr_i,48))
    loss=loss_fn(m(F_[i]),SL[i],MK[i])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%400==0: print(f"step {step} loss {loss.item():.3f}", flush=True)

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

def iou(a,b):
    A=cv2.dilate(a,np.ones((3,3),np.uint8))>0; B=cv2.dilate(b,np.ones((3,3),np.uint8))>0
    return float((A&B).sum()/max((A|B).sum(),1))

# 评估：用简笔画自己的 CLIP 特征还原（成对数据上的上限参考）
m.eval(); sc=[]
with torch.no_grad():
    for j in va_i[:200]:
        outs=m(F_[j:j+1]); pr=[o[0].argmax(-1) for o in outs]
        nsl=int(MK[j].sum())
        rec=render([tuple(int(pr[k][t]) for k in range(5)) for t in range(nsl)])
        gt=(tr["imgs"][0]*0+0)
        sc.append(rec)
torch.save({"model":m.state_dict(),"cent":cent,"cfg":tr["cfg"]}, ROOT/"ckpt/p2s.pt")
print("CLIP→配方 训练完成，权重已存")

# 真实照片：Flickr8k CLIP 特征 → 画简笔画
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
pf=torch.from_numpy(z["feats"]); caps=[str(c) for c in z["captions"]]
want=["cat","dog","fish","bird","train","car","tree","house","sun","star","flower"]
sel=[]
for k,w in enumerate(want):
    idx=[i for i,c in enumerate(caps) if w in c.lower()][:2]
    for i in idx: sel.append((w,i))
rows=[]; tiles=[]
with torch.no_grad():
    for w,i in sel:
        outs=m(pf[i:i+1]); pr=[o[0].argmax(-1) for o in outs]
        nsl=int(MK.mean(0).sum().round().item())
        rec=render([tuple(int(pr[k][t]) for k in range(5)) for t in range(nsl)])
        rows.append((w,i,rec))
print("生成了",len(rows),"张「看照片画的简笔画」")
tiles=[r[2] for r in rows]
blank=np.zeros((S,S),np.uint8)
while len(tiles)%5: tiles.append(blank)
cv2.imwrite(str(ROOT/"logs/turtle/photo2sketch.png"), np.vstack([np.hstack(tiles[k:k+5]) for k in range(0,len(tiles),5)]))
print("图:", ROOT/"logs/turtle/photo2sketch.png")
torch.save({"model":m.state_dict(),"cent":cent,"cfg":tr["cfg"]}, ROOT/"ckpt/p2s.pt")
