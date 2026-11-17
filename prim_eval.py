#!/usr/bin/env python3
"""积木重建评估：膨胀 IoU + 倒角距离 + 对照图。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/prim_ds2.npz",allow_pickle=True)
slots=torch.from_numpy(d["slots"]); mask=torch.from_numpy(d["mask"]).float()
imgs=torch.from_numpy(d["imgs"]).float().unsqueeze(1)/255.0
cent=torch.from_numpy(d["cent"]); N,G,RES,SB=[int(x) for x in d["cfg"]]; K=len(cent)
class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                               nn.Conv2d(64,96,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(96*8*8,h),nn.GELU())
        self.emb=nn.Embedding(K+1,h); self.gru=nn.GRU(h*2,h,batch_first=True)
        self.id_h=nn.Linear(h,K); self.cell_h=nn.Linear(h,G*G); self.res_h=nn.Linear(h,RES*RES); self.sc_h=nn.Linear(h,SB)
    def forward(self,x,tgt):
        f=self.cnn(x); bos=torch.full((x.size(0),1),K,dtype=torch.long)
        inp=torch.cat([bos,tgt[:,:,0]],1)[:,:-1]; e=self.emb(inp)
        o,_=self.gru(torch.cat([e,f.unsqueeze(1).expand(-1,e.size(1),-1)],-1))
        return self.id_h(o), self.cell_h(o), self.res_h(o), self.sc_h(o)
ck=torch.load(ROOT/"ckpt/prim2.pt",map_location="cpu",weights_only=False)
m=Net(); m.load_state_dict(ck["model"]); m.eval()
print("checkpoint step", ck.get("step"))
def render(ids,cells,ress,scs):
    img=np.zeros((S,S),np.uint8)
    for pid,cell,res,sb in zip(ids,cells,ress,scs):
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6
        sh=cent[int(pid)].numpy()*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img
def chamfer(a,b):
    ya,xa=np.nonzero(a>0); yb,xb=np.nonzero(b>0)
    if len(xa)==0 or len(xb)==0: return 99.
    A=np.stack([xa,ya],1).astype(np.float32); B=np.stack([xb,yb],1).astype(np.float32)
    D=((A[:,None]-B[None])**2).sum(-1)
    return float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2
n=len(imgs); perm=torch.randperm(n); te=perm[int(n*0.9):]
ious=[]; chs=[]; rows=[]
with torch.no_grad():
    for i in te[:150].tolist():
        f=m.cnn(imgs[i:i+1]); prev=torch.tensor([[K]]); h=None
        ids=[];cells=[];ress=[];scs=[]
        for t in range(N):
            o,h=m.gru(torch.cat([m.emb(prev),f.unsqueeze(1)],-1),h)
            pid=int(m.id_h(o)[0,0].argmax()); cell=int(m.cell_h(o)[0,0].argmax())
            res=int(m.res_h(o)[0,0].argmax()); sc=int(m.sc_h(o)[0,0].argmax())
            ids.append(pid);cells.append(cell);ress.append(res);scs.append(sc); prev=torch.tensor([[pid]])
            if t+1>=int(mask[i].sum()): break
        rec=render(ids,cells,ress,scs)
        gt=(d["imgs"][i]>127).astype(np.uint8)
        o1=cv2.dilate(gt,np.ones((3,3),np.uint8)); r1=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o1>0;B=r1>0
        ious.append((A&B).sum()/max((A|B).sum(),1)); chs.append(chamfer(gt,rec))
        if len(rows)<5: rows.append(np.hstack([(255*(1-gt)).astype(np.uint8),np.full((S,4),200,np.uint8),(255*(1-rec)).astype(np.uint8)]))
print(f"膨胀IoU {np.mean(ious)*100:.1f}%  倒角 {np.mean(chs):.2f}px  上限约 54%")
cv2.imwrite(str(ROOT/"logs/turtle/prim_final.png"), np.vstack(rows))
print("图:", ROOT/"logs/turtle/prim_final.png")
