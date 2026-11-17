#!/usr/bin/env python3
"""用当前最优权重重画一组测试图，生成对照图。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=128
tr=np.load(ROOT/"data/prim_v3_train.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v3_test.npz",allow_pickle=True)
cent=torch.from_numpy(tr["cent"]); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
Xte=torch.from_numpy(te["slots"]); Mte=torch.from_numpy(te["mask"]).float()
Ite=torch.from_numpy(te["imgs"]).float().unsqueeze(1)/255.
class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                               nn.Conv2d(64,96,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(96*8*8,h),nn.GELU())
        self.emb=nn.Embedding(K+1,h); self.gru=nn.GRU(h*2,h,batch_first=True)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def step(self,prev,f,h):
        o,h=self.gru(torch.cat([self.emb(prev),f.unsqueeze(1)],-1),h); return o,h
ck=torch.load(ROOT/"ckpt/prim3.pt",map_location="cpu",weights_only=False)
m=Net(); m.load_state_dict(ck["model"]); m.eval()
print("最优 step", ck.get("step"), "val IoU", round(ck.get("best",0)*100,1), "%")
def render(items):
    img=np.zeros((S,S),np.uint8)
    for pid,cell,res,sb,rb in items:
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th); sh=cent[int(pid)].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img
rows=[]
with torch.no_grad():
    for i in range(6):
        f=m.cnn(Ite[i:i+1]); h=None; prev=torch.full((1,1),K,dtype=torch.long); items=[]
        for t in range(N):
            o,h=m.step(prev,f,h); o=o[:,0]
            pid=int(m.h_id(o).argmax()); cell=int(m.h_cell(o).argmax()); res=int(m.h_res(o).argmax())
            sb=int(m.h_sc(o).argmax()); rb=int(m.h_rot(o).argmax())
            items.append((pid,cell,res,sb,rb)); prev=torch.tensor([[pid]])
            if t+1>=int(Mte[i].sum()): break
        rec=render(items)
        gt=cv2.resize((Ite[i,0].numpy()*255).clip(0,255).astype(np.uint8),(S,S),interpolation=cv2.INTER_NEAREST)
        gt=255-gt
        rows.append(np.hstack([gt,np.full((S,6),150,np.uint8),(255*(1-rec)).astype(np.uint8)]))
out=np.vstack([np.hstack([r,np.full((S,14),200,np.uint8)]) for r in rows])
out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
cv2.putText(out,"LEFT = real sketch   RIGHT = our reconstruction",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),1)
cv2.imwrite(str(ROOT/"logs/turtle/prim_v3_show.png"), out)
print("图:", ROOT/"logs/turtle/prim_v3_show.png")
