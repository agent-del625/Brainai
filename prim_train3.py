#!/usr/bin/env python3
"""积木重建 v3：K=96 + 旋转/缩放，调度采样修曝光偏差，验证集早停，测试集评估。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, os
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
tr=np.load(ROOT/"data/prim_v3_train.npz",allow_pickle=True)
va=np.load(ROOT/"data/prim_v3_val.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v3_test.npz",allow_pickle=True)
cent=torch.from_numpy(tr["cent"]); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
def T(d,k): return torch.from_numpy(d[k])
Xtr,Mtr,Itr=T(tr,"slots"),T(tr,"mask").float(),T(tr,"imgs").float().unsqueeze(1)/255.
Xva,Mva,Iva=T(va,"slots"),T(va,"mask").float(),T(va,"imgs").float().unsqueeze(1)/255.
Xte,Mte,Ite=T(te,"slots"),T(te,"mask").float(),T(te,"imgs").float().unsqueeze(1)/255.
print("训练",len(Itr),"验证",len(Iva),"测试",len(Ite),"K",K)

class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                               nn.Conv2d(64,96,3,2,1),nn.GELU(),nn.Flatten(),nn.Linear(96*8*8,h),nn.GELU())
        self.emb=nn.Embedding(K+1,h); self.gru=nn.GRU(h*2,h,batch_first=True)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def step(self,prev,f,h):
        o,h=self.gru(torch.cat([self.emb(prev),f.unsqueeze(1)],-1),h)
        return o,h
    def heads(self,o):
        return self.h_id(o),self.h_cell(o),self.h_res(o),self.h_sc(o),self.h_rot(o)

m=Net(); opt=torch.optim.AdamW(m.parameters(), lr=6e-4)
def batch(d_slots,d_mask,d_img,idx):
    return d_slots[idx],d_mask[idx],d_img[idx]
def run_batch(X,M,I,ss_prob):
    f=m.cnn(I); h=None; prev=torch.full((I.size(0),1),K,dtype=torch.long)
    loss=0.; w=0.
    for t in range(N):
        o,h=m.step(prev,f,h)
        outs=m.heads(o[:,0])
        tgt=[X[:,t,0],X[:,t,1],X[:,t,2],X[:,t,3],X[:,t,4]]
        msk=M[:,t]
        for lg,tg in zip(outs,tgt):
            ce=F.cross_entropy(lg,tg,reduction='none')
            loss=loss+(ce*msk).sum(); w+=msk.sum()
        # 调度采样：以 ss_prob 用模型自己的预测作为下一步输入
        pid=outs[0].argmax(1)
        use_self=(torch.rand(I.size(0))<ss_prob)
        prev=torch.where(use_self, pid, X[:,t,0]).unsqueeze(1)
    return loss/w/5

def evaluate(X,M,I,limit=300):
    m.eval(); ious=[]; chs=[]
    with torch.no_grad():
        for i in range(min(limit,len(I))):
            f=m.cnn(I[i:i+1]); h=None; prev=torch.full((1,1),K,dtype=torch.long)
            items=[]
            for t in range(N):
                o,h=m.step(prev,f,h); outs=m.heads(o[:,0])
                pid=int(outs[0].argmax()); cell=int(outs[1].argmax()); res=int(outs[2].argmax())
                sb=int(outs[3].argmax()); rb=int(outs[4].argmax())
                items.append((pid,cell,res,sb,rb)); prev=torch.tensor([[pid]])
                if t+1>=int(M[i].sum()): break
            rec=render(items); gt=(I[i,0].numpy()>0.5).astype(np.uint8)
            o1=cv2.dilate(gt,np.ones((3,3),np.uint8)); r1=cv2.dilate(rec,np.ones((3,3),np.uint8))
            A=o1>0;B=r1>0; ious.append((A&B).sum()/max((A|B).sum(),1))
            ya,xa=np.nonzero(gt>0); yb,xb=np.nonzero(rec>0)
            if len(xa) and len(xb):
                A2=np.stack([xa,ya],1).astype(np.float32); B2=np.stack([xb,yb],1).astype(np.float32)
                D=((A2[:,None]-B2[None])**2).sum(-1)
                chs.append(float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2)
    m.train()
    return float(np.mean(ious)), float(np.mean(chs)) if chs else 99.

def render(items):
    img=np.zeros((S,S),np.uint8)
    for pid,cell,res,sb,rb in items:
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th)
        sh=cent[int(pid)].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img

best=0; START=1
if os.environ.get("RESUME")=="1" and (ROOT/"ckpt/prim3.pt").exists():
    st=torch.load(ROOT/"ckpt/prim3.pt",map_location="cpu",weights_only=False)
    m.load_state_dict(st["model"]); START=st.get("step",1)+1; best=st.get("best",0); print("续训自",START)
for step in range(START, 12001):
    i=torch.randint(0,len(Itr),(48,))
    ss=min(0.5, step/4000*0.5)
    loss=run_batch(Xtr[i],Mtr[i],Itr[i],ss)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%200==0:
        iou,ch=evaluate(Xva,Mva,Iva,150)
        print(f"step {step} loss {loss.item():.3f} ss {ss:.2f} | val IoU {iou*100:.1f}% 倒角 {ch:.2f}", flush=True)
        if iou>best:
            best=iou; torch.save({"model":m.state_dict(),"cent":cent,"cfg":tr["cfg"],"step":step,"best":best}, ROOT/"ckpt/prim3.pt")
print("最优 val IoU", round(best*100,1), "%")
iou,ch=evaluate(Xte,Mte,Ite,400)
print(f"测试集：膨胀IoU {iou*100:.1f}%  倒角 {ch:.2f}px（上限约 57%）")
