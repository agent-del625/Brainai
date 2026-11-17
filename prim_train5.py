#!/usr/bin/env python3
"""积木重建 v5：把「上一槽的完整信息（ID+位置+尺度+旋转）」喂回解码器，
   让模型知道"我已经画到哪了"，而不是只记得上一个积木的编号。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, os
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
tr=np.load(ROOT/"data/prim_v3_train.npz",allow_pickle=True)
va=np.load(ROOT/"data/prim_v3_val.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v3_test.npz",allow_pickle=True)
cent=torch.from_numpy(tr["cent"]); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
Xtr=torch.from_numpy(tr["slots"]); Mtr=torch.from_numpy(tr["mask"]).float()
Itr=torch.from_numpy(tr["imgs"]).float().unsqueeze(1)/255.
Xva=torch.from_numpy(va["slots"]); Mva=torch.from_numpy(va["mask"]).float()
Iva=torch.from_numpy(va["imgs"]).float().unsqueeze(1)/255.
Xte=torch.from_numpy(te["slots"]); Mte=torch.from_numpy(te["mask"]).float()
Ite=torch.from_numpy(te["imgs"]).float().unsqueeze(1)/255.
print("训练",len(Itr),"K",K)

class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.feat=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                                nn.Conv2d(64,96,3,2,1),nn.GELU())
        self.proj=nn.Conv2d(96,h,1); self.pos=nn.Parameter(torch.randn(1,64,h)*0.02)
        self.e_id=nn.Embedding(K+1,h); self.e_cell=nn.Embedding(G*G,h); self.e_res=nn.Embedding(RES*RES,h)
        self.e_sc=nn.Embedding(SB,h); self.e_rot=nn.Embedding(RB,h)
        self.gru=nn.GRU(h,h,batch_first=True)
        self.att=nn.MultiheadAttention(h,4,batch_first=True)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def encode(self,x): return self.proj(self.feat(x)).flatten(2).transpose(1,2)+self.pos
    def prev_vec(self,ids,cells,ress,scs,rots):
        return self.e_id(ids)+self.e_cell(cells)+self.e_res(ress)+self.e_sc(scs)+self.e_rot(rots)
    def step(self,pv,mem,h):
        o,h=self.gru(pv.unsqueeze(1),h)
        a,_=self.att(o,mem,mem)
        return o+a,h
    def heads(self,o):
        return self.h_id(o),self.h_cell(o),self.h_res(o),self.h_sc(o),self.h_rot(o)

m=Net(); opt=torch.optim.AdamW(m.parameters(), lr=8e-4)
def run_batch(X,M,I,ss):
    mem=m.encode(I); h=None
    zeros=torch.zeros(I.size(0),dtype=torch.long)
    pv=m.prev_vec(torch.full_like(zeros,K),zeros,zeros,zeros,zeros)
    loss=0.; w=0.
    for t in range(N):
        o,h=m.step(pv,mem,h); outs=m.heads(o[:,0]); msk=M[:,t]
        for lg,idx in zip(outs,range(5)):
            ce=F.cross_entropy(lg,X[:,t,idx],reduction='none')
            loss=loss+(ce*msk).sum(); w+=msk.sum()
        preds=[o_.argmax(1) for o_ in outs]
        gt=[X[:,t,k] for k in range(5)]
        use=(torch.rand(I.size(0))<ss).unsqueeze(1)
        cur=torch.stack([torch.where(use[:,0],p,g) for p,g in zip(preds,gt)],1)
        pv=m.prev_vec(cur[:,0],cur[:,1],cur[:,2],cur[:,3],cur[:,4])
    return loss/w/5

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

def evaluate(X,M,I,limit=150,collect=False):
    m.eval(); ious=[]; chs=[]; pairs=[]; accs=np.zeros(5); cnt=0
    with torch.no_grad():
        for i in range(min(limit,len(I))):
            mem=m.encode(I[i:i+1]); h=None
            z=torch.zeros(1,dtype=torch.long); pv=m.prev_vec(torch.full_like(z,K),z,z,z,z)
            items=[]
            for t in range(N):
                o,h=m.step(pv,mem,h); outs=m.heads(o[:,0])
                pr=[int(x.argmax()) for x in outs]
                for k in range(5): accs[k]+=float(pr[k]==int(X[i,t,k]))
                cnt+=1
                items.append(tuple(pr)); pv=m.prev_vec(*[torch.tensor([v]) for v in pr])
                if t+1>=int(M[i].sum()): break
            rec=render(items); gt=(I[i,0].numpy()>0.5).astype(np.uint8)
            o1=cv2.dilate(gt,np.ones((3,3),np.uint8)); r1=cv2.dilate(rec,np.ones((3,3),np.uint8))
            A=o1>0;B=r1>0; ious.append((A&B).sum()/max((A|B).sum(),1))
            ya,xa=np.nonzero(gt>0); yb,xb=np.nonzero(rec>0)
            if len(xa) and len(xb):
                A2=np.stack([xa,ya],1).astype(np.float32); B2=np.stack([xb,yb],1).astype(np.float32)
                D=((A2[:,None]-B2[None])**2).sum(-1)
                chs.append(float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2)
            if collect and len(pairs)<6: pairs.append((gt,rec))
    m.train()
    return float(np.mean(ious)), (float(np.mean(chs)) if chs else 99.), pairs, accs/max(cnt,1)

best=0; START=1; CK=ROOT/"ckpt/prim5.pt"
if os.environ.get("RESUME")=="1" and CK.exists():
    st=torch.load(CK,map_location="cpu",weights_only=False)
    m.load_state_dict(st["model"]); START=st.get("step",1)+1; best=st.get("best",0); print("续训自",START)
for step in range(START, 12001):
    i=torch.randint(0,len(Itr),(48,))
    ss=min(0.5, step/4000*0.5)
    loss=run_batch(Xtr[i],Mtr[i],Itr[i],ss)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%200==0:
        iou,ch,_,acc=evaluate(Xva,Mva,Iva,100)
        print(f"step {step} loss {loss.item():.3f} ss {ss:.2f} | IoU {iou*100:.1f}% 倒角 {ch:.2f} | "
              f"ID {acc[0]*100:.0f}% 位置 {acc[1]*100:.0f}% 尺度 {acc[3]*100:.0f}% 旋转 {acc[4]*100:.0f}%", flush=True)
        if iou>best:
            best=iou; torch.save({"model":m.state_dict(),"cent":cent,"cfg":tr["cfg"],"step":step,"best":best}, CK)
print("最优 val IoU", round(best*100,1))
iou,ch,pairs,acc=evaluate(Xte,Mte,Ite,300,collect=True)
print(f"测试集 IoU {iou*100:.1f}% 倒角 {ch:.2f}px | 槽位准确率 ID {acc[0]*100:.0f}% 位置 {acc[1]*100:.0f}%")
rows=[]
for gt,rec in pairs:
    g=cv2.resize((255*(1-gt)).astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST)
    r=cv2.resize((255*(1-rec)).astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST)
    rows.append(np.hstack([g,np.full((128,6),150,np.uint8),r]))
cv2.imwrite(str(ROOT/"logs/turtle/prim_v5.png"), np.vstack(rows))
print("图:", ROOT/"logs/turtle/prim_v5.png")
