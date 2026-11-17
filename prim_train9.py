#!/usr/bin/env python3
"""积木重建 v9：128 画布 + 32×32 位置用「空间热力图」预测（不是 1024 分类）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, os
from pathlib import Path
ROOT=Path("/srv/brainai"); S=128
tr=np.load(ROOT/"data/prim_v5_train.npz",allow_pickle=True)
va=np.load(ROOT/"data/prim_v5_val.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v5_test.npz",allow_pickle=True)
cent=torch.from_numpy(tr["cent"]); N,G,RES,SB,RB,K=[int(x) for x in tr["cfg"]]
T=lambda d,k: torch.from_numpy(d[k])
Xtr,Mtr,Itr=T(tr,"slots"),T(tr,"mask").float(),T(tr,"imgs").float().unsqueeze(1)/255.
Xva,Mva,Iva=T(va,"slots"),T(va,"mask").float(),T(va,"imgs").float().unsqueeze(1)/255.
Xte,Mte,Ite=T(te,"slots"),T(te,"mask").float(),T(te,"imgs").float().unsqueeze(1)/255.
print("训练",len(Itr),"K",K,"网格",G)

class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.feat=nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.GELU(), nn.Conv2d(32,64,3,2,1), nn.GELU(),
            nn.Conv2d(64,96,3,2,1), nn.GELU(), nn.Conv2d(96,96,3,2,1), nn.GELU())     # 8×8
        self.up=nn.Sequential(nn.Upsample(scale_factor=2,mode="nearest"), nn.Conv2d(96,96,3,1,1), nn.GELU(),
                              nn.Upsample(scale_factor=2,mode="nearest"), nn.Conv2d(96,96,3,1,1), nn.GELU())  # 32×32
        self.sp=nn.Conv2d(96,h,1)                     # 位置用的空间特征
        self.gl=nn.Conv2d(96,h,1)                     # 全局特征
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.att=nn.MultiheadAttention(h,4,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(h,h*2),nn.GELU(),nn.Linear(h*2,h))
        self.h_id=nn.Linear(h,K); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
        self.q_pos=nn.Parameter(torch.randn(N,h)*0.02)   # 位置查询
    def forward(self,x):
        f=self.up(self.feat(x))                       # (B,96,32,32)
        sp=self.sp(f).flatten(2).transpose(1,2)       # (B,1024,h)
        gl=self.gl(f).mean(-1).mean(-1)               # (B,h)
        q=self.q.unsqueeze(0).expand(x.size(0),-1,-1)
        a,_=self.att(q, sp, sp); q=q+a+gl.unsqueeze(1); q=q+self.ff(q)
        pos=torch.einsum("bnh,bmh->bnm", self.q_pos.unsqueeze(0).expand(x.size(0),-1,-1), sp)
        return self.h_id(q), pos, self.h_res(q), self.h_sc(q), self.h_rot(q)

m=Net(); opt=torch.optim.AdamW(m.parameters(), lr=8e-4)
def loss_fn(outs,X,M):
    lid,pos,lres,lsc,lrot=outs
    ce_id=F.cross_entropy(lid.transpose(1,2), X[:,:,0], reduction='none')
    ce_pos=F.cross_entropy(pos.transpose(1,2), X[:,:,1], reduction="none")
    ce_res=F.cross_entropy(lres.transpose(1,2), X[:,:,2], reduction='none')
    ce_sc=F.cross_entropy(lsc.transpose(1,2), X[:,:,3], reduction='none')
    ce_rot=F.cross_entropy(lrot.transpose(1,2), X[:,:,4], reduction='none')
    tot=(ce_id+ce_pos+ce_res+ce_sc+ce_rot)
    return (tot*M).sum()/(M.sum()*5)
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
def evaluate(X,M,I,limit=120,collect=False):
    m.eval(); ious=[]; chs=[]; pairs=[]; acc=np.zeros(5); cnt=0
    with torch.no_grad():
        for i in range(min(limit,len(I))):
            outs=m(I[i:i+1]); pr=[o[0].argmax(-1) for o in outs]
            nsl=int(M[i].sum())
            for t in range(nsl):
                for k in range(5): acc[k]+=float(pr[k][t]==int(X[i,t,k]))
                cnt+=1
            rec=render([tuple(int(pr[k][t]) for k in range(5)) for t in range(nsl)])
            gt=(I[i,0].numpy()>0.5).astype(np.uint8)
            o1=cv2.dilate(gt,np.ones((3,3),np.uint8)); r1=cv2.dilate(rec,np.ones((3,3),np.uint8))
            A=o1>0;B=r1>0; ious.append((A&B).sum()/max((A|B).sum(),1))
            ya,xa=np.nonzero(gt>0); yb,xb=np.nonzero(rec>0)
            if len(xa) and len(xb):
                A2=np.stack([xa,ya],1).astype(np.float32); B2=np.stack([xb,yb],1).astype(np.float32)
                D=((A2[:,None]-B2[None])**2).sum(-1)
                chs.append(float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2)
            if collect and len(pairs)<6: pairs.append((gt,rec))
    m.train()
    return float(np.mean(ious)), (float(np.mean(chs)) if chs else 99.), pairs, acc/max(cnt,1)
best=0; START=1; CK=ROOT/"ckpt/prim9.pt"
if os.environ.get("RESUME")=="1" and CK.exists():
    st=torch.load(CK,map_location="cpu",weights_only=False); m.load_state_dict(st["model"]); START=st.get("step",1)+1; best=st.get("best",0)
for step in range(START, 6001):
    i=torch.randint(0,len(Itr),(32,))
    loss=loss_fn(m(Itr[i]),Xtr[i],Mtr[i])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%150==0:
        iou,ch,_,acc=evaluate(Xva,Mva,Iva,100)
        print(f"step {step} loss {loss.item():.3f} | IoU {iou*100:.1f}% 倒角 {ch:.2f} | ID {acc[0]*100:.0f}% 位置 {acc[1]*100:.0f}% 尺度 {acc[3]*100:.0f}% 旋转 {acc[4]*100:.0f}%", flush=True)
        if iou>best: best=iou; torch.save({"model":m.state_dict(),"cent":cent,"cfg":tr["cfg"],"step":step,"best":best}, CK)
print("最优 val IoU", round(best*100,1))
iou,ch,pairs,acc=evaluate(Xte,Mte,Ite,240,collect=True)
print(f"测试集 IoU {iou*100:.1f}% 倒角 {ch:.2f} | ID {acc[0]*100:.0f}% 位置 {acc[1]*100:.0f}%")
rows=[]
for gt,rec in pairs:
    g=cv2.resize((255*(1-gt)).astype(np.uint8),(160,160),interpolation=cv2.INTER_NEAREST)
    r=cv2.resize((255*(1-rec)).astype(np.uint8),(160,160),interpolation=cv2.INTER_NEAREST)
    rows.append(np.hstack([g,np.full((160,6),150,np.uint8),r]))
cv2.imwrite(str(ROOT/"logs/turtle/prim_v9.png"), np.vstack(rows))
print("图:", ROOT/"logs/turtle/prim_v9.png")
