#!/usr/bin/env python3
"""Sketch-RNN v2：按论文标准做 stroke-3（dx,dy 归一化到统一尺度 + 三段 pen 状态）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, math
from pathlib import Path
ROOT=Path("/srv/brainai"); M,H=20,384
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])
cats=sorted(set(words)); c2i={c:i for i,c in enumerate(cats)}

def to_stroke3(seq, maxlen=120):
    """返回 (dx,dy,p1,p2,p3) 序列；每个笔画最后一点是 pen-up。"""
    pts=[]
    for si,s in enumerate(seq):
        for pi,p in enumerate(s):
            pts.append((p[0],p[1],si,pi,len(s)))
    if len(pts)<5: return None
    pts=pts[:maxlen]
    out=np.zeros((len(pts),5),np.float32)
    prev=(0,0)
    for i,(x,y,si,pi,len_s) in enumerate(pts):
        out[i,0]=x-prev[0]; out[i,1]=y-prev[1]; prev=(x,y)
        is_stroke_end = (pi==len_s-1)
        out[i,2]=0.0 if is_stroke_end else 1.0     # pen down (继续画)
        out[i,3]=1.0 if is_stroke_end else 0.0     # pen up
        out[i,4]=0.0
    out[-1,4]=1.0
    return out

data=[]
for i,s in enumerate(seqs):
    a=to_stroke3(s)
    if a is not None: data.append((i,a))
# 统一尺度：所有 offset 除以全局标准差
alloff=np.concatenate([a[:,:2] for _,a in data])
scale=float(np.std(alloff))+1e-6
for _,a in data: a[:,:2]/=scale
print("样本",len(data),"offset 尺度",round(scale,2))

class MDN(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.lstm=nn.LSTM(5+64,H,num_layers=2,batch_first=True)
        self.cls=nn.Embedding(ncls,64)
        self.z=nn.Linear(H,M*6); self.pen=nn.Linear(H,3)
    def forward(self,x,c):
        cc=self.cls(c).unsqueeze(1).expand(-1,x.size(1),-1)
        h,_=self.lstm(torch.cat([x,cc],-1))
        z=self.z(h); pen=self.pen(h)
        pi,mu1,mu2,s1,s2,rho=z.split(M,-1)
        return (F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),pen)

m=MDN(len(cats)); opt=torch.optim.AdamW(m.parameters(), lr=8e-4)
L=120
def batch(ids):
    B=len(ids); X=torch.zeros(B,L,5); C=torch.zeros(B,dtype=torch.long)
    for k,j in enumerate(ids):
        a=data[j][1]; X[k,:len(a)]=torch.from_numpy(a); C[k]=c2i[words[data[j][0]]]
    return X,C

def loss_fn(pi,mu1,mu2,s1,s2,rho,pen,X):
    pi,mu1,mu2,s1,s2,rho,pen=[t[:,:-1] for t in (pi,mu1,mu2,s1,s2,rho,pen)]
    dx=X[:,1:,0].unsqueeze(-1); dy=X[:,1:,1].unsqueeze(-1); pn=X[:,1:,2:].argmax(-1)
    n1=(dx-mu1)/s1; n2=(dy-mu2)/s2
    z=n1**2+n2**2-2*rho*n1*n2
    G=-z/(2*(1-rho**2))-torch.log(s1*s2*torch.sqrt(1-rho**2))-math.log(2*math.pi)
    l1=-torch.logsumexp(pi+G,-1).mean()
    l2=F.cross_entropy(pen.reshape(-1,3), pn.reshape(-1))
    return l1+l2, l1, l2

for step in range(1, 6001):
    ids=random.sample(range(len(data)), 32)
    X,C=batch(ids)
    pi,mu1,mu2,s1,s2,rho,pen=m(X,C)
    loss,l1,l2=loss_fn(pi,mu1,mu2,s1,s2,rho,pen,X)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%250==0:
        print(f"step {step} loss {loss.item():.3f} (mdn {l1.item():.3f} pen {l2.item():.3f})", flush=True)
        torch.save({"model":m.state_dict(),"cats":cats,"scale":scale}, ROOT/"ckpt/srn2.pt")
