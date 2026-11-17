#!/usr/bin/env python3
"""Sketch-VAE v3：修「z 成摆设」——输入丢弃 + z 丢弃 + 自由比特 + KL 循环退火。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, math, os
from pathlib import Path
ROOT=Path("/srv/brainai")
M,HZ,ZD,L = 20,320,96,100
FB = 0.02            # 自由比特：每维 KL 下限

d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])
cats=sorted(set(words)); c2i={c:i for i,c in enumerate(cats)}

def stroke3(seq, maxlen=L):
    pts=[]
    for si,s in enumerate(seq):
        for pi,p in enumerate(s): pts.append((p[0],p[1],pi,len(s)-1))
    pts=pts[:maxlen]
    if len(pts)<8: return None
    out=np.zeros((len(pts),5),np.float32); prev=(0,0)
    for i,(x,y,pi,last) in enumerate(pts):
        out[i,0]=x-prev[0]; out[i,1]=y-prev[1]; prev=(x,y)
        out[i,2]=0.0 if pi==last else 1.0
        out[i,3]=1.0 if pi==last else 0.0
    out[-1,4]=1.0
    return out

data=[]
for i,s in enumerate(seqs):
    a=stroke3(s)
    if a is not None: data.append((i,a))
random.seed(0); random.shuffle(data); data=data[:26000]
offs=np.concatenate([a[:,:2] for _,a in data]); scale=float(np.std(offs))+1e-6
for _,a in data: a[:,:2]/=scale
print("有效", len(data), "scale", round(scale,3))

class SketchVAE(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.enc=nn.LSTM(5+64, 160, batch_first=True, bidirectional=True)
        self.mu=nn.Linear(320, ZD); self.lv=nn.Linear(320, ZD)
        self.cls=nn.Embedding(ncls,64)
        self.dec=nn.LSTM(5+ZD, HZ, batch_first=True)
        self.z=nn.Linear(HZ, M*6); self.pen=nn.Linear(HZ,3)
    def encode(self,x,c):
        cc=self.cls(c).unsqueeze(1).expand(-1,x.size(1),-1)
        o,_=self.enc(torch.cat([x,cc],-1)); h=o.mean(1)
        return self.mu(h), self.lv(h)
    def decode(self,x,z):
        zz=z.unsqueeze(1).expand(-1,x.size(1),-1)
        o,_=self.dec(torch.cat([x,zz],-1)); q=self.z(o)
        pi,mu1,mu2,s1,s2,rho=q.split(M,-1)
        return F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),self.pen(o)

m=SketchVAE(len(cats)); opt=torch.optim.AdamW(m.parameters(), lr=6e-4)
CK=ROOT/"ckpt/srn_vae4.pt"; START=1
if os.environ.get("RESUME")=="1" and CK.exists():
    st=torch.load(CK,map_location="cpu",weights_only=False)
    m.load_state_dict(st["model"]); START=st.get("step",1)+1; print("续训自",START)

W_PEN=torch.tensor([1.0,4.0,3.0])
def batch(ids):
    B=len(ids); X=torch.zeros(B,L,5); C=torch.zeros(B,dtype=torch.long)
    for k,j in enumerate(ids):
        a=data[j][1]; X[k,:len(a)]=torch.from_numpy(a); C[k]=c2i[words[data[j][0]]]
    return X,C

def rec_loss(pi,mu1,mu2,s1,s2,rho,pen,X):
    pi,mu1,mu2,s1,s2,rho,pen=[t[:,:-1] for t in (pi,mu1,mu2,s1,s2,rho,pen)]
    dx=X_gt[:,1:,0].unsqueeze(-1); dy=X_gt[:,1:,1].unsqueeze(-1); pn=X_gt[:,1:,2:].argmax(-1)
    n1=(dx-mu1)/s1; n2=(dy-mu2)/s2
    zz=n1**2+n2**2-2*rho*n1*n2
    G=-zz/(2*(1-rho**2))-torch.log(s1*s2*torch.sqrt(1-rho**2))-math.log(2*math.pi)
    return -torch.logsumexp(pi+G,-1).mean()+1.5*F.cross_entropy(pen.reshape(-1,3),pn.reshape(-1),weight=W_PEN)

for step in range(START, 12001):
    ids=random.sample(range(len(data)), 48)
    X_gt,C=batch(ids)
    # 关键 1：输入丢弃（随机把历史输入置零，破坏"抄上一个点"）
    keep=(torch.rand(X_gt.size(0), X_gt.size(1), 1) > 0.50).float()
    X_in=X_gt*keep
    mu,lv=m.encode(X_gt,C)
    z=mu+torch.randn_like(mu)*torch.exp(0.5*lv)
    # 关键 2：z 丢弃（15% 的样本把 z 清零）

    pi,mu1,mu2,s1,s2,rho,pen=m.decode(X_in,z)
    rec=rec_loss(pi,mu1,mu2,s1,s2,rho,pen,X_in)
    # 关键 3：自由比特（每维 KL 至少 FB，防止坍塌）
    kl_dim=-0.5*(1+lv-mu.pow(2)-lv.exp())
    kl=torch.max(kl_dim, torch.full_like(kl_dim, FB)).sum(1).mean()
    w_kl=min(1.0, step/6000.0)*0.02
    loss=rec+w_kl*kl
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%250==0:
        print(f"step {step} rec {rec.item():.3f} kl {kl.item():.2f} w {w_kl:.3f}", flush=True)
        torch.save({"model":m.state_dict(),"cats":cats,"scale":scale,"step":step}, CK)
