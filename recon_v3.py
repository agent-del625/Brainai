#!/usr/bin/env python3
"""v3 重建 + 采样对照（三列：真实 / 重建 / 采样）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); M,HZ,ZD=20,320,96; S=96
class SketchVAE(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.enc=nn.LSTM(5+64,160,batch_first=True,bidirectional=True)
        self.mu=nn.Linear(320,ZD); self.lv=nn.Linear(320,ZD)
        self.cls=nn.Embedding(ncls,64)
        self.dec=nn.LSTM(5+ZD,HZ,batch_first=True)
        self.z=nn.Linear(HZ,M*6); self.pen=nn.Linear(HZ,3)
    def encode(self,x,c):
        cc=self.cls(c).unsqueeze(1).expand(-1,x.size(1),-1)
        o,_=self.enc(torch.cat([x,cc],-1)); h=o.mean(1)
        return self.mu(h), self.lv(h)
    def decode_step(self,x,z):
        o,_=self.dec(torch.cat([x,z],-1)); q=self.z(o)
        pi,mu1,mu2,s1,s2,rho=q.split(M,-1)
        return F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),self.pen(o)
ck=torch.load(ROOT/"ckpt/srn_vae4.pt",map_location="cpu",weights_only=False)
cats=[str(c) for c in ck["cats"]]; scale=ck["scale"]
m=SketchVAE(len(cats)); m.load_state_dict(ck["model"]); m.eval()
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True); seqs,words=list(d["seqs"]),list(d["words"])
def render_seq(seq,size=S):
    c=np.zeros((size,size),np.float32)
    for s in seq:
        pts=[((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts,pts[1:]): cv2.line(c,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.,1)
    return c
def s3(seq,maxlen=100):
    pts=[]
    for si,s in enumerate(seq):
        for pi,p in enumerate(s): pts.append((p[0],p[1],pi,len(s)-1))
    pts=pts[:maxlen]; n=len(pts); out=np.zeros((n,5),np.float32); prev=(0,0)
    for i,(x,y,pi,last) in enumerate(pts):
        out[i,0]=x-prev[0]; out[i,1]=y-prev[1]; prev=(x,y)
        out[i,2]=0. if pi==last else 1.; out[i,3]=1. if pi==last else 0.
    out[-1,4]=1.
    return out
def decode(z,cid,steps=100,temp=0.0,minst=55):
    x=torch.zeros(1,1,5); px=py=0.; down=False; c=np.zeros((S,S),np.float32)
    with torch.no_grad():
        for t in range(steps):
            pi,mu1,mu2,s1,s2,rho,pen=m.decode_step(x,z.view(1,1,-1))
            pi=pi[0,0].clone()
            if t<minst: pi[...,2]=-1e9
            k=int(torch.multinomial(torch.exp(pi/0.45),1)) if temp>0 else int(pi.argmax())
            dx=float(mu1[0,0,k]); dy=float(mu2[0,0,k]); p=int(pen[0,0].argmax())
            x=torch.tensor([[[dx,dy,1. if p==0 else 0.,1. if p==1 else 0.,1. if p==2 else 0.]]])
            nx,ny=px+dx*scale,py+dy*scale
            if down and p==0: cv2.line(c,(int((px+1)*S/2),int((py+1)*S/2)),(int((nx+1)*S/2),int((ny+1)*S/2)),1.,2)
            down=(p==0); px,py=nx,ny
            if p==2: break
    return c
rows=[]
for w in ["cat","dog","fish","train"]:
    i=[j for j,x in enumerate(words) if x==w][0]
    a=s3(seqs[i]).copy(); a[:,:2]/=scale
    with torch.no_grad(): mu,_=m.encode(torch.from_numpy(a)[None], torch.tensor([cats.index(w)]))
    cols=[(255*(1-render_seq(seqs[i]))).astype(np.uint8),
          (255*(1-decode(mu,cats.index(w)))).astype(np.uint8),
          (255*(1-decode(torch.randn(1,ZD)*0.8,cats.index(w),temp=0.45))).astype(np.uint8)]
    rows.append(np.hstack([np.hstack([c,np.full((S,4),200,np.uint8)]) for c in cols]))
OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(OUT/"v3_triple.png"), np.vstack(rows))
print("step", ck.get("step"), "→", OUT/"v3_triple.png")
