#!/usr/bin/env python3
"""诊断：只学一个类别（cat），看这套流水线到底能不能画出东西。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, math, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); M,H,L=20,256,80
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])
idx=[i for i,w in enumerate(words) if w=="cat"][:3000]
print("猫样本", len(idx))

def stroke3(seq, maxlen=L):
    pts=[]
    for si,s in enumerate(seq):
        for pi,p in enumerate(s): pts.append((p[0],p[1],pi,len(s)-1))
    pts=pts[:maxlen]
    out=np.zeros((len(pts),5),np.float32); prev=(0,0)
    for i,(x,y,pi,last) in enumerate(pts):
        out[i,0]=x-prev[0]; out[i,1]=y-prev[1]; prev=(x,y)
        out[i,2]=0.0 if pi==last else 1.0
        out[i,3]=1.0 if pi==last else 0.0
    out[-1,4]=1.0
    return out

data=[stroke3(seqs[i]) for i in idx]
data=[a for a in data if len(a)>=5]
offs=np.concatenate([a[:,:2] for a in data]); scale=float(np.std(offs))+1e-6
for a in data: a[:,:2]/=scale
print("有效", len(data), "scale", round(scale,3))

class MDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(5,H,num_layers=1,batch_first=True)
        self.z=nn.Linear(H,M*6); self.pen=nn.Linear(H,3)
    def forward(self,x):
        h,_=self.lstm(x); z=self.z(h)
        pi,mu1,mu2,s1,s2,rho=z.split(M,-1)
        return F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),self.pen(h)
m=MDN(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)

def batch(ids):
    B=len(ids); X=torch.zeros(B,L,5)
    for k,j in enumerate(ids): X[k,:len(data[j])]=torch.from_numpy(data[j])
    return X

for step in range(1, 1501):
    ids=random.sample(range(len(data)),64)
    X=batch(ids)
    pi,mu1,mu2,s1,s2,rho,pen=m(X)
    pi,mu1,mu2,s1,s2,rho,pen=[t[:,:-1] for t in (pi,mu1,mu2,s1,s2,rho,pen)]
    dx=X[:,1:,0].unsqueeze(-1); dy=X[:,1:,1].unsqueeze(-1); pn=X[:,1:,2:].argmax(-1)
    n1=(dx-mu1)/s1; n2=(dy-mu2)/s2
    z=n1**2+n2**2-2*rho*n1*n2
    G=-z/(2*(1-rho**2))-torch.log(s1*s2*torch.sqrt(1-rho**2))-math.log(2*math.pi)
    loss=-torch.logsumexp(pi+G,-1).mean()+F.cross_entropy(pen.reshape(-1,3),pn.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%500==0: print(f"step {step} loss {loss.item():.3f}", flush=True)

def sample(steps=100,temp=0.4):
    canvas=np.zeros((64,64),np.float32); px=py=0.0; down=False
    hist=torch.zeros(1,1,5)
    with torch.no_grad():
        for _ in range(steps):
            pi,mu1,mu2,s1,s2,rho,pen = m(hist)
            pi,mu1,mu2,s1,s2,rho,pen = [t[:,-1:] for t in (pi,mu1,mu2,s1,s2,rho,pen)]
            pi=pi[0,0]; k=int(torch.multinomial(torch.exp(pi/temp),1))
            dx=float(mu1[0,0,k])+float(s1[0,0,k])*float(torch.randn(()))*temp
            dy=float(mu2[0,0,k])+float(s2[0,0,k])*float(torch.randn(()))*temp
            p=int(pen[0,0].argmax())
            x=torch.tensor([[[dx,dy,1.0 if p==0 else 0.,1.0 if p==1 else 0.,1.0 if p==2 else 0.]]])
            hist=torch.cat([hist,x],1)[:, -100:]
            nx,ny=px+dx*scale,py+dy*scale
            if down and p==0:
                cv2.line(canvas,(int((px+1)*32),int((py+1)*32)),(int((nx+1)*32),int((ny+1)*32)),1.0,1)
            down=(p==0); px,py=nx,ny
            if p==2: break
    return canvas

torch.save({"model":m.state_dict(),"scale":scale}, ROOT/"ckpt/srn_cat.pt")
tiles=[(255*(1-sample())).astype(np.uint8) for _ in range(8)]
OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(OUT/"cat_only.png"), np.hstack(tiles))
print("图:",OUT/"cat_only.png")
