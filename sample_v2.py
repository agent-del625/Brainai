#!/usr/bin/env python3
"""Sketch-VAE v2 采样网格（每类 6 张）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); M,HZ,ZD=20,384,96; S=96
class SketchVAE(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.enc=nn.LSTM(5+64,160,batch_first=True,bidirectional=True)
        self.mu=nn.Linear(320,ZD); self.lv=nn.Linear(320,ZD)
        self.cls=nn.Embedding(ncls,64)
        self.dec=nn.LSTM(5+ZD,HZ,num_layers=2,batch_first=True,dropout=0.05)
        self.z=nn.Linear(HZ,M*6); self.pen=nn.Linear(HZ,3)
    def decode_step(self,x,z):
        o,_=self.dec(torch.cat([x,z],-1)); q=self.z(o)
        pi,mu1,mu2,s1,s2,rho=q.split(M,-1)
        return F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),self.pen(o)
ck=torch.load(ROOT/"ckpt/srn_vae2.pt",map_location="cpu",weights_only=False)
cats=[str(c) for c in ck["cats"]]; scale=ck["scale"]
m=SketchVAE(len(cats)); m.load_state_dict(ck["model"]); m.eval()
print("checkpoint step", ck.get("step"))

def sample(cid, steps=120, temp=0.45, min_steps=70, zs=0.8):
    z=torch.randn(1,1,ZD)*zs
    x=torch.zeros(1,1,5); px=py=0.; down=False; canvas=np.zeros((S,S),np.float32)
    with torch.no_grad():
        for t in range(steps):
            pi,mu1,mu2,s1,s2,rho,pen=m.decode_step(x,z)
            pi=pi[0,0].clone()
            if t<min_steps: pi[...,2]=-1e9
            k=int(torch.multinomial(torch.exp(pi/temp),1))
            dx=float(mu1[0,0,k])+float(s1[0,0,k])*float(torch.randn(()))*temp
            dy=float(mu2[0,0,k])+float(s2[0,0,k])*float(torch.randn(()))*temp
            p=int(pen[0,0].argmax())
            x=torch.tensor([[[dx,dy,1. if p==0 else 0.,1. if p==1 else 0.,1. if p==2 else 0.]]])
            nx,ny=px+dx*scale,py+dy*scale
            if down and p==0:
                cv2.line(canvas,(int((px+1)*S/2),int((py+1)*S/2)),(int((nx+1)*S/2),int((ny+1)*S/2)),1.,2)
            down=(p==0); px,py=nx,ny
            if p==2: break
    return canvas
rows=[]
for c in cats:
    tiles=[(255*(1-sample(cats.index(c)))).astype(np.uint8) for _ in range(6)]
    rows.append(np.hstack(tiles))
OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True,exist_ok=True)
cv2.imwrite(str(OUT/"grid_v2.png"), np.vstack(rows))
print("图:",OUT/"grid_v2.png")
