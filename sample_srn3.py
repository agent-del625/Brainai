#!/usr/bin/env python3
"""Sketch-RNN v2 采样（stroke-3 版，携带隐状态）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); M,H,S=20,384,64
class MDN(nn.Module):
    def __init__(self,ncls):
        super().__init__()
        self.lstm=nn.LSTM(5+64,H,num_layers=2,batch_first=True)
        self.cls=nn.Embedding(ncls,64)
        self.z=nn.Linear(H,M*6); self.pen=nn.Linear(H,3)
    def forward(self,x,c,h=None):
        cc=self.cls(c).unsqueeze(1).expand(-1,x.size(1),-1)
        out,h=self.lstm(torch.cat([x,cc],-1),h)
        z=self.z(out); pen=self.pen(out)
        pi,mu1,mu2,s1,s2,rho=z.split(M,-1)
        return (F.log_softmax(pi,-1),mu1,mu2,F.softplus(s1)+1e-3,F.softplus(s2)+1e-3,torch.tanh(rho),pen,h)
ck=torch.load(ROOT/"ckpt/srn2.pt",map_location="cpu",weights_only=False)
cats=[str(c) for c in ck["cats"]]; scale=ck.get("scale",1.0)
m=MDN(len(cats)); m.load_state_dict(ck["model"]); m.eval()

def sample(cid, steps=120, temp=0.6):
    canvas=np.zeros((S,S),np.float32); h=None; x=torch.zeros(1,1,5)
    px=py=0.0; down=False
    with torch.no_grad():
        for t in range(steps):
            pi,mu1,mu2,s1,s2,rho,pen,h = m(x, torch.tensor([cid]), h)
            pi=pi[0,0]; k=int(torch.multinomial(torch.exp(pi/temp),1))
            dx=float(mu1[0,0,k])+float(s1[0,0,k])*float(torch.randn(()))*temp
            dy=float(mu2[0,0,k])+float(s2[0,0,k])*float(torch.randn(()))*temp
            p=int(pen[0,0].argmax())
            x=torch.tensor([[[dx,dy,1.0 if p==0 else 0.0,1.0 if p==1 else 0.0,1.0 if p==2 else 0.0]]])
            nx,ny=px+dx*scale, py+dy*scale
            if down and p==0:
                cv2.line(canvas,(int((px+1)*S/2),int((py+1)*S/2)),
                                (int((nx+1)*S/2),int((ny+1)*S/2)),1.0,1)
            down=(p==0); px,py=nx,ny
            if p==2: break
    return canvas

OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True,exist_ok=True)
tiles=[]
for c in cats:
    best=None
    for _ in range(4):
        im=sample(cats.index(c),120,0.5)
        if best is None or im.sum()>best.sum(): best=im
    tiles.append((255*(1-best)).astype(np.uint8))
    cv2.imwrite(str(OUT/f"srn3_{c}.png"), tiles[-1])
cv2.imwrite(str(OUT/"srn3.png"), np.hstack(tiles))
print("图:",OUT/"srn3.png")
