#!/usr/bin/env python3
"""渲染位置感知模型的生成结果，与原图对比。"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
MOVE=1024;PEN=1024;EOS=1025;BOS=1026;PAD=1027;V=1028;L=72;BIN=32;E=0.35
dd,h,layers=384,8,6
class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.h=h; self.dh=d//h
        self.qkv=nn.Linear(d,d*3); self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x,mask):
        B,T,D=x.shape
        q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        q=q.view(B,T,self.h,self.dh).transpose(1,2); k=k.view(B,T,self.h,self.dh).transpose(1,2); v=v.view(B,T,self.h,self.dh).transpose(1,2)
        att=((q@k.transpose(-1,-2))/math.sqrt(self.dh)+mask).softmax(-1)
        o=(att@v).transpose(1,2).reshape(B,T,D)
        x=x+self.proj(o); return x+self.ff(self.ln2(x))
class DrawPos(nn.Module):
    def __init__(self,NC=11):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                               nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.Conv2d(128,256,3,2,1),nn.GELU())
        self.spatial=nn.Linear(256,dd); self.cls=nn.Embedding(NC,dd)
        self.tok=nn.Embedding(V,dd)
        self.posxy=nn.Sequential(nn.Linear(2,dd),nn.GELU(),nn.Linear(dd,dd))
        self.pos=nn.Embedding(256,dd)
        self.blocks=nn.ModuleList([Block(dd,h) for _ in range(layers)])
        self.ln=nn.LayerNorm(dd); self.head=nn.Linear(dd,V,bias=False); self.head.weight=self.tok.weight
        self.cls_head=nn.Linear(dd,NC)
ck=torch.load("/tmp/snap_pos.pt",map_location="cpu")
m=DrawPos(); m.load_state_dict(ck["model"]); m.eval()
print("模型步数",ck.get("step"))
d=np.load(ROOT/"data/stroke_ds.npz",allow_pickle=True)
I=torch.from_numpy(d["imgs"]).float().unsqueeze(1)/255.0
LAB=torch.from_numpy(d["labels"]).long()
CLS=["bird","car","cat","dog","fish","flower","house","star","sun","train","tree"]
@torch.no_grad()
def gen(img,cl,seed=0,temp=0.85):
    torch.manual_seed(seed)
    f=m.spatial(m.cnn(img).flatten(2).transpose(1,2))+m.cls(cl).unsqueeze(1)
    P=f.size(1); ids=[BOS]; x=y=0.5; out=[]
    for _ in range(L):
        xin=torch.tensor([ids])
        pxy=torch.tensor([[[x,y]]],dtype=torch.float32).expand(1,len(ids),2)
        hh=torch.cat([f,m.tok(xin)+m.posxy(pxy)],1)+m.pos(torch.arange(P+xin.size(1))[None])
        Tt=P+xin.size(1); causal=torch.triu(torch.full((Tt,Tt),float("-inf")),1)[None,None]
        for b in m.blocks: hh=b(hh,causal)
        lp=m.head(m.ln(hh))[:,-1][0]
        if out: lp[out[-1]]-=1.5
        v,k=torch.topk(lp,20); nxt=int(k[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==EOS: break
        ids.append(nxt); out.append(nxt)
        if nxt<PEN:
            qx,qy=divmod(nxt,BIN); x+=(qx/(BIN-1))*2*E-E; y+=(qy/(BIN-1))*2*E-E
    return out
def decode(toks,S=64):
    pts=[];x=y=0.5;cur=[(x,y)]
    for t in toks:
        if t>=PEN:
            if cur: pts.append(cur); cur=[(x,y)]
            continue
        qx,qy=divmod(int(t),BIN); x+=(qx/(BIN-1))*2*E-E; y+=(qy/(BIN-1))*2*E-E
        cur.append((x,y))
    if cur: pts.append(cur)
    img=np.zeros((S,S),np.uint8)
    for s in pts:
        for i in range(len(s)-1):
            p1=(int(np.clip(s[i][0],0,1)*(S-1)),int(np.clip(s[i][1],0,1)*(S-1)))
            p2=(int(np.clip(s[i+1][0],0,1)*(S-1)),int(np.clip(s[i+1][1],0,1)*(S-1)))
            cv2.line(img,p1,p2,255,1)
    return img
def chamfer(a,b):
    ya,xa=np.nonzero(a>50); yb,xb=np.nonzero(b>50)
    if len(xa)==0 or len(xb)==0: return 99.0
    A=np.stack([xa,ya],1).astype(np.float32); B=np.stack([xb,yb],1).astype(np.float32)
    Dm=((A[:,None,:]-B[None,:,:])**2).sum(-1)
    return float(np.sqrt(Dm.min(1)).mean()+np.sqrt(Dm.min(0)).mean())
W,H=150,308
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",13)
sel=list(range(0,27500,2200))[:12]
tiles=[];cds=[]
for j,i in enumerate(sel):
    toks=gen(I[i:i+1],LAB[i:i+1],seed=j); g=decode(toks)
    cd=chamfer(g,d["imgs"][i]); cds.append(cd)
    tiles.append((d["imgs"][i],g,cd,CLS[int(LAB[i])],len(toks)))
sheet=Image.new("RGB",(W*4,H*3+46),"white"); dr=ImageDraw.Draw(sheet)
dr.text((10,10),f"上=原图  中=模型生成（位置感知版 {ck.get('step')} 步）",font=font,fill=(0,0,0))
for k,(orig,g,cd,cl,ln) in enumerate(tiles):
    cx=(k%4)*W; cy=46+(k//4)*H
    sheet.paste(Image.fromarray(orig).resize((128,128)).convert("RGB"),(cx+11,cy))
    sheet.paste(Image.fromarray(g).resize((128,128)).convert("RGB"),(cx+11,cy+134))
    dr.text((cx+11,cy+268),f"{cl} cd={cd:.0f}",font=font,fill=(180,0,0))
sheet.save(ROOT/"logs/draw_pos.png")
print("chamfer 平均 %.1f px"%np.mean(cds))
