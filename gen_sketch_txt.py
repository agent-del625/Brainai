#!/usr/bin/env python3
"""本地用快照生成「词→简笔画」并渲染（不依赖远端）。"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai")
MOVE=1024;PEN=1024;EOS=1025;BOS=1026;PAD=1027;V=1028;L=72;BIN=32;E=0.35
dd,h,layers,P=384,8,6,2
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
class Sketch(nn.Module):
    def __init__(self):
        super().__init__(); self.P=P
        self.cp=nn.Sequential(nn.Linear(384,dd),nn.GELU(),nn.Linear(dd,dd*P))
        self.tok=nn.Embedding(V,dd); self.pos=nn.Embedding(P+L+2,dd)
        self.blocks=nn.ModuleList([Block(dd,h) for _ in range(layers)])
        self.ln=nn.LayerNorm(dd); self.head=nn.Linear(dd,V,bias=False); self.head.weight=self.tok.weight
ck=torch.load("/tmp/snap_sk.pt",map_location="cpu",weights_only=False)
print("模型步数",ck.get("step"))
m=Sketch(); m.load_state_dict(ck["model"]); m.eval()
CE=np.load(ROOT/"data/concept_emb.npz",allow_pickle=True)
QD=[str(w) for w in CE["qd_words"]]; QE=torch.from_numpy(CE["qd_emb"]).float()
@torch.no_grad()
def gen(c,seed,temp=0.85):
    torch.manual_seed(seed)
    pre=m.cp(c).view(1,P,-1); ids=[BOS]; out=[]
    for _ in range(L):
        x=torch.tensor([ids])
        hh=torch.cat([pre,m.tok(x)],1)+m.pos(torch.arange(P+x.size(1))[None])
        Tt=P+x.size(1); causal=torch.triu(torch.full((Tt,Tt),float("-inf")),1)[None,None]
        for b in m.blocks: hh=b(hh,causal)
        lp=m.head(m.ln(hh))[:,-1][0]
        if out: lp[out[-1]]-=1.5
        v,k=torch.topk(lp,20); nxt=int(k[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==EOS: break
        ids.append(nxt); out.append(nxt)
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
from PIL import Image, ImageDraw, ImageFont
W,H=132,170
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",15)
sheet=Image.new("RGB",(W*3, H*11+50),"white"); dr=ImageDraw.Draw(sheet)
dr.text((8,10),f"词 → 简笔画（{ck.get('step')} 步）  每行一个词，3 种画法",font=font,fill=(0,0,0))
for wi,w in enumerate(QD):
    for k in range(3):
        img=decode(gen(QE[wi:wi+1],wi*10+k))
        sheet.paste(Image.fromarray(img).resize((112,112)).convert("RGB"),(k*W+8, 44+wi*H))
    dr.text((8,44+wi*H+114),w,font=font,fill=(180,0,0))
sheet.save(ROOT/"logs/sketch_txt.png")
print("图:",ROOT/"logs/sketch_txt.png",sheet.size)
