#!/usr/bin/env python3
"""v3 演示：显著性识别（150 类，top-1 67%）+ 英文描述 + 中英对照。"""
import re, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
ck=torch.load(ROOT/"ckpt/concept150c.pt",map_location="cpu",weights_only=False)
names=[str(x) for x in ck["names"]]; K=len(names)
class Net(nn.Module):
    def __init__(self,dd=512,h=768,k=150):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dd,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),nn.Linear(h,h),nn.GELU())
        self.multi=nn.Linear(h,k); self.sal=nn.Linear(h,k)
    def forward(self,x):
        z=self.trunk(x); return self.multi(z),self.sal(z)
cn=Net(k=K); cn.load_state_dict(ck["model"]); cn.eval()
old=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
on=[str(c) for c in old["names"]]
class OClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(on)))
    def forward(self,x): return self.net(x)
oclf=OClf(); oclf.load_state_dict(old["model"]); oclf.eval()
ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"];SKY=["clouds","sky","sunny","night","trees","sunset"];CNT=["one","two","three","group","crowd"]
ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
# ★属性头只用训练图像（修泄漏）
perm=torch.randperm(len(FE0)); ntr=int(len(FE0)*0.9); tr_img=set(perm[:ntr].tolist())
tr=[k for k in range(len(CAPS)) if int(IMG[k]) in tr_img]
for s in range(1500):
    i=torch.tensor(np.random.choice(tr,256))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(oclf(FE)), torch.sigmoid(at(FE))], -1)
ck2=torch.load(ROOT/"ckpt/cap10.pt",map_location="cpu",weights_only=False)
vocab=ck2["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
cap=Cap(); cap.load_state_dict(ck2["model"]); cap.eval()
@torch.no_grad()
def gen_caption(cond,seed=0,temp=0.7,k=20):
    torch.manual_seed(seed); h=cap.h0(cond); tok=torch.tensor([[1]]); cp=cap.cproj(cond); out=[]
    for _ in range(40):
        o,hn=cap.gru(torch.cat([cap.emb(tok),cp.unsqueeze(1)],-1),h)
        v,i=torch.topk(cap.out(o[:,-1])[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
raw=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    raw.extend(pq.read_table(f).to_pydict()["image"])
n=len(FE0); te=perm[ntr:].tolist()[:6]
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",20)
W,Hh=400,330; tiles=[]
for idx,i in enumerate(te):
    f=FE[i:i+1]
    with torch.no_grad(): lo,ls=cn(f)
    probs=torch.sigmoid(lo[0]); order=ls[0].argsort(descending=True).tolist()
    subject=names[order[0]]
    # 组合规则：显著性排序里第一个被识别为存在的
    subject=next((names[j] for j in order if probs[j]>0.25), subject)
    others=[names[j] for j in probs.argsort(descending=True).tolist()[:5] if j!=names.index(subject) and probs[j]>0.2][:3]
    en=gen_caption(COND[i:i+1],seed=idx)
    ref=CAPS[int(np.where(IMG.numpy()==i)[0][0])]
    arr=cv2.imdecode(np.frombuffer(raw[i]["bytes"],np.uint8),cv2.IMREAD_COLOR)
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((W-16,150))
    canvas=Image.new("RGB",(W,Hh),"white"); dr=ImageDraw.Draw(canvas); canvas.paste(im,(8,8))
    y=166
    dr.text((8,y),f"主体：{subject}",font=font,fill=(0,110,0)); y+=24
    dr.text((8,y),f"其他：{'、'.join(others) if others else '—'}",font=font,fill=(80,80,80)); y+=26
    dr.text((8,y),"模型描述：",font=font,fill=(180,0,0)); y+=20
    for line in [en[k:k+48] for k in range(0,len(en),48)][:2]:
        dr.text((8,y),line,font=font,fill=(0,0,0)); y+=20
    y+=4; dr.text((8,y),"参考答案：",font=font,fill=(150,150,150)); y+=20
    for line in [ref[k:k+48] for k in range(0,len(ref),48)][:2]:
        dr.text((8,y),line,font=font,fill=(110,110,110)); y+=20
    tiles.append(canvas)
cols=3; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols,Hh*rows+56),"white"); dr=ImageDraw.Draw(sheet)
dr.text((14,12),"识物 v3：显著性排序（top-1 67%）+ 英文描述（属性头已去泄漏）",font=fontb,fill=(20,20,20))
for k,t in enumerate(tiles): sheet.paste(t,((k%cols)*W,56+(k//cols)*Hh))
sheet.save(str(ROOT/"logs/demo60/v3.png")); print("图:", ROOT/"logs/demo60/v3.png", sheet.size)
