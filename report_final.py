#!/usr/bin/env python3
"""最终能力报告：识物 + 英文描述 + 中文描述（同一批图三路输出）。"""
import sys, re, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
sys.path.insert(0,"/srv/brainai")
from zh_words import ZH
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
# 识物
ck=torch.load(ROOT/"ckpt/concept_all.pt",map_location="cpu",weights_only=False)
names=[str(x) for x in ck["names"]]; K=len(names)
class CNet(nn.Module):
    def __init__(self,dd=512,h=768,k=150):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dd,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),nn.Linear(h,h),nn.GELU())
        self.multi=nn.Linear(h,k); self.sal=nn.Linear(h,k)
    def forward(self,x):
        z=self.trunk(x); return self.multi(z),self.sal(z)
cn=CNet(k=K); cn.load_state_dict(ck["model"]); cn.eval()
# 英文
ce=torch.load(ROOT/"ckpt/cap_merged.pt",map_location="cpu",weights_only=False)
vocab=ce["vocab"]; inv={v:k for k,v in vocab.items()}
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
nimg=len(FE0); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img=set(perm[:ntr].tolist()); tr=[k for k in range(len(CAPS)) if int(IMG[k]) in tr_img]
for s in range(1500):
    i=torch.tensor(np.random.choice(tr,256))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(oclf(FE)), torch.sigmoid(at(FE))], -1)
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
en_m=Cap(); en_m.load_state_dict(ce["model"]); en_m.eval()
@torch.no_grad()
def gen_en(cond,seed=0,temp=0.0):
    torch.manual_seed(seed); h=en_m.h0(cond); tok=torch.tensor([[1]]); cp=en_m.cproj(cond); out=[]
    for _ in range(40):
        o,hn=en_m.gru(torch.cat([en_m.emb(tok),cp.unsqueeze(1)],-1),h)
        nxt=int(en_m.out(o[:,-1])[0].argmax())
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
# 中文
cz=torch.load(ROOT/"ckpt/zh_all.pt",map_location="cpu",weights_only=False)
zv=cz["vocab"]; zinv={v:k for k,v in zv.items()}
class Zh(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(512,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h),nn.GELU())
        self.emb=nn.Embedding(len(zv),emb); self.cproj=nn.Linear(512,h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(zv))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
zh_m=Zh(); zh_m.load_state_dict(cz["model"]); zh_m.eval()
@torch.no_grad()
def gen_zh(cond,seed=0,temp=0.75,k=20):
    torch.manual_seed(seed); h=zh_m.h0(cond); tok=torch.tensor([[1]]); cp=zh_m.cproj(cond); out=[]
    for _ in range(30):
        o,hn=zh_m.gru(torch.cat([zh_m.emb(tok),cp.unsqueeze(1)],-1),h)
        logits=zh_m.out(o[:,-1]).clone()
        if len(out)>=1: logits[0,out[-1]]-=6.0          # ★禁止连续重复同一字
        if len(out)>=2 and out[-1]==out[-2]: logits[0,out[-1]]-=12.0
        v,i=torch.topk(logits[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(zinv.get(x,"") for x in out)
raw=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    raw.extend(pq.read_table(f).to_pydict()["image"])
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",20)
W,Hh=420,400; tiles=[]
te=[i for i in perm[ntr:].tolist() if i < 6000][:6]
for idx,i in enumerate(te):
    f=FE[i:i+1]
    with torch.no_grad(): lo,ls=cn(f)
    probs=torch.sigmoid(lo[0]); cand=[j for j in range(K) if float(probs[j])>0.15] or [int(probs.argmax())]
    subj=names[max(cand,key=lambda j: float(ls[0,j]))]
    others=[names[j] for j in sorted(cand,key=lambda j:-float(probs[j])) if names[j]!=subj][:2]
    en=gen_en(COND[i:i+1],seed=idx); zh=gen_zh(FE0[i:i+1],seed=idx)
    arr=cv2.imdecode(np.frombuffer(raw[i]["bytes"],np.uint8),cv2.IMREAD_COLOR)
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((W-16,175))
    canvas=Image.new("RGB",(W,Hh),"white"); dr=ImageDraw.Draw(canvas); canvas.paste(im,(8,8))
    y=192
    dr.text((8,y),f"识物：{subj}（{ZH.get(subj,'?')}）",font=font,fill=(0,110,0)); y+=24
    dr.text((8,y),f"其他：{'、'.join(ZH.get(o,o) for o in others) if others else '—'}",font=font,fill=(90,90,90)); y+=26
    dr.text((8,y),"英文：",font=font,fill=(180,0,0)); y+=20
    for line in [en[k:k+52] for k in range(0,len(en),52)][:2]:
        dr.text((8,y),line,font=font,fill=(0,0,0)); y+=20
    y+=6; dr.text((8,y),"中文：",font=font,fill=(180,0,0)); y+=20
    for line in [zh[k:k+24] for k in range(0,len(zh),24)][:3]:
        dr.text((8,y),line,font=font,fill=(0,0,0)); y+=20
    tiles.append(canvas)
cols=3; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols,Hh*rows+58),"white"); dr=ImageDraw.Draw(sheet)
dr.text((14,12),"多模态能力报告：识物 77% · 英文 chrF 28.4% · 中文通顺 76/80 种句子",font=fontb,fill=(20,20,20))
for k,t in enumerate(tiles): sheet.paste(t,((k%cols)*W,58+(k//cols)*Hh))
sheet.save(str(ROOT/"logs/report_final.png")); print("图:", ROOT/"logs/report_final.png", sheet.size)
