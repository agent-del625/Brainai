#!/usr/bin/env python3
"""生成可视化报告：真实照片 + 模型生成的描述 + 参考描述。"""
import re, glob, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()
ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"];SKY=["clouds","sky","sunny","night","trees","sunset"];CNT=["one","two","three","group","crowd"]
ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
for s in range(1200):
    i=torch.randint(0,len(CAPS),(256,))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(clf(FE)), torch.sigmoid(at(FE))], -1)
cc=torch.load(ROOT/"ckpt/cap10.pt",map_location="cpu",weights_only=False)
vocab=cc["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
m=Cap(); m.load_state_dict(cc["model"]); m.eval()
@torch.no_grad()
def gen(cond,temp=0.7,k=20,steps=40,seed=0):
    torch.manual_seed(seed); h=m.h0(cond); tok=torch.tensor([[1]]); cp=m.cproj(cond); out=[]
    for _ in range(steps):
        o,hn=m.gru(torch.cat([m.emb(tok),cp.unsqueeze(1)],-1),h)
        lg=m.out(o[:,-1])[0]; v,i=torch.topk(lg,k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
n=len(FE); perm=torch.randperm(n); te=perm[int(n*0.9):].tolist()[:8]

# 取原图
raw=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    t=pq.read_table(f).to_pydict(); raw.extend(t["image"])
W,Hh=360,420
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",22)
tiles=[]
for idx,i in enumerate(te):
    b=raw[i]["bytes"]; arr=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((W-20,250))
    cap=gen(COND[i:i+1],seed=idx); ref=CAPS[int(np.where(IMG.numpy()==i)[0][0])]
    canvas=Image.new("RGB",(W,Hh),"white"); canvas.paste(im,(10,10))
    dr=ImageDraw.Draw(canvas)
    y=270
    dr.text((10,y),"模型生成：",font=font,fill=(200,0,0)); y+=24
    words=cap.split(); line=""
    for w_ in words:
        if len(line)+len(w_)>40: dr.text((10,y),line,font=font,fill=(0,0,0)); y+=22; line=w_+" "
        else: line+=w_+" "
    if line: dr.text((10,y),line,font=font,fill=(0,0,0)); y+=26
    dr.text((10,y),"参考："+ref[:40],font=font,fill=(110,110,110))
    tiles.append(canvas)
cols=4; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols, Hh*rows+70),"white")
dr=ImageDraw.Draw(sheet)
dr.text((16,14),"看图说英文 —— 修复后（chrF 26.4%，每张图生成不同句子）",font=fontb,fill=(20,20,20))
for k,t in enumerate(tiles):
    sheet.paste(t,((k%cols)*W, 70+(k//cols)*Hh))
sheet.save(str(ROOT/"logs/cap_report.png"))
print("图:", ROOT/"logs/cap_report.png", sheet.size)
