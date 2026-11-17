#!/usr/bin/env python3
"""60 概念版演示：照片 → 概念（60 类）+ 英文描述 + 中文 + 简笔画(若可画) + 声音(若有)。"""
import re, glob, wave, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from audio_feat import mel_to_wav
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
c60=torch.load(ROOT/"ckpt/concept60.pt",map_location="cpu",weights_only=False)
names=[str(x) for x in c60["names"]]; TABLE=c60["table"]
CN={t[0]:t[1] for t in TABLE}; DRAW={t[0]:t[2] for t in TABLE}; SND={t[0]:t[3] for t in TABLE}
class Clf(nn.Module):
    def __init__(self,d=512,h=512,k=60):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.1),nn.Linear(h,h),nn.GELU(),nn.Linear(h,k))
    def forward(self,x): return self.net(x)
clf=Clf(k=len(names)); clf.load_state_dict(c60["model"]); clf.eval()
# 描述模型沿用 cap10 的条件（需要 23 类 + 属性头，这里用旧 clf 保持一致性）
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
for s in range(1200):
    i=torch.randint(0,len(CAPS),(256,))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(oclf(FE)), torch.sigmoid(at(FE))], -1)
ck=torch.load(ROOT/"ckpt/cap10.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
cap=Cap(); cap.load_state_dict(ck["model"]); cap.eval()
@torch.no_grad()
def gen_caption(cond,seed=0,temp=0.7,k=20,steps=40):
    torch.manual_seed(seed); h=cap.h0(cond); tok=torch.tensor([[1]]); cp=cap.cproj(cond); out=[]
    for _ in range(steps):
        o,hn=cap.gru(torch.cat([cap.emb(tok),cp.unsqueeze(1)],-1),h)
        v,i=torch.topk(cap.out(o[:,-1])[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
pk=torch.load(ROOT/"ckpt/p2s.pt",map_location="cpu",weights_only=False)
cent=torch.as_tensor(pk["cent"]).float(); N,G,RES,SB,RB,K=[int(x) for x in pk["cfg"]]
class P2S(nn.Module):
    def __init__(self,h=384):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h),nn.GELU())
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def forward(self,f):
        h=self.net(f).unsqueeze(1)+self.q.unsqueeze(0)
        return self.h_id(h),self.h_cell(h),self.h_res(h),self.h_sc(h),self.h_rot(h)
p2s=P2S(); p2s.load_state_dict(pk["model"]); p2s.eval()
def draw_sketch(f,seed=0,size=170):
    torch.manual_seed(seed)
    with torch.no_grad(): outs=p2s(f); probs=[torch.softmax(o[0],-1) for o in outs]
    img=np.zeros((size,size),np.uint8)
    for t in range(N):
        pid,cell,res,sb,rb=[int(torch.multinomial(probs[j][t],1)) for j in range(5)]
        gx,gy=divmod(cell,G); rx,ry=divmod(res,RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th); sh=cent[pid].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*size/2-0.5,(sh[:,1]+1)*size/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),255,2)
    return 255-img
ep=np.load(ROOT/"data/esc50_proto.npz",allow_pickle=True); ec=[str(c) for c in ep["cats"]]; em=ep["mels"]
raw=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    raw.extend(pq.read_table(f).to_pydict()["image"])
n=len(FE); perm=torch.randperm(n); te=perm[int(n*0.9):].tolist()[:6]
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",20)
W,Hh=400,430; tiles=[]; OUT=ROOT/"logs/demo60"; OUT.mkdir(parents=True,exist_ok=True)
for idx,i in enumerate(te):
    f=FE0[i:i+1]
    with torch.no_grad(): p60=torch.sigmoid(clf(f))[0]
    top=p60.argsort(descending=True).tolist()[:3]
    concept=names[top[0]]; cn=CN.get(concept,concept)
    en=gen_caption(COND[i:i+1],seed=idx)
    can_draw=DRAW.get(concept) is not None
    snd=SND.get(concept)
    if snd and snd in ec:
        wav=mel_to_wav(em[ec.index(snd)][0])
        with wave.open(str(OUT/f"c{idx}_{concept}.wav"),"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(wav.tobytes())
    arr=cv2.imdecode(np.frombuffer(raw[i]["bytes"],np.uint8),cv2.IMREAD_COLOR)
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((185,140))
    canvas=Image.new("RGB",(W,Hh),"white"); dr=ImageDraw.Draw(canvas)
    canvas.paste(im,(8,8))
    if can_draw:
        sk=Image.fromarray(draw_sketch(f,seed=idx)).resize((185,140)); canvas.paste(sk,(200,8))
    else:
        dr.rectangle([200,8,385,148],outline=(200,200,200)); dr.text((215,70),"此概念无语料配方",font=font,fill=(150,150,150))
    y=160
    dr.text((8,y),f"概念：{concept} / {cn}",font=font,fill=(0,110,0)); y+=24
    dr.text((8,y),f"次选：{names[top[1]]}、{names[top[2]]}",font=font,fill=(90,90,90)); y+=26
    dr.text((8,y),"英文：",font=font,fill=(180,0,0)); y+=20
    for line in [en[k:k+46] for k in range(0,len(en),46)][:2]:
        dr.text((8,y),line,font=font,fill=(0,0,0)); y+=20
    y+=6
    dr.text((8,y),f"能画：{'是' if can_draw else '否'}   出声：{snd or '无'}",font=font,fill=(0,0,150))
    tiles.append(canvas)
cols=3; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols,Hh*rows+56),"white"); dr=ImageDraw.Draw(sheet)
dr.text((14,12),"60 概念版：照片 → 概念（top-3）+ 英文描述 + 简笔画/声音（有配方才画/出声）",font=fontb,fill=(20,20,20))
for k,t in enumerate(tiles): sheet.paste(t,((k%cols)*W,56+(k//cols)*Hh))
sheet.save(str(ROOT/"logs/demo60/sheet.png")); print("图:", ROOT/"logs/demo60/sheet.png", sheet.size)
print("音频:", sorted([p.name for p in OUT.glob("*.wav")]))
