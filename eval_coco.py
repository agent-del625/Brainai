#!/usr/bin/env python3
"""把现有模型（Flickr8k 上训的）拿到 COCO 上测：跨数据集泛化能力。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/coco_clip.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
ck=torch.load(ROOT/"ckpt/cap10.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; inv={v:k for k,v in vocab.items()}
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
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
m=Cap(); m.load_state_dict(ck["model"]); m.eval()
@torch.no_grad()
def gen(cond,seed=0,temp=0.7,k=20):
    torch.manual_seed(seed); h=m.h0(cond); tok=torch.tensor([[1]]); cp=m.cproj(cond); out=[]
    for _ in range(40):
        o,hn=m.gru(torch.cat([m.emb(tok),cp.unsqueeze(1)],-1),h)
        v,i=torch.topk(m.out(o[:,-1])[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
def chrf(p,r,n=3):
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp_,cr=Counter(gp),Counter(gr); ov=sum((cp_&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
imgs=sorted(set(IMG.tolist()))[:300]
sc=[]
for j,i in enumerate(imgs):
    g=gen(COND[i:i+1],seed=j)
    refs=[CAPS[k] for k in range(len(CAPS)) if int(IMG[k])==i]
    sc.append(max(chrf(g,r) for r in refs))
print(f"COCO 上（没见过）chrF(3) 最好匹配：{np.mean(sc)*100:.1f}%")
for i in imgs[:5]:
    print("  生成:",gen(COND[i:i+1],seed=i)); print("  参考:",CAPS[int(np.where(IMG.numpy()==i)[0][0])][:70])
