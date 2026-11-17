#!/usr/bin/env python3
"""在现有描述模型上对比：贪心 vs 束搜索（beam=4）的 chrF。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()
for p in clf.parameters(): p.requires_grad=False
ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"]
SKY=["clouds","sky","sunny","night","trees","sunset"]
CNT=["one","two","three","group","crowd"]
ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in caps])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
for s in range(800):
    i=torch.randint(0,len(feats),(128,))
    l=F.binary_cross_entropy_with_logits(at(feats[i]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
for p in at.parameters(): p.requires_grad=False
with torch.no_grad():
    COND=torch.cat([feats, torch.sigmoid(clf(feats)), torch.sigmoid(at(feats))], -1)
cc=torch.load(ROOT/"ckpt/cap2.pt",map_location="cpu",weights_only=False)
vocab=cc["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=320,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],512), nn.GELU(), nn.LayerNorm(512),
                               nn.Linear(512,h), nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.gru=nn.GRU(emb,h,batch_first=True)
        self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
m=Cap(); m.load_state_dict(cc["model"]); m.eval()

def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)

@torch.no_grad()
def greedy(cond,steps=38):
    h=m.h0(cond); tok=torch.tensor([[1]]); out=[]
    for _ in range(steps):
        h2,_=m.gru(m.emb(tok),h)
        nxt=m.out(h2[:,-1]).argmax(-1,keepdim=True)
        if int(nxt)==2: break
        out.append(int(nxt)); tok=nxt; h=h2
    return "".join(inv.get(i,"") for i in out)

@torch.no_grad()
def beam(cond,width=4,steps=38):
    h0=m.h0(cond)
    beams=[([1],h0,0.0)]
    done=[]
    for _ in range(steps):
        cand=[]
        for toks,h,score in beams:
            tok=torch.tensor([[toks[-1]]])
            h2,_=m.gru(m.emb(tok),h)
            lp=F.log_softmax(m.out(h2[:,-1]),-1)[0]
            top=torch.topk(lp,width)
            for v,idx in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((toks+[idx], h2, score+v))
        cand.sort(key=lambda t:-t[2])
        beams=[]
        for toks,h,sc in cand[:width]:
            if toks[-1]==2: done.append((toks,sc/max(len(toks)-1,1)))
            else: beams.append((toks,h,sc))
        if not beams: break
    if not done: done=[(b[0],b[2]/max(len(b[0])-1,1)) for b in beams]
    toks=max(done,key=lambda t:t[1])[0]
    return "".join(inv.get(i,"") for i in toks[1:-1] if i!=2)

n=len(COND); perm=torch.randperm(n); te=perm[int(n*0.9):][:80].tolist()
g=[chrF:=chrf(greedy(COND[i:i+1]), caps[i]) for i in te]
b=[chrf(beam(COND[i:i+1],4), caps[i]) for i in te]
print(f"贪心 chrF {np.mean(g)*100:.1f}%   束搜索(4) chrF {np.mean(b)*100:.1f}%")
for i in te[:3]:
    print("  贪心:", greedy(COND[i:i+1])[:60])
    print("  束搜:", beam(COND[i:i+1],4)[:60])
    print("  参考:", caps[i][:60])
