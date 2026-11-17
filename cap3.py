#!/usr/bin/env python3
"""描述生成 v3：2 层 GRU 解码 + 标签平滑 + 余弦学习率，目标 chrF > 37%（贪心基线）。"""
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
chars=sorted({ch for c in caps for ch in c})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
L=38
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]
    return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(c) for c in caps])
class Cap(nn.Module):
    def __init__(self,h=384,emb=128):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],640), nn.GELU(), nn.LayerNorm(640), nn.Linear(640,h), nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb)
        self.gru=nn.GRU(emb,h,num_layers=2,batch_first=True,dropout=0.1)
        self.init=nn.Linear(h,h*2); self.out=nn.Linear(h,len(vocab))
    def h0(self,c):
        h=self.init(self.enc(c)).view(2,-1,self.gru.hidden_size)
        return h.contiguous()
m=Cap(); opt=torch.optim.AdamW(m.parameters(), lr=1.2e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)
n=len(feats); perm=torch.randperm(n); tr,te=perm[:int(n*0.9)],perm[int(n*0.9):]
for step in range(1,3001):
    i=tr[torch.randint(0,len(tr),(64,))]
    logits=m(COND[i],X[i][:,:-1]) if hasattr(m,"decode") else None
    h0=m.h0(COND[i]); out,_=m.gru(m.emb(X[i][:,:-1]),h0); logits=m.out(out)
    loss=F.cross_entropy(logits.reshape(-1,len(vocab)), X[i][:,1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sched.step()
    if step%250==0: print(f"step {step} ce {loss.item():.3f}", flush=True)
def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
@torch.no_grad()
def gen(cond,steps=38):
    h=m.h0(cond); tok=torch.tensor([[1]]); out=[]
    for _ in range(steps):
        o,hn=m.gru(m.emb(tok),h)
        nxt=m.out(o[:,-1]).argmax(-1,keepdim=True)
        if int(nxt)==2: break
        out.append(int(nxt)); tok=nxt; h=hn
    return "".join(inv.get(i,"") for i in out)
m.eval()
idx=te[:120].tolist()
sc=[chrf(gen(COND[i:i+1]),caps[i]) for i in idx]
print(f"chrF(3) 贪心：{np.mean(sc)*100:.1f}%")
for i in idx[:3]:
    print("  生成:",gen(COND[i:i+1])[:60]); print("  参考:",caps[i][:60])
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/cap3.pt")
