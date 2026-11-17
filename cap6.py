#!/usr/bin/env python3
"""描述生成 v5：6000 图 × 5 caption = 3 万对，按图像分组划分避免泄漏。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
print("图像",len(FE),"图文对",len(CAPS))
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
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=5e-4)
for s in range(1200):
    i=torch.randint(0,len(CAPS),(256,))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
for p in at.parameters(): p.requires_grad=False
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(clf(FE)), torch.sigmoid(at(FE))], -1)
chars=sorted({ch for c in CAPS for ch in c})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
L=40
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]
    return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(c) for c in CAPS])
class Cap(nn.Module):
    def __init__(self,h=320,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512,h), nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.gru=nn.GRU(emb,h,batch_first=True)
        self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
m=Cap(); opt=torch.optim.AdamW(m.parameters(), lr=5e-4)
nimg=len(FE); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img,te_img=set(perm[:ntr].tolist()),set(perm[ntr:].tolist())
tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]
te=[i for i in range(len(CAPS)) if int(IMG[i]) in te_img]
print("训练对",len(tr),"测试对",len(te))
for step in range(1,10001):
    i=torch.tensor(np.random.choice(tr,96))
    cnd=COND[IMG[i]]
    drop=(torch.rand(cnd.size(0),1)<0.10).float()
    cnd=cnd*(1-drop)
    h0=m.h0(cnd); out,_=m.gru(m.emb(X[i][:,:-1]),h0)
    loss=F.cross_entropy(m.out(out).reshape(-1,len(vocab)), X[i][:,1:].reshape(-1), ignore_index=0, )
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%500==0: print(f"step {step} ce {loss.item():.3f}", flush=True)
def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
@torch.no_grad()
def gen(cond,steps=40):
    h=m.h0(cond); tok=torch.tensor([[1]]); out=[]
    for _ in range(steps):
        o,hn=m.gru(m.emb(tok),h)
        nxt=m.out(o[:,-1]).argmax(-1,keepdim=True)
        if int(nxt)==2: break
        out.append(int(nxt)); tok=nxt; h=hn
    return "".join(inv.get(i,"") for i in out)
m.eval()
sel=te[:150]
sc=[chrf(gen(COND[IMG[i]:IMG[i]+1]), CAPS[i]) for i in sel]
print(f"chrF(3) 贪心：{np.mean(sc)*100:.1f}%（基线 37.0%）")
for i in sel[:3]:
    print("  生成:",gen(COND[IMG[i]:IMG[i]+1])[:60]); print("  参考:",CAPS[i][:60])
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/cap6.pt")
