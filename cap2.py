#!/usr/bin/env python3
"""描述生成 v2：编码输入 = CLIP + 概念概率 + 属性概率，解码 GRU 字符级。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
L=38

# 概念/属性头（复用之前的定义，直接加载 ckpt）
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

chars=sorted({ch for c in caps for ch in c})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]
    return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(c) for c in caps])

with torch.no_grad():
    CP=torch.sigmoid(clf(feats)); AP=torch.sigmoid(at(feats))
COND=torch.cat([feats, CP, AP], -1)          # 512+23+32
print("条件维度", COND.shape)

class Cap(nn.Module):
    def __init__(self, h=320, emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],512), nn.GELU(), nn.LayerNorm(512),
                               nn.Linear(512,h), nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb)
        self.gru=nn.GRU(emb,h,batch_first=True)
        self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def forward(self,cond,toks):
        h0=self.init(self.enc(cond)).unsqueeze(0)
        h,_=self.gru(self.emb(toks),h0)
        return self.out(h)
    @torch.no_grad()
    def generate(self,cond,steps=38,temp=0.6):
        h=self.init(self.enc(cond)).unsqueeze(0); tok=torch.tensor([[1]]); out=[]
        for _ in range(steps):
            h,_=self.gru(self.emb(tok),h)
            lg=self.out(h[:,-1])
            nxt=torch.multinomial(torch.softmax(lg/temp,-1),1)
            if int(nxt)==2: break
            out.append(int(nxt)); tok=nxt
        return "".join(inv.get(i,"") for i in out)

m=Cap(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
n=len(feats); perm=torch.randperm(n); tr,te=perm[:int(n*0.9)],perm[int(n*0.9):]
for step in range(1, 2501):
    i=tr[torch.randint(0,len(tr),(48,))]
    logits=m(COND[i], X[i][:,:-1])
    loss=F.cross_entropy(logits.reshape(-1,len(vocab)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%250==0: print(f"step {step} ce {loss.item():.3f}", flush=True)

def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
sc=[]; samples=[]
for i in te[:50].tolist():
    p=m.generate(COND[i:i+1]); sc.append(chrf(p,caps[i]))
    if len(samples)<5: samples.append((p,caps[i]))
print(f"chrF(3): {np.mean(sc)*100:.1f}%")
for p,r in samples:
    print(f"  生成: {p[:58]}")
    print(f"  参考: {r[:58]}")
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/cap2.pt")
