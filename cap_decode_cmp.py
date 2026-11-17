#!/usr/bin/env python3
"""对比解码策略：贪心 / 温度采样 / top-k，看 chrF 与多样性。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
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
def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
@torch.no_grad()
def gen(cond,mode="greedy",temp=0.8,k=10,steps=40,seed=0):
    torch.manual_seed(seed)
    h=m.h0(cond); tok=torch.tensor([[1]]); cp=m.cproj(cond); out=[]
    for _ in range(steps):
        o,hn=m.gru(torch.cat([m.emb(tok),cp.unsqueeze(1)],-1),h)
        lg=m.out(o[:,-1])[0]
        if mode=="greedy": nxt=int(lg.argmax())
        else:
            v,i=torch.topk(lg,min(k,lg.numel()))
            probs=torch.softmax(v/temp,-1); nxt=int(i[torch.multinomial(probs,1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(i,"") for i in out)
n=len(FE); perm=torch.randperm(n); te=perm[int(n*0.9):].tolist()
sel=[]
for i in te:
    if len(sel)>=120: break
    sel.append(i)
ref=[CAPS[int(np.where(IMG.numpy()==i)[0][0])] for i in sel]
for mode,temp,k in [("greedy",1.0,1),("sample",0.7,20),("sample",0.9,20),("sample",1.1,20)]:
    outs=[gen(COND[i:i+1],mode,temp,k,seed=j) for j,i in enumerate(sel)]
    sc=np.mean([chrf(o,r) for o,r in zip(outs,ref)])
    div=len(set(outs))
    print(f"{mode} temp={temp} k={k}: chrF {sc*100:.1f}%  120 张图 {div} 种句子")
print("样例（采样 temp0.9）:")
for j in range(3):
    i=sel[j]
    print("  生成:",gen(COND[i:i+1],"sample",0.9,20,seed=j)); print("  参考:",ref[j][:60])
