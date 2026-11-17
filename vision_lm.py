#!/usr/bin/env python3
"""把图像接进自建 token LM：CLIP 特征作为前缀，生成描述。"""
import sys, math, re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
from collections import Counter
sys.path.insert(0,"/srv/brainai")
sys.argv=["x"]
from tok_pipe import Block
from pathlib import Path
ROOT=Path("/srv/brainai")
# 1) 复用大语料学到的 token 表
S=[str(s).lower() for s in np.load(ROOT/"data/en_big_sents.npy",allow_pickle=True)]
WORD_RE=re.compile(r"[a-z']+|[.,]")
def norm(s): return WORD_RE.findall(s)
wc=Counter(w for s in S for w in norm(s))
words={w for w,c in wc.items() if c>=25}
gram=Counter()
for w,c in wc.items():
    if len(w)<5 or w in words: continue
    for L in (4,3,2):
        for i in range(len(w)-L+1): gram[w[i:i+L]]+=c
pieces={g for g,c in gram.items() if c>=400}
cnt=Counter()
for s in S:
    for w in norm(s):
        if w in words: cnt[w]+=1
        else:
            o=[];i=0
            while i<len(w):
                hit=None
                for L in (4,3,2):
                    if i+L<=len(w) and w[i:i+L] in pieces: hit=w[i:i+L]; break
                if hit: o.append(hit); i+=len(hit)
                else: o.append(w[i]); i+=1
            for x in o: cnt[x]+=1
        cnt[" "]+=1
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for w,n in cnt.most_common(): vocab[w]=len(vocab)
imp=np.zeros(len(vocab),np.float32); mx=max(cnt.values())
for w,i in vocab.items():
    if i>3: imp[i]=cnt[w]/mx
print("token 表",len(vocab))
def tok_sent(s):
    out=[]
    for w in norm(s):
        if w in words: out.append(w)
        else:
            i=0
            while i<len(w):
                hit=None
                for L in (4,3,2):
                    if i+L<=len(w) and w[i:i+L] in pieces: hit=w[i:i+L]; break
                if hit: out.append(hit); i+=len(hit)
                else: out.append(w[i]); i+=1
        out.append(" ")
    return out
# 2) 图文对
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
FE=torch.from_numpy(d["feats"]).float(); FE=F.normalize(FE,dim=-1)
CAPS=[str(c).lower() for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
L=26; P=4
def enc(ts):
    ids=[1]+[vocab.get(x,3) for x in ts][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
Xs=torch.tensor([enc(tok_sent(c)) for c in CAPS])
print("图文对",len(Xs),"平均 token",float((Xs!=0).sum(1).float().mean()))
class VLM(nn.Module):
    def __init__(self,d=192,h=6,layers=3,Vn=None):
        super().__init__()
        Vn=Vn or len(vocab)
        self.proj=nn.Linear(512,d*P)
        self.tok=nn.Embedding(Vn,d); self.pos=nn.Embedding(P+L+1,d)
        self.imp=nn.Parameter(torch.tensor(imp))
        self.blocks=nn.ModuleList([Block(d,h) for _ in range(layers)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,Vn,bias=False); self.head.weight=self.tok.weight
    def forward(self,f,x):
        B,T=x.shape
        pre=self.proj(f).view(B,P,-1)
        h=torch.cat([pre, self.tok(x)],1)+self.pos(torch.arange(P+T)[None])
        impx=torch.cat([torch.ones(B,P), self.imp[x].clamp(min=1e-3)],1)
        for b in self.blocks: h=b(h,impx)
        h=self.ln(h)[:,P:]
        return self.head(h)+self.imp.unsqueeze(0).unsqueeze(0)
n=len(Xs); perm=torch.randperm(len(FE)); ntr=int(len(FE)*0.9)
tr_img=set(perm[:ntr].tolist()); tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]
te=[i for i in range(len(CAPS)) if int(IMG[i]) not in tr_img]
m=VLM(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
print("训练对",len(tr),"测试对",len(te))
for step in range(1,3001):
    i=torch.tensor(np.random.choice(tr,64))
    lg=m(FE[IMG[i]], Xs[i][:,:-1])
    loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), Xs[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"  step {step} ppl {math.exp(min(loss.item(),15)):.1f}",flush=True)
m.eval()
with torch.no_grad():
    i=torch.tensor(te[:400]); lg=m(FE[IMG[i]], Xs[i][:,:-1])
    ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), Xs[i][:,1:].reshape(-1), ignore_index=0).item()
print(f"测试困惑度 {math.exp(min(ll,15)):.1f}")
i2t={v:k for k,v in vocab.items()}
@torch.no_grad()
def gen(fi,seed=0,temp=0.8):
    torch.manual_seed(seed)
    B=1; pre=m.proj(FE[fi:fi+1]).view(B,P,-1)
    ids=[1]; h=pre+ m.pos(torch.arange(P)[None])
    for t in range(L):
        x=torch.tensor([ids]) if ids else torch.tensor([[1]])
        seq=torch.cat([pre, m.tok(x)+m.pos(torch.arange(P,P+x.size(1))[None])],1)
        impx=torch.cat([torch.ones(B,P), m.imp[x].clamp(min=1e-3)],1)
        for b in m.blocks: seq=b(seq,impx)
        logits=m.head(m.ln(seq))[:,-1][0]
        val,idx=torch.topk(logits,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
        if nxt==2: break
        ids.append(nxt)
    return "".join(i2t.get(x,"?") for x in ids[1:])
def chrf(p,r,n=3):
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp_,cr=Counter(gp),Counter(gr); ov=sum((cp_&cr).values())
    if ov==0: return 0.0
    P_,R_=ov/len(gp),ov/len(gr); return 2*P_*R_/(P_+R_)
sel=te[:60]; sc=[]
for j,i in enumerate(sel):
    g=gen(int(IMG[i]),seed=j)
    refs=[CAPS[k] for k in range(len(CAPS)) if int(IMG[k])==int(IMG[i])]
    sc.append(max(chrf(g,r) for r in refs))
print(f"chrF(3) {np.mean(sc)*100:.1f}%（参考：旧 caption 模型 28.4%）")
for j,i in enumerate(sel[:4]):
    print("  生成:",gen(int(IMG[i]),seed=j)); print("  参考:",CAPS[i][:70])
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/vlm.pt")
