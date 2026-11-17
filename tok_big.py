#!/usr/bin/env python3
"""大语料（52.8 万句）上的 token 学习 + 语言模型。"""
import sys, math, re, numpy as np, torch, torch.nn.functional as F
from collections import Counter
sys.path.insert(0,"/srv/brainai")
from tok_pipe import TokLM
from pathlib import Path
ROOT=Path("/srv/brainai")
S=[str(s).lower() for s in np.load(ROOT/"data/en_big_sents.npy",allow_pickle=True)]
print("语料",len(S),"句")
WORD_RE=re.compile(r"[a-z']+|[.,]")
sents=[WORD_RE.findall(s) for s in S]
wc=Counter(w for s in sents for w in s)
print("不同词",len(wc))
words={w for w,c in wc.items() if c>=25}
gram=Counter()
for w,c in wc.items():
    if len(w)<5 or w in words: continue
    for L in (4,3,2):
        for i in range(len(w)-L+1): gram[w[i:i+L]]+=c
pieces={g for g,c in gram.items() if c>=400}
chars=set(ch for w in wc for ch in w)
print(f"整词 {len(words)}  片段 {len(pieces)}  字符 {len(chars)}")
def decompose(w):
    if w in words: return [w]
    out=[];i=0
    while i<len(w):
        hit=None
        for L in (4,3,2):
            if i+L<=len(w) and w[i:i+L] in pieces: hit=w[i:i+L]; break
        if hit: out.append(hit); i+=len(hit)
        else: out.append(w[i]); i+=1
    return out
def tokenize(s):
    out=[]
    for w in s: out+=decompose(w); out.append(" ")
    return out
seqs=[tokenize(s) for s in sents]
cnt=Counter(x for t in seqs for x in t)
v={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for w,n in cnt.most_common(): v[w]=len(v)
imp=np.zeros(len(v),np.float32); mx=max(cnt.values())
for w,i in v.items():
    if i>3: imp[i]=cnt[w]/mx
print(f"词表 {len(v)}  token/句 {np.mean([len(t) for t in seqs]):.1f}  OOV {sum(1 for t in seqs for x in t if x not in v)}")
L=30
def enc(ts):
    ids=[1]+[v.get(x,3) for x in ts][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(s) for s in seqs])
print("张量",tuple(X.shape))
n=len(X); perm=torch.randperm(n); ntr=int(n*0.95); tr,te=perm[:ntr],perm[ntr:]
m=TokLM(len(v),d=192,h=6,layers=3,imp0=imp,L=L); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
for step in range(1,4001):
    i=tr[torch.randint(0,len(tr),(128,))]
    lg,_=m(X[i][:,:-1])
    loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0: print(f"  step {step} ppl {math.exp(min(loss.item(),15)):.1f}",flush=True)
m.eval()
with torch.no_grad():
    i=te[:600]; lg,_=m(X[i][:,:-1])
    ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0).item()
print(f"测试困惑度 {math.exp(min(ll,15)):.1f}")
i2t={vv:k for k,vv in v.items()}
@torch.no_grad()
def gen(seed,temp=0.8):
    torch.manual_seed(seed); ids=[1]
    for _ in range(L):
        x=torch.tensor([ids])[:,-(L+1):]; lg,_=m(x); lg=lg[:,-1][0]
        val,idx=torch.topk(lg,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
        if nxt==2: break
        ids.append(nxt)
    return "".join(i2t.get(x,"?") for x in ids[1:])
print("生成样例：")
for k in range(5): print("  ",gen(k))
torch.save({"model":m.state_dict(),"vocab":v}, ROOT/"ckpt/toklm_big.pt")
