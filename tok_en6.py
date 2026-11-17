#!/usr/bin/env python3
"""英文 v4：让模型自己学"拆词"——高频词整词，生词拆成高频字母片段（无人工规则）。"""
import sys, math, re, numpy as np, torch, torch.nn.functional as F
from collections import Counter
sys.path.insert(0,"/srv/brainai")
from tok_pipe import TokLM
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
TXT=[str(c).lower().strip() for c in d["captions"]]
WORD_RE=re.compile(r"[a-z']+|[.,]")
sents=[WORD_RE.findall(t) for t in TXT]
wc=Counter(w for s in sents for w in s)
print("句子",len(sents),"不同词",len(wc))
# 1) 整词（高频）
WORD_MIN=12
words={w for w,c in wc.items() if c>=WORD_MIN}
# 2) 字母片段（2-4 长），从所有词内部统计——这就是"模型自己发现的拆词单位"
gram=Counter()
for w,c in wc.items():
    if len(w)<5 or w in words: continue          # 只从生词里学片段
    for L in (4,3,2):
        for i in range(len(w)-L+1):
            gram[w[i:i+L]]+=c
PIECE_MIN=150
pieces={g for g,c in gram.items() if c>=PIECE_MIN}
chars=set(ch for w in wc for ch in w)
print(f"整词 {len(words)}  学到的字母片段 {len(pieces)}（示例 {list(pieces)[:10]}）  字符 {len(chars)}")
VOC=set(words)|pieces|chars|{"<pad>","<bos>","<eos>"," ","<unk>"}
def decompose(w):
    if w in words: return [w]
    out=[];i=0
    while i<len(w):
        hit=None
        for L in (4,3,2):
            if i+L<=len(w) and w[i:i+L] in pieces and w[i:i+L] in VOC: hit=w[i:i+L]; break
        if hit: out.append(hit); i+=len(hit)
        else: out.append(w[i]); i+=1
    return out
def tokenize(s):
    out=[]
    for w in s:
        out+=decompose(w); out.append(" ")
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
n=len(X); perm=torch.randperm(n); ntr=int(n*0.9); tr,te=perm[:ntr],perm[ntr:]
m=TokLM(len(v),d=160,h=4,layers=2,imp0=imp,L=L); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
for step in range(1,1501):
    i=tr[torch.randint(0,len(tr),(96,))]
    lg,_=m(X[i][:,:-1])
    loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"  step {step} ppl {math.exp(min(loss.item(),15)):.1f}",flush=True)
m.eval()
with torch.no_grad():
    i=te[:400]; lg,_=m(X[i][:,:-1])
    ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0).item()
print(f"测试困惑度 {math.exp(min(ll,15)):.1f}")
i2t={vv:k for k,vv in v.items()}
@torch.no_grad()
def gen(seed,temp=0.85):
    torch.manual_seed(seed); ids=[1]
    for _ in range(L):
        x=torch.tensor([ids])[:,-(L+1):]; lg,_=m(x); lg=lg[:,-1][0]
        val,idx=torch.topk(lg,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
        if nxt==2: break
        ids.append(nxt)
    return "".join(i2t.get(x,"?") for x in ids[1:]).replace(" "," ")
print("生成样例：")
for k in range(4): print("  ",gen(k))
