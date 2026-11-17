#!/usr/bin/env python3
"""英文真实语料：对比「纯词」vs「自动合并词块 + 重要性」两种 token 表。"""
import sys, math, re, numpy as np, torch, torch.nn.functional as F
from collections import Counter
sys.path.insert(0,"/srv/brainai")
from tok_pipe import TokLM
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
S=[re.findall(r"[a-z']+|[.,]", str(c).lower()) for c in d["captions"]]
print("英文语料",len(S),"句")
wc=Counter(w for s in S for w in s)
print("不同词",len(wc))
# 候选词块：相邻 2-3 词组合，按"覆盖率"筛选（这就是集合并/差的统计版本）
big=Counter(); tri=Counter()
for s in S:
    for i in range(len(s)-1): big[(s[i],s[i+1])]+=1
    for i in range(len(s)-2): tri[(s[i],s[i+1],s[i+2])]+=1
merged={}
for (a,b),c in big.items():
    if c>=40 and c/min(wc[a],wc[b])>=0.35: merged[(a,b)]=c
for (a,b,cc2),c in tri.items():
    if c>=25 and c/min(wc[a],wc[b],wc[cc2])>=0.4: merged[(a,b,cc2)]=c
print("学到的词块",len(merged),"示例",[" ".join(k) for k in list(merged)[:6]])
def to_tokens(s, use_merge):
    out=[];i=0
    while i<len(s):
        hit=None
        if use_merge:
            for L in (3,2):
                if i+L<=len(s) and tuple(s[i:i+L]) in merged: hit=tuple(s[i:i+L]); break
        if hit: out.append(" ".join(hit)); i+=len(hit)
        else: out.append(s[i]); i+=1
    return out
def make_vocab(seqs):
    c=Counter(x for t in seqs for x in t)
    v={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
    for w,n in c.most_common():
        if n>=3: v[w]=len(v)
    imp=np.zeros(len(v),np.float32)
    mx=max(c.values())
    for w,i in v.items():
        if i>3: imp[i]=c[w]/mx
    return v,imp
def run(seqs,tag,steps=2500,L=20):
    v,imp=make_vocab(seqs)
    def enc(ts):
        ids=[1]+[v.get(t,3) for t in ts][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
    X=torch.tensor([enc(s) for s in seqs])
    n=len(X); perm=torch.randperm(n); ntr=int(n*0.9); tr,te=perm[:ntr],perm[ntr:]
    m=TokLM(len(v),d=160,h=4,layers=2,imp0=imp,L=L); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    for step in range(1,steps+1):
        i=tr[torch.randint(0,len(tr),(96,))]
        lg,_=m(X[i][:,:-1])
        loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        i=te[:400]; lg,_=m(X[i][:,:-1])
        ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0).item()
    i2t={vv:k for k,vv in v.items()}
    @torch.no_grad()
    def gen(seed,temp=0.85):
        torch.manual_seed(seed); ids=[1]
        for _ in range(L):
            x=torch.tensor([ids])[:,-(L+1):]; lg,_=m(x); lg=lg[:,-1][0]
            val,idx=torch.topk(lg,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
            if nxt==2: break
            ids.append(nxt)
        return "".join(i2t.get(x,"?") for x in ids[1:])
    print(f"[{tag}] 词表 {len(v)}  token/句 {np.mean([len(t) for t in seqs]):.1f}  测试困惑度 {math.exp(min(ll,15)):.1f}")
    for k in range(3): print("    ",gen(k))
    return math.exp(min(ll,15))
a=run([to_tokens(s,False) for s in S],"纯词")
b=run([to_tokens(s,True) for s in S],"自动合并词块")
print(f"结论：合并后困惑度 {a:.1f} → {b:.1f}（{'改善' if b<a else '变差'} {abs(a-b):.1f}）")
