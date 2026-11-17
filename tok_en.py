#!/usr/bin/env python3
"""同一套 token 框架跑英文真实语料（55k caption），验证方法可迁移。"""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/srv/brainai")
from tok_pipe import build, TokLM
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
S=[str(c).strip().lower() for c in d["captions"]]
S=[s if s.endswith(".") else s+"." for s in S]
print("== 英文真实语料",len(S),"句")
# 英文按词切分更自然：先按空格还原成"词序列"，用同样的集合切分找词块
v,imp,seqs=build(S, min_cov=10, max_len=12, min_freq=8)
print("词表大小",len(v))
L=20
def enc(ts):
    ids=[1]+[v.get(t,3) for t in ts][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(s) for s in seqs])
n=len(X); perm=torch.randperm(n); ntr=int(n*0.9); tr,te=perm[:ntr],perm[ntr:]
m=TokLM(len(v),d=160,h=4,layers=2,imp0=imp,L=L); opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
for step in range(1,4001):
    i=tr[torch.randint(0,len(tr),(96,))]
    lg,_=m(X[i][:,:-1])
    loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0: print(f"  step {step} loss {loss.item():.3f} ppl {math.exp(min(loss.item(),15)):.1f}",flush=True)
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
        x=torch.tensor([ids])[:, -(L+1):]; lg,_=m(x); lg=lg[:,-1][0]
        val,idx=torch.topk(lg,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
        if nxt==2: break
        ids.append(nxt)
    return "".join(i2t.get(x,"?") for x in ids[1:])
print("生成样例：")
for k in range(5): print("  ",gen(k))
torch.save({"model":m.state_dict(),"vocab":v}, ROOT/"ckpt/toklm_en.pt")
