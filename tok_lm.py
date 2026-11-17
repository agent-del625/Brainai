#!/usr/bin/env python3
"""自建 LLM 框架：集合切分 token + 重要性维度 + 小型 Transformer 语言模型。"""
import sys, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
sys.path.insert(0,"/srv/brainai")
from pathlib import Path
ROOT=Path("/srv/brainai")
V=json.load(open(ROOT/"data/tok_zh.json"))
vocab=V["vocab"]; cov=V["cov"]
tok2i={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for t in vocab: tok2i[t]=len(tok2i)
i2tok={v:k for k,v in tok2i.items()}
mx=max(cov.values()) if cov else 1
w=np.ones(len(tok2i),np.float32)
for t in vocab: w[tok2i[t]]=cov[t]/mx          # 重要性初值 = 归一化覆盖度
seqs=[s.split("|") for s in np.load(ROOT/"data/tok_zh_seqs.npz")["seqs"]]
print("token 表",len(tok2i),"句数",len(seqs),"平均 token/句 %.1f"%np.mean([len(s) for s in seqs]))
L=16
def enc(ts):
    ids=[1]+[tok2i.get(t,3) for t in ts][:L-1]+[2]
    return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(s) for s in seqs])
n=len(X); perm=torch.randperm(n); ntr=int(n*0.9); tr,te=perm[:ntr],perm[ntr:]

class TokLM(nn.Module):
    def __init__(self,d=128,h=4,layers=2,Vn=None,w0=None):
        super().__init__()
        Vn=Vn or len(tok2i)
        self.tok=nn.Embedding(Vn,d)
        self.pos=nn.Embedding(L+1,d)
        self.imp=nn.Parameter(torch.tensor(w0 if w0 is not None else np.ones(Vn,np.float32)))
        self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(d,h,dim_feedforward=d*4,
                                  batch_first=True,activation="gelu",norm_first=True) for _ in range(layers)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,Vn,bias=False)
        self.head.weight=self.tok.weight            # 权重共享
    def forward(self,x):
        B,T=x.shape
        h=self.tok(x)+self.pos(torch.arange(T)[None])
        mask=torch.triu(torch.full((T,T),float("-inf")),diagonal=1)
        for b in self.blocks: h=b(h, src_mask=mask)
        h=self.ln(h)
        logits=self.head(h)
        return logits + self.imp.unsqueeze(0).unsqueeze(0)   # ★重要性作为输出偏置

m=TokLM(w0=w); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(1,3001):
    i=tr[torch.randint(0,len(tr),(64,))]
    logits=m(X[i][:,:-1])
    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"step {step} loss {loss.item():.3f} ppl {math.exp(min(loss.item(),20)):.1f}", flush=True)
m.eval()
with torch.no_grad():
    i=te[:300]
    lg=m(X[i][:,:-1])
    ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0).item()
print(f"测试困惑度 {math.exp(min(ll,20)):.1f}")
@torch.no_grad()
def gen(seed=0,temp=0.9,steps=16):
    torch.manual_seed(seed)
    ids=[1]
    for _ in range(steps):
        x=torch.tensor([ids])[:, -(L+1):]
        lg=m(x)[:, -1][0]
        v,i=torch.topk(lg,20)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        ids.append(nxt)
    return "".join(i2tok.get(i,"?") for i in ids[1:])
print("生成样例：")
for k in range(5): print("  ",gen(seed=k))
torch.save({"model":m.state_dict(),"vocab":tok2i}, ROOT/"ckpt/toklm.pt")
