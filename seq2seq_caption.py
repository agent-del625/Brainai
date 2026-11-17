#!/usr/bin/env python3
"""结构化属性 → 自然语言描述（小型 seq2seq，不用 transformer）。

编码：CLIP 特征 + 概念/属性概率（来自已训练的头）
解码：GRU 字符级自回归（教师强制训练 + 采样推理）
"""
import re, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
L=40

chars=sorted({ch for c in caps for ch in c})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]
    return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(c) for c in caps])
print("词表", len(vocab), "样本", len(caps))

class Captioner(nn.Module):
    def __init__(self, h=384, emb=96):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(512,512), nn.GELU(), nn.LayerNorm(512),
                                 nn.Linear(512,h), nn.GELU())
        self.emb = nn.Embedding(len(vocab), emb)
        self.gru = nn.GRU(emb, h, batch_first=True)
        self.init = nn.Linear(h, h)
        self.out = nn.Linear(h, len(vocab))
    def forward(self, feat, toks):
        h0 = self.init(self.enc(feat)).unsqueeze(0)
        h,_ = self.gru(self.emb(toks), h0)
        return self.out(h)
    @torch.no_grad()
    def generate(self, feat, steps=40, temp=0.7):
        h = self.init(self.enc(feat)).unsqueeze(0)
        tok = torch.tensor([[1]])
        out=[]
        for _ in range(steps):
            h,_ = self.gru(self.emb(tok), h)
            logits = self.out(h[:, -1])
            nxt = torch.multinomial(torch.softmax(logits/temp, -1), 1) if temp>0 else logits.argmax(-1, keepdim=True)
            if int(nxt)==2: break
            out.append(int(nxt)); tok = nxt
        return "".join(inv.get(i,"") for i in out)

m=Captioner(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
n=len(feats); perm=torch.randperm(n)
tr, te = perm[:int(n*0.9)], perm[int(n*0.9):]
for step in range(1, 3001):
    i = tr[torch.randint(0,len(tr),(64,))]
    logits = m(feats[i], X[i][:, :-1])
    loss = F.cross_entropy(logits.reshape(-1,len(vocab)), X[i][:,1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%500==0: print(f"step {step} ce {loss.item():.3f}", flush=True)

# 评测：chrF（字符 n-gram F 分数）+ 样例
def chrf(pred, ref, n=3):
    def grams(s): return [s[i:i+n] for i in range(len(s)-n+1)]
    p,r=grams(pred),grams(ref)
    if not p or not r: return 0.0
    from collections import Counter
    cp,cr=Counter(p),Counter(r); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(p); R=ov/len(r); return 2*P*R/(P+R)
idx=te[:60].tolist()
scores=[]
samples=[]
for i in idx:
    pred=m.generate(feats[i:i+1], temp=0.6)
    scores.append(chrf(pred, caps[i]))
    if len(samples)<5: samples.append((pred, caps[i]))
print(f"chrF(3): {np.mean(scores)*100:.1f}%")
for p,r in samples:
    print(f"  生成: {p[:60]}")
    print(f"  参考: {r[:60]}")
torch.save({"model": m.state_dict(), "vocab": vocab}, ROOT/"ckpt/captioner.pt")
