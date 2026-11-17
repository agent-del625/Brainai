#!/usr/bin/env python3
"""概念级语言模型：学习"哪些概念会一起出现、以什么顺序"，从分布里采样新组合。

这才是生成：不是随机拼，而是从数据学到的概念共现分布中采样。
"""
import random, re
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
ck = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
concepts = ck["names"]
s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
vocab = s2["content"]; ci = {w:i for i,w in enumerate(vocab)}
V = len(vocab) + 2   # +bos +eos

seqs = []
for c in caps:
    ids = [ci[w] for w in words_of(c) if w in ci]
    if len(ids) >= 2:
        seqs.append([V-2] + ids + [V-1])
print(f"训练序列 {len(seqs)} 条，内容词表 {len(vocab)}")

class ConceptLM(nn.Module):
    def __init__(self, d=96, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.gru = nn.GRU(d, h, batch_first=True)
        self.out = nn.Linear(h, V)
    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.out(h)

m = ConceptLM(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
L = 8
batch = []
for s in seqs:
    s = s[:L] + [0]*(L-len(s))
    batch.append(s)
X = torch.tensor(batch)
for step in range(1, 2001):
    idx = torch.randint(0, X.size(0), (128,))
    logits = m(X[idx][:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), X[idx][:, 1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step} ce {loss.item():.3f}")

inv = {i: w for w, i in ci.items()}
m.eval()
def sample(temp=0.9, n=1):
    out = []
    with torch.no_grad():
        x = torch.full((n, 1), V-2)
        h = None
        for _ in range(6):
            logits = m(x[:, -1:])[:, -1]
            p = torch.softmax(logits/temp, -1)
            nxt = torch.multinomial(p, 1)
            if (nxt == V-1).all(): break
            x = torch.cat([x, nxt], 1)
            out.append(nxt.squeeze(1))
    res = []
    for j in range(n):
        res.append([inv[int(t[j])] for t in out if int(t[j]) in inv])
    return res

seen = {" ".join(sorted({w for w in words_of(c) if w in ci})) for c in caps}
novel, dup = [], 0
for _ in range(60):
    for s in sample(0.9, 8):
        if len(s) < 2: continue
        if not any(w in concepts for w in s): continue
        key = " ".join(sorted(set(s)))
        if key in seen: dup += 1
        else: novel.append(s)
print(f"\n采样 480 条：新组合 {len(novel)} 条（训练集没出现过），重复 {dup} 条")
for s in novel[:10]:
    print("  ", " · ".join(s))
torch.save({"model": m.state_dict(), "vocab": vocab}, ROOT/"ckpt/concept_lm.pt")
