#!/usr/bin/env python3
"""验证"先检索关键词袋，再排列成句"的第一步：关键词袋能不能学出来。"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/srv/brainai")
z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
caps = [str(c) for c in z["captions"]]

tok = lambda s: re.findall(r"[a-z']+", s.lower())
cnt = Counter(w for c in caps for w in tok(c))
words = [w for w, n in cnt.most_common(1200) if n >= 2]
w2i = {w: i for i, w in enumerate(words)}
print(f"词表 {len(words)} 个词，样本 {len(caps)}")

Y = torch.zeros(len(caps), len(words))
for i, c in enumerate(caps):
    for w in set(tok(c)):
        if w in w2i:
            Y[i, w2i[w]] = 1.0

class Bag(nn.Module):
    def __init__(self, d=512, h=512, v=len(words)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, v))
    def forward(self, x):
        return self.net(x)

m = Bag()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
n = len(caps)
for step in range(1, 1201):
    idx = torch.randint(0, n, (64,))
    logits = m(feats[idx])
    loss = F.binary_cross_entropy_with_logits(logits, Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"step {step} bce {loss.item():.4f}")

with torch.no_grad():
    idx = torch.arange(64)
    logits = m(feats[idx])
    top5 = logits.topk(5, dim=1).indices
    hit = 0
    for i in range(64):
        pred = {words[j] for j in top5[i].tolist()}
        true = set(tok(caps[i]))
        hit += len(pred & true) / max(len(pred), 1)
    print(f"\nTop-5 关键词命中率: {hit/64*100:.1f}%")
    for i in range(5):
        pred = [words[j] for j in top5[i].tolist()]
        print(f"  图{i}: {pred}  ← 参考: {caps[i][:50]}")
