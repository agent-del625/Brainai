#!/usr/bin/env python3
"""袋头：CLIP 特征 → 多标签关键词（真实端到端的第一步）。"""
import re
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]
cnt = Counter(w for c in caps for w in words_of(c))
content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
c2i = {w: i for i, w in enumerate(content)}
Y = torch.zeros(len(caps), len(content))
for i, c in enumerate(caps):
    for w in set(words_of(c)):
        if w in c2i: Y[i, c2i[w]] = 1.0

class Bag(nn.Module):
    def __init__(self, d=512, h=768, v=len(content)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(0.1),
                                 nn.Linear(h,h//2), nn.GELU(), nn.Linear(h//2, v))
    def forward(self, x): return self.net(x)

m = Bag(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
n = len(caps)
for step in range(1, 2001):
    idx = torch.randint(0, n, (64,))
    loss = F.binary_cross_entropy_with_logits(m(feats[idx]), Y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step} bce {loss.item():.4f}")

with torch.no_grad():
    lg = m(feats[:300])
    # 用"概率>0.5"的词构成袋，命中率
    hit, size = 0, 0
    for i in range(300):
        pred = {content[j] for j in (lg[i] > 0.0).nonzero().flatten().tolist()}
        gold = set(words_of(caps[i])) & set(content)
        hit += len(pred & gold); size += len(pred)
    print(f"阈值 0：平均每张出 {size/300:.1f} 个词，命中率（精确率）{hit/max(size,1)*100:.0f}%")
torch.save({"model": m.state_dict(), "content": content}, ROOT/"ckpt/bag_head.pt")
