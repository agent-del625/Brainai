#!/usr/bin/env python3
"""想象力检验：由句子预测它"应该长什么样"（CLIP 空间），再看这个预测像不像真实图片。"""
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

class Imagine(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.emb = nn.Embedding(len(content), d)
        self.net = nn.Sequential(nn.Linear(d, 384), nn.GELU(), nn.LayerNorm(384),
                                 nn.Linear(384, 512))
    def forward(self, ids, mask):
        e = self.emb(ids) * mask.unsqueeze(-1)
        x = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.net(x), dim=-1)

m = Imagine(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
ids, mask = [], []
for c in caps:
    ws = [c2i[w] for w in words_of(c) if w in c2i][:10]
    ids.append(ws + [0]*(10-len(ws))); mask.append([1]*len(ws) + [0]*(10-len(ws)))
ids = torch.tensor(ids); mask = torch.tensor(mask).float()
target = F.normalize(feats, dim=-1)

for step in range(1, 2501):
    idx = torch.randint(0, len(caps), (64,))
    pred = m(ids[idx], mask[idx])
    loss = (1 - (pred * target[idx]).sum(-1)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step} cosine-loss {loss.item():.3f}")

with torch.no_grad():
    pred_all = m(ids[:500], mask[:500])
    sim = pred_all @ target[:500].t()
    r1 = (sim.argmax(1) == torch.arange(500)).float().mean().item()*100
print(f"想象→真实图片 检索 R@1: {r1:.0f}%（500 选 1，随机 0.2%）")
torch.save({"model": m.state_dict(), "content": content}, ROOT/"ckpt/imagine.pt")

# 用想象检验组合
def imagine_score(words):
    ws = [c2i[w] for w in words if w in c2i][:10]
    if not ws: return -1.0
    ids = torch.tensor([ws + [0]*(10-len(ws))]); msk = torch.tensor([[1]*len(ws) + [0]*(10-len(ws))]).float()
    with torch.no_grad():
        p = m(ids, msk)
        sim = (p @ target.t()).max().item()      # 和最近真实图片的余弦
    return sim

for w in [["beach","dog"], ["train","cat","dog"], ["river","dog"], ["beach","train","dog"]]:
    print(f"  组合 {w} → 想象相似度 {imagine_score(w):.3f}")
