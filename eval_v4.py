#!/usr/bin/env python3
"""v4 评测：图文检索 R@1/R@5、生成样句、自评头与真实误差的相关系数。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_v4 import BrainV4

ROOT = Path("/srv/brainai")
z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
caps = [str(c) for c in z["captions"]]
st = torch.load(ROOT / "ckpt/v4.pt", map_location="cpu", weights_only=False)
vocab = st["vocab_map"] if "vocab_map" in st else None

counter: dict[str, int] = {}
for c in caps:
    for ch in c:
        counter[ch] = counter.get(ch, 0) + 1
if vocab is None:
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for ch, n in sorted(counter.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            vocab[ch] = len(vocab)

L = 24


def encode(text: str) -> torch.Tensor:
    ids = [2] + [vocab.get(ch, 1) for ch in text][: L - 1]
    ids += [0] * (L - len(ids))
    return torch.tensor(ids, dtype=torch.long)


model = BrainV4(vocab=len(vocab), rounds=12)
model.load_state_dict(st["model"])
model.eval()

n = 128
idx = torch.arange(n)
tok = torch.stack([encode(caps[i]) for i in idx])
with torch.no_grad():
    out = model(clip=feats[idx], t_in=tok)
z_img = torch.nn.functional.normalize(out["zc"], dim=-1)
z_txt = torch.nn.functional.normalize(out["zt"], dim=-1)
zc_sim = z_img @ z_txt.t()
ranks = zc_sim.argsort(1, descending=True)
tgt = torch.arange(n).unsqueeze(1)
r1 = ((ranks[:, :1] == tgt).any(1)).float().mean().item() * 100
r5 = ((ranks[:, :5] == tgt).any(1)).float().mean().item() * 100
print(f"[CLIP 对齐] 图→文 R@1 {r1:.1f}%  R@5 {r5:.1f}% （随机 {100/n:.1f}%）")
sim = out["z"] @ z_txt.t()
rk = sim.argsort(1, descending=True)
r1w = ((rk[:, :1] == tgt).any(1)).float().mean().item() * 100
print(f"[工作空间] 图→文 R@1 {r1w:.1f}%")

# 生成样句
inv = {v: k for k, v in vocab.items()}
g = model.text_head.generate(out["z"][:5], start_id=2, length=L)
for i in range(5):
    txt = "".join(inv.get(int(c), "") for c in g[i]).replace("<eos>", "")
    print(f"  生成: {txt[:48]}")
    print(f"  参考: {caps[i][:48]}")

# 自评头相关性
with torch.no_grad():
    lg, tg = out["chars"]
    per = F.cross_entropy(lg.reshape(-1, len(vocab)), tg.reshape(-1),
                          ignore_index=0, reduction="none").view(n, -1).mean(1)
    pred = out["err_pred"]
    a = (pred - pred.mean()) / (pred.std() + 1e-6)
    b = (per - per.mean()) / (per.std() + 1e-6)
    print(f"自评相关性 r={float((a*b).mean()):.3f}")
