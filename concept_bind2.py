#!/usr/bin/env python3
"""三元绑定（真实声音事件）：图片 ↔ 词 ↔ 环境声。"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
feats, caps = z["feats"], [str(c) for c in z["captions"]]
e = np.load(ROOT / "data/esc50_proto.npz", allow_pickle=True)
cats, mels = list(e["cats"]), e["mels"]

# 图里能找到对应词的声音类别
MAP = {"dog": "dog", "cat": "cat", "bird": "chirping_birds", "water": "sea_waves",
       "rain": "rain", "fire": "crackling_fire", "car": "engine", "train": "train",
       "door": "door_wood_knock", "clock": "clock_alarm", "wind": "wind",
       "thunder": "thunderstorm", "footsteps": "footsteps", "laugh": "laughing",
       "baby": "crying_baby", "helicopter": "helicopter", "siren": "siren"}
concepts, imgs, auds = [], [], []
for word, cat in MAP.items():
    idx = [i for i, c in enumerate(caps) if word in re.findall(r"[a-z']+", c.lower())]
    if len(idx) < 5 or cat not in cats:
        continue
    concepts.append(word)
    imgs.append(feats[idx].mean(0))
    auds.append(mels[cats.index(cat)])
print(f"概念 {len(concepts)} 个: {concepts}")
proto, mel = torch.tensor(np.stack(imgs)).float(), torch.tensor(np.stack(auds)).float()

class Tri(nn.Module):
    def __init__(self, d=128, n=0):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, d))
        self.aud = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(32 * 10 * 16, d))
        self.txt = nn.Embedding(n, d)
    def forward(self, i, a, t):
        return (F.normalize(self.img(i), -1), F.normalize(self.aud(a), -1), F.normalize(self.txt(t), -1))

m = Tri(n=len(concepts)); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
idx = torch.arange(len(concepts))
def nce(x, y, temp=0.1):
    lg = x @ y.t() / temp
    return (F.cross_entropy(lg, idx) + F.cross_entropy(lg.t(), idx)) / 2

for step in range(1, 601):
    zi, za, zt = m(proto, mel, idx)
    loss = nce(zi, zt) + nce(za, zt) + nce(zi, za)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0: print(f"step {step} loss {loss.item():.3f}")

with torch.no_grad():
    zi, za, zt = m(proto, mel, idx)
    def acc(x, y):
        r = (x @ y.t()).argsort(1, descending=True)
        t = torch.arange(len(concepts)).unsqueeze(1)
        return ((r[:, :1] == t).any(1)).float().mean().item() * 100
    print(f"\n图→词 {acc(zi,zt):.0f}%   声→词 {acc(za,zt):.0f}%   图→声 {acc(zi,za):.0f}%   声→图 {acc(za,zi):.0f}%   随机 {100/len(concepts):.0f}%")
