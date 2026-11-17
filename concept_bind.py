#!/usr/bin/env python3
""""小山"：图片 ↔ 词 ↔ 发音 三元绑定。

每个概念取三样东西：
  · 图片原型：Flickr8k 里含该词的图片 CLIP 特征均值
  · 文字：词本身（字符编码）
  · 声音：TTS 合成的发音（梅尔谱）
训练三路对比对齐，测跨模态检索。
"""
import glob, re, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from audio_feat import mel_spec

ROOT = Path("/srv/brainai")
z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
feats, caps = z["feats"], [str(c) for c in z["captions"]]
words = [p.stem for p in sorted((ROOT / "data/words").glob("*.mp3"))]

# 每个词：找含它的图片，取 CLIP 特征均值
by_word = defaultdict(list)
for i, c in enumerate(caps):
    for w in set(re.findall(r"[a-z']+", c.lower())):
        if w in words:
            by_word[w].append(i)
concepts = [w for w in words if len(by_word[w]) >= 5]
print(f"概念 {len(concepts)} 个")

proto = torch.tensor(np.stack([feats[by_word[w]].mean(0) for w in concepts])).float()

def word_audio(w):
    mp3 = ROOT / f"data/words/{w}.mp3"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16)
    if len(pcm) < 16000:
        pcm = np.pad(pcm, (0, 16000 - len(pcm)))
    seg = pcm[: int(2.5 * 16000)]          # 取前 2.5 秒
    return mel_spec(seg)

mels = torch.tensor(np.stack([word_audio(w) for w in concepts])).float()
print("图像原型", proto.shape, "音频", mels.shape)

class Tri(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, d))
        self.aud = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(32 * 10 * 16, d))
        self.txt = nn.Embedding(len(concepts), d)      # 每个概念一个可学习词向量（小而准）
    def forward(self, i, a, t):
        return (F.normalize(self.img(i), -1), F.normalize(self.aud(a), -1), F.normalize(self.txt(t), -1))

m = Tri(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
tidx = torch.arange(len(concepts))
def nce(x, y, temp=0.07):
    lg = x @ y.t() / temp
    return (F.cross_entropy(lg, tidx) + F.cross_entropy(lg.t(), tidx)) / 2

for step in range(1, 801):
    zi, za, zt = m(proto, mels, tidx)
    loss = nce(zi, zt) + nce(za, zt) + nce(zi, za)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0: print(f"step {step} loss {loss.item():.3f}")

with torch.no_grad():
    zi, za, zt = m(proto, mels, tidx)
    def acc(x, y):
        r = (x @ y.t()).argsort(1, descending=True)
        t = torch.arange(len(concepts)).unsqueeze(1)
        return ((r[:, :1] == t).any(1)).float().mean().item() * 100
    print(f"\n图→词 {acc(zi,zt):.0f}%   声→词 {acc(za,zt):.0f}%   图→声 {acc(zi,za):.0f}%   （随机 {100/len(concepts):.1f}%）")
