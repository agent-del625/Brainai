#!/usr/bin/env python3
"""合并概念集：ImageNet 物体 + Places365 场景，统一配 ESC-50 声音。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
ip = np.load(ROOT/"data/imagenet_proto.npz", allow_pickle=True)
sp = np.load(ROOT/"data/scene_proto.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
cats, mels = list(ep["cats"]), ep["mels"]
SOUND = {"dog":"dog","cat":"cat","bird":"chirping_birds","train":"train","clock":"clock_alarm",
         "keyboard":"keyboard_typing","plane":"airplane","bike":"chainsaw",
         "beach":"sea_waves","forest":"crickets","ocean":"sea_waves",
         "river":"water_drops","waterfall":"pouring_water"}
names, protos, sounds = [], [], []
for src in (ip, sp):
    for c, f in zip(list(src["cats"]), src["feats"]):
        s = SOUND.get(c)
        if s and s in cats:
            names.append(c); protos.append(f); sounds.append(mels[cats.index(s)])
print(f"概念 {len(names)}: {names}")
proto = torch.tensor(np.stack(protos)).float(); mel = torch.tensor(np.stack(sounds)).float()

class Tri(nn.Module):
    def __init__(self, d=128, n=0):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,d))
        self.aud = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(32*10*16, d))
        self.txt = nn.Embedding(n, d)
    def forward(self, i, a, t):
        return (F.normalize(self.img(i),-1), F.normalize(self.aud(a),-1), F.normalize(self.txt(t),-1))

m = Tri(n=len(names)); opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
idx = torch.arange(len(names))
nce = lambda x,y,temp=0.1: (F.cross_entropy(x@y.t()/temp, idx) + F.cross_entropy((x@y.t()/temp).t(), idx))/2
for step in range(1, 801):
    zi, za, zt = m(proto, mel, idx)
    loss = nce(zi,zt) + nce(za,zt) + nce(zi,za)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0: print(f"step {step} loss {loss.item():.2f}")

with torch.no_grad():
    zi, za, zt = m(proto, mel, idx)
    def acc(x,y):
        r=(x@y.t()).argsort(1,descending=True); t=torch.arange(len(names)).unsqueeze(1)
        return ((r[:,:1]==t).any(1)).float().mean().item()*100
    print(f"图→词 {acc(zi,zt):.0f}%  声→词 {acc(za,zt):.0f}%  图→声 {acc(zi,za):.0f}%  声→图 {acc(za,zi):.0f}%  随机 {100/len(names):.0f}%")
