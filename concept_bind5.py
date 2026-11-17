#!/usr/bin/env python3
"""18 概念三元绑定（高清梅尔 80×128）。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = [str(c) for c in ep["cats"]], ep["mels"]
SOUND = {"beach":"sea_waves","ocean":"sea_waves","river":"water_drops","waterfall":"pouring_water",
         "forest":"crickets","crickets":"crickets","insects":"crickets","chirping_birds":"chirping_birds"}
names, protos, mels, used = [], [], [], set()
for c, f in zip([str(x) for x in cp["cats"]], cp["feats"]):
    snd = SOUND.get(c, c)
    if snd in ecats and snd not in used:      # 一个声音只配一个概念，避免音频重复
        used.add(snd); names.append(c); protos.append(f); mels.append(emels[ecats.index(snd)])
print(f"概念 {len(names)}: {names}")
proto = torch.tensor(np.stack(protos)).float(); mel = torch.tensor(np.stack(mels)).float()
print("mel", mel.shape)

class Tri(nn.Module):
    def __init__(self, d=128, n=0):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,d))
        self.aud = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*10*16, d))
        self.txt = nn.Embedding(n, d)
    def forward(self,i,a,t):
        return (F.normalize(self.img(i),-1), F.normalize(self.aud(a),-1), F.normalize(self.txt(t),-1))

m = Tri(n=len(names)); opt = torch.optim.AdamW(m.parameters(), lr=8e-3)
idx = torch.arange(len(names))
nce = lambda x,y,tau=0.1: (F.cross_entropy(x@y.t()/tau, idx) + F.cross_entropy((x@y.t()/tau).t(), idx))/2
for step in range(1, 1001):
    zi, za, zt = m(proto, mel, idx)
    loss = nce(zi,zt) + nce(za,zt) + nce(zi,za)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0: print(f"step {step} loss {loss.item():.2f}")

with torch.no_grad():
    zi, za, zt = m(proto, mel, idx)
    def acc(x,y):
        r=(x@y.t()).argsort(1,descending=True); t=torch.arange(len(names)).unsqueeze(1)
        return ((r[:,:1]==t).any(1)).float().mean().item()*100
    print(f"图→词 {acc(zi,zt):.0f}%  声→词 {acc(za,zt):.0f}%  图→声 {acc(zi,za):.0f}%  声→图 {acc(za,zi):.0f}%  随机 {100/len(names):.0f}%")
torch.save({"model": m.state_dict(), "names": names}, ROOT/"ckpt/tri18.pt")
