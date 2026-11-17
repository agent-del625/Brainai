#!/usr/bin/env python3
"""听声辨物：给一段声音 → 找到最接近的概念 → 输出概念词 + 对应图片。"""
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT = Path("/srv/brainai")
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = [str(c) for c in ep["cats"]], ep["mels"]
ck = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
names = ck["names"]

class Tri(nn.Module):
    def __init__(self, d=128, n=0):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,d))
        self.aud = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64*10*16,d))
        self.txt = nn.Embedding(n,d)
    def forward(self,i,a,t):
        return (F.normalize(self.img(i),-1), F.normalize(self.aud(a),-1), F.normalize(self.txt(t),-1))
m = Tri(n=len(names)); m.load_state_dict(ck["model"]); m.eval()

TESTS = ["dog","cat","rain","train","church_bells","cow"]
rows = []
for snd in TESTS:
    if snd not in ecats: continue
    a = torch.tensor(emels[ecats.index(snd)]).float()[None]
    with torch.no_grad():
        _, za, _ = m(torch.zeros(1,512), a, torch.zeros(1,dtype=torch.long))
        sims = (za @ m.txt.weight.t()).squeeze(0)
        best = names[int(sims.argmax())]
    print(f"声音 {snd:15s} → 概念 {best}")
    gal = sorted((ROOT/"data/gallery").rglob(f"{best}_*.jpg"))[:2]
    tiles = []
    for g in gal: tiles.append(cv2.resize(cv2.imread(str(g)), (150,150)))
    if tiles: rows.append(np.hstack(tiles))
if rows:
    sheet = np.vstack(rows)
    out = ROOT/"logs/stage3/audio_sheet.png"
    cv2.imwrite(str(out), sheet)
    print("图:", out)
