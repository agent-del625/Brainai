#!/usr/bin/env python3
"""Turtle 手 v2：300 条笔画数据 + 覆盖度状态 + IoU 奖励评估。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, random
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64

d = np.load(ROOT/"data/strokes.npz", allow_pickle=True)
seqs, idxs = d["seqs"], d["idx"]
feats = torch.from_numpy(np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)["feats"]).float()
print("训练序列", len(seqs))

def render(strokes, size=S, blur=False):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts = [((p[0]+1)*size/2-0.5, (p[1]+1)*size/2-0.5) for p in s]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), 1.0, 1)
    if blur:
        img = cv2.GaussianBlur(img, (9,9), 0)
        img = img/ (img.max()+1e-6)
    return img

class Turtle(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,192), nn.GELU())
        self.intent = nn.Sequential(nn.Linear(512,192), nn.GELU())
        self.step = nn.Embedding(64, 32)
        self.out = nn.Sequential(nn.Linear(192+192+2+32, 256), nn.GELU(), nn.Linear(256,2))
    def forward(self, state, intent, prev, t):
        st = self.step(torch.full((state.size(0),), min(int(t),63), dtype=torch.long))
        h = torch.cat([self.enc(state), self.intent(intent), prev, st], -1)
        return torch.tanh(self.out(h))

m = Turtle(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
order = list(range(len(seqs)))
for step in range(1, 1501):
    j = random.choice(order)
    strokes = seqs[j]; intent = feats[idxs[j]][None]
    drawn = [[]]; prev = torch.zeros(1,2); t = 0; loss = 0.0; n = 0
    for s in strokes:
        for p in s:
            c = render(drawn); cov = render(drawn, blur=True)
            state = torch.from_numpy(np.stack([c, cov]))[None].float()
            pred = m(state, intent, prev, t)
            tgt = torch.tensor([[p[0],p[1]]], dtype=torch.float32)
            loss = loss + F.smooth_l1_loss(pred, tgt); n += 1
            prev = tgt; t += 1; drawn[-1].append(p)
    loss = loss/max(n,1)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

# 评估：自己的画 vs 目标线稿的 IoU
def iou(a, b, thr=0.3):
    A = a > thr; B = b > thr
    return (A&B).sum()/max((A|B).sum(),1)
m.eval(); scores, tiles = [], []
with torch.no_grad():
    for k in range(8):
        j = random.randrange(len(seqs)); strokes = seqs[j]; intent = feats[idxs[j]][None]
        drawn = [[]]; prev = torch.zeros(1,2)
        for t in range(16):
            c = render(drawn); cov = render(drawn, blur=True)
            state = torch.from_numpy(np.stack([c,cov]))[None].float()
            p = m(state, intent, prev, t)[0].numpy()
            drawn[-1].append(p); prev = torch.tensor(p,dtype=torch.float32)[None]
        mine = render(drawn, 160); target = render(strokes, 160)
        scores.append(iou(mine, target))
        tiles.append(np.hstack([cv2.cvtColor((255*(1-mine)).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor((255*(1-target)).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
print(f"IoU（自己的画 vs 目标线稿）：{np.mean(scores)*100:.1f}%（8 个样本）")
OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT/"compare.png"), np.vstack(tiles[:4]))
print("图:", OUT/"compare.png")
