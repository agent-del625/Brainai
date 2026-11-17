#!/usr/bin/env python3
"""Turtle v4：抬笔/落笔 + 空间化条件（目标 + 差值），支持多笔画。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, random
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64

import glob
seqs = []
for f in sorted(glob.glob(str(ROOT/"data/strokes*.npz"))):
    z = np.load(f, allow_pickle=True)
    seqs.extend(list(z["seqs"]))
print("序列", len(seqs))

def render(strokes, size=S, blur=False):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts=[((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts,pts[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
    if blur:
        img = cv2.GaussianBlur(img,(9,9),0); img/= (img.max()+1e-6)
    return img

def target_map(strokes):
    """目标粗形状（8×8 → 64×64 上采样），作为空间条件。"""
    small = cv2.resize(render(strokes,64), (8,8), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (S,S), interpolation=cv2.INTER_LINEAR)

class Turtle(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4,24,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,192), nn.GELU())
        self.step = nn.Embedding(80,32)
        self.pt = nn.Sequential(nn.Linear(192+2+32,256), nn.GELU(), nn.Linear(256,2))
        self.lift = nn.Sequential(nn.Linear(192+2+32+1,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, state, prev, t, pen_down):
        st = self.step(torch.full((state.size(0),), min(int(t),79), dtype=torch.long))
        x = torch.cat([self.enc(state), prev, st], -1)
        return torch.tanh(self.pt(x)), self.lift(torch.cat([x, pen_down], -1))

m = Turtle(); opt = torch.optim.AdamW(m.parameters(), lr=5e-4)

def make_state(drawn, tgt, pen_down):
    cur = render(drawn); cov = render(drawn, blur=True); diff = np.clip(tgt - cur, 0, 1)
    st = np.stack([cur, cov, tgt, diff]).astype(np.float32)
    st = st / (st.max(axis=(1,2), keepdims=True) + 1e-6)      # 每通道归一化
    return st[None]

for step in range(1, 1601):
    j = random.randrange(len(seqs)); strokes = seqs[j]; tgt = target_map(strokes)
    drawn=[[]]; prev=torch.zeros(1,2); t=0; loss=0.0; n=0
    for si, s in enumerate(strokes):
        for pi, p in enumerate(s):
            pd = 0.0 if pi > 0 else 1.0        # 每条笔画第一个点之前抬笔
            state = torch.from_numpy(make_state(drawn, tgt, pd))
            pred, lift = m(state, prev, t, torch.tensor([[pd]]))
            tgtp = torch.tensor([[p[0],p[1]]], dtype=torch.float32)
            loss = loss + F.smooth_l1_loss(pred, tgtp) + 0.3*F.binary_cross_entropy_with_logits(
                lift, torch.tensor([[pd]]), pos_weight=torch.tensor(4.0))
            n += 1; prev = tgtp; t += 1
            if pd > 0.5: drawn.append([p])
            else: drawn[-1].append(p)
    loss = loss/max(n,1)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if step % 400 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

def rollout(j, steps=24):
    strokes = seqs[j]; tgt = target_map(strokes)
    drawn=[[]]; prev=torch.zeros(1,2)
    with torch.no_grad():
        for t in range(steps):
            state = torch.from_numpy(make_state(drawn, tgt, 0.0))
            p, lift = m(state, prev, t, torch.tensor([[0.0]]))
            p = p.clamp(-1,1)
            if lift.item() > 0 and len(drawn[-1]) > 0: drawn.append([])
            drawn[-1].append(p[0].numpy()); prev = p
    return render(drawn, 64)

def iou(a,b,thr=0.3):
    A=a>thr; B=b>thr; return float((A&B).sum()/max((A|B).sum(),1))
sc=[iou(rollout(j), render(seqs[j],64)) for j in random.sample(range(len(seqs)), 12)]
print(f"多笔画 IoU: {np.mean(sc)*100:.1f}%")
OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for j in random.sample(range(len(seqs)), 3):
    mine = rollout(j); tgt = render(seqs[j],64)
    tiles.append(np.hstack([cv2.cvtColor((255*(1-mine)).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor((255*(1-tgt)).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
cv2.imwrite(str(OUT/"compare4.png"), np.vstack(tiles))
print("图:", OUT/"compare4.png")
