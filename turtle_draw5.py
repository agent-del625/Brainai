#!/usr/bin/env python3
"""Turtle v5：画布保持 64×64，优化靠"增量绘制 + 状态缓存"，不降分辨率。"""
import glob, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64

seqs = []
for f in sorted(glob.glob(str(ROOT/"data/strokes*.npz"))):
    z = np.load(f, allow_pickle=True); seqs.extend(list(z["seqs"]))
print("序列", len(seqs))

def draw_line(canvas, a, b):
    """在已有画布上只画新增的一条线段（增量，不重绘）。"""
    p1 = (int((a[0]+1)*S/2-0.5), int((a[1]+1)*S/2-0.5))
    p2 = (int((b[0]+1)*S/2-0.5), int((b[1]+1)*S/2-0.5))
    cv2.line(canvas, p1, p2, 1.0, 1)

def coverage(canvas):
    """覆盖度：先降到 16×16 再放大（比高斯模糊快得多，但画布分辨率不变）。"""
    small = cv2.resize(canvas, (16,16), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (S,S), interpolation=cv2.INTER_LINEAR)

def target_map(strokes):
    small = cv2.resize(cv2.resize(_render_all(strokes), (8,8), interpolation=cv2.INTER_AREA), (S,S))
    return small

def _render_all(strokes, size=S):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts=[((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts,pts[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
    return img

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

def steps_of(j):
    """把一个样本展开成 (点, 是否抬笔) 序列。"""
    out = []
    for si, s in enumerate(seqs[j]):
        for pi, p in enumerate(s):
            out.append((p, 1.0 if pi == 0 else 0.0))
    return out

for step in range(1, 1601):
    j = random.randrange(len(seqs)); strokes = seqs[j]; tgt = target_map(strokes)
    canvas = np.zeros((S,S), np.float32); prev_pt = None
    seq = steps_of(j); loss = 0.0; n = 0
    for t, (p, pd) in enumerate(seq):
        cov = coverage(canvas); diff = np.clip(tgt - canvas, 0, 1)
        state = torch.from_numpy(np.stack([canvas, cov, tgt, diff])
                                 / (np.stack([canvas, cov, tgt, diff]).max(axis=(1,2), keepdims=True)+1e-6))[None].float()
        prev = torch.zeros(1,2) if prev_pt is None else torch.tensor([prev_pt], dtype=torch.float32)
        pred, lift = m(state, prev, t, torch.tensor([[pd]]))
        tp = torch.tensor([p], dtype=torch.float32)
        loss = loss + F.smooth_l1_loss(pred, tp) + \
               0.3*F.binary_cross_entropy_with_logits(lift, torch.tensor([[pd]]), pos_weight=torch.tensor(4.0))
        n += 1
        if pd > 0.5: prev_pt = None
        else:
            draw_line(canvas, prev_pt, p)
        prev_pt = p
    loss = loss/max(n,1)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if step % 200 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)

def rollout(j, steps=24):
    strokes = seqs[j]; tgt = target_map(strokes)
    canvas = np.zeros((S,S), np.float32); prev_pt=None
    with torch.no_grad():
        for t in range(steps):
            cov = coverage(canvas); diff = np.clip(tgt-canvas,0,1)
            st = np.stack([canvas,cov,tgt,diff]); st = st/(st.max(axis=(1,2),keepdims=True)+1e-6)
            prev = torch.zeros(1,2) if prev_pt is None else torch.tensor([prev_pt],dtype=torch.float32)
            p, lift = m(torch.from_numpy(st)[None].float(), prev, t, torch.tensor([[0.0]]))
            p = p.clamp(-1,1)[0].numpy()
            if prev_pt is not None and lift.item() <= 0: draw_line(canvas, prev_pt, p)
            prev_pt = p
    return canvas

def iou(a,b,thr=0.3):
    A=a>thr; B=b>thr; return float((A&B).sum()/max((A|B).sum(),1))
sc=[iou(rollout(j), _render_all(seqs[j])) for j in random.sample(range(len(seqs)), 12)]
print(f"v5 IoU: {np.mean(sc)*100:.1f}%")
OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for j in random.sample(range(len(seqs)),3):
    mine=rollout(j); tgt=_render_all(seqs[j])
    tiles.append(np.hstack([cv2.cvtColor((255*(1-mine)).astype(np.uint8),cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor((255*(1-tgt)).astype(np.uint8),cv2.COLOR_GRAY2BGR)]))
cv2.imwrite(str(OUT/"compare5.png"), np.vstack(tiles))
print("图:", OUT/"compare5.png")
