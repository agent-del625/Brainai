#!/usr/bin/env python3
"""Turtle v3：条件改成「目标线稿的粗形状」，并用 IoU 奖励做拒绝采样微调。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, random
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64

d = np.load(ROOT/"data/strokes.npz", allow_pickle=True)
seqs = d["seqs"]
print("序列", len(seqs))

def render(strokes, size=S, blur=False):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts = [((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts, pts[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
    if blur:
        img = cv2.GaussianBlur(img,(9,9),0); img /= (img.max()+1e-6)
    return img

def coarse(strokes):
    """目标的粗形状：8×8 的密度图（这就是"画画指令"）。"""
    img = render(strokes, 64)
    small = cv2.resize(img, (8,8), interpolation=cv2.INTER_AREA)
    return small.reshape(-1)

COND = [torch.tensor(coarse(s), dtype=torch.float32) for s in seqs]

class Turtle(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,192), nn.GELU())
        self.cond = nn.Sequential(nn.Linear(64,128), nn.GELU(), nn.Linear(128,128), nn.GELU())
        self.step = nn.Embedding(64,32)
        self.out = nn.Sequential(nn.Linear(192+128+2+32,256), nn.GELU(), nn.Linear(256,2))
    def forward(self, state, cond, prev, t):
        st = self.step(torch.full((state.size(0),), min(int(t),63), dtype=torch.long))
        return torch.tanh(self.out(torch.cat([self.enc(state), self.cond(cond), prev, st], -1)))

m = Turtle(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)

def rollout(j, steps=16, noise=0.0):
    strokes = seqs[j]; cond = COND[j][None]
    drawn=[[]]; prev=torch.zeros(1,2); pts=[]
    with torch.no_grad():
        for t in range(steps):
            c = render(drawn); cov = render(drawn, blur=True)
            state = torch.from_numpy(np.stack([c,cov]))[None].float()
            p = m(state, cond, prev, t)
            if noise: p = p + noise*torch.randn_like(p)
            p = p.clamp(-1,1)
            drawn[-1].append(p[0].numpy()); pts.append(p[0].numpy())
            prev = p
    return render(drawn, 64), pts

def iou(a,b,thr=0.3):
    A=a>thr; B=b>thr
    return float((A&B).sum()/max((A|B).sum(),1))

# 监督预训练
for step in range(1, 1201):
    j = random.randrange(len(seqs)); strokes = seqs[j]; cond = COND[j][None]
    drawn=[[]]; prev=torch.zeros(1,2); t=0; loss=0.0; n=0
    for s in strokes:
        for p in s:
            c = render(drawn); cov = render(drawn, blur=True)
            state = torch.from_numpy(np.stack([c,cov]))[None].float()
            pred = m(state, cond, prev, t)
            tgt = torch.tensor([[p[0],p[1]]], dtype=torch.float32)
            loss = loss + F.smooth_l1_loss(pred, tgt); n+=1
            prev=tgt; t+=1; drawn[-1].append(p)
    loss = loss/max(n,1)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0: print(f"[SFT] step {step} loss {loss.item():.4f}", flush=True)

def evaluate(k=16):
    sc=[]
    for _ in range(k):
        j = random.randrange(len(seqs))
        mine,_ = rollout(j); tgt = render(seqs[j], 64)
        sc.append(iou(mine,tgt))
    return float(np.mean(sc))

base = evaluate()
print(f"监督后 IoU: {base*100:.1f}%")

# 拒绝采样微调：采样多条，留 IoU 最高的
for rnd in range(3):
    improved = []
    for _ in range(40):
        j = random.randrange(len(seqs))
        best = None
        for _ in range(6):
            mine, pts = rollout(j, noise=0.15)
            sc = iou(mine, render(seqs[j],64))
            if best is None or sc > best[0]: best = (sc, pts)
        improved.append((j, best[1], best[0]))
    # 用最好的轨迹做监督
    for j, pts, sc in improved:
        if sc < 0.02: continue
        cond = COND[j][None]; drawn=[[]]; prev=torch.zeros(1,2)
        lg = 0.0
        for t,p in enumerate(pts):
            c = render(drawn); cov = render(drawn, blur=True)
            state = torch.from_numpy(np.stack([c,cov]))[None].float()
            pred = m(state, cond, prev, t)
            lg = lg + F.smooth_l1_loss(pred, torch.tensor([p], dtype=torch.float32))
            drawn[-1].append(p); prev = torch.tensor([p], dtype=torch.float32)
        opt.zero_grad(); (lg/len(pts)).backward(); opt.step()
    print(f"第 {rnd+1} 轮 RFT 后 IoU: {evaluate()*100:.1f}%", flush=True)

OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for k in range(3):
    j = random.randrange(len(seqs))
    mine,_ = rollout(j); tgt = render(seqs[j],64)
    tiles.append(np.hstack([cv2.cvtColor((255*(1-mine)).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor((255*(1-tgt)).astype(np.uint8), cv2.COLOR_GRAY2BGR)]))
cv2.imwrite(str(OUT/"compare3.png"), np.vstack(tiles))
print("图:", OUT/"compare3.png")
