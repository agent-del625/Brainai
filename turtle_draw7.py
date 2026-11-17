#!/usr/bin/env python3
"""Turtle v7：加"覆盖度奖励"——每一步都奖励它朝"还没画的目标区域"走。"""
import glob, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64

seqs = []
for f in sorted(glob.glob(str(ROOT/"data/strokes*.npz"))):
    z = np.load(f, allow_pickle=True); seqs.extend(list(z["seqs"]))
print("序列", len(seqs))

def render_all(strokes, size=S):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts=[((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts,pts[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
    return img

def to_px(p):  # [-1,1] → 像素
    return ((p[0]+1)*S/2-0.5, (p[1]+1)*S/2-0.5)

def build(j, max_steps=None):
    strokes = seqs[j]
    tgt = cv2.resize(cv2.resize(render_all(strokes), (8,8), interpolation=cv2.INTER_AREA), (S,S))
    canvas = np.zeros((S,S), np.float32)
    states, prevs, pts, lifts, steps, dists = [], [], [], [], [], []
    prev_pt = np.zeros(2, np.float32); t = 0
    for s in strokes:
        for pi, p in enumerate(s):
            if max_steps and t >= max_steps: break
            cov = cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            diff = np.clip(tgt-canvas,0,1)
            st = np.stack([canvas,cov,tgt,diff]); st = st/(st.max(axis=(1,2),keepdims=True)+1e-6)
            # 距离图：离"还没画的目标线"多远
            need = (tgt > 0.3) & (canvas < 0.3)
            inv = (1-need).astype(np.uint8)
            d = cv2.distanceTransform(inv, cv2.DIST_L2, 3) / S
            states.append(st); prevs.append(prev_pt.copy()); pts.append(p)
            lifts.append(1.0 if pi==0 else 0.0); steps.append(t); dists.append(d)
            if pi>0:
                a=to_px(prev_pt); b=to_px(p)
                cv2.line(canvas,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
            prev_pt = p; t += 1
    return (torch.from_numpy(np.stack(states)).float(), torch.from_numpy(np.stack(prevs)).float(),
            torch.from_numpy(np.stack(pts)).float(), torch.from_numpy(np.array(lifts)).float(),
            torch.arange(len(steps)), torch.from_numpy(np.stack(dists)).float())

def sample_dist(dmap, p):
    """在距离图上双线性采样预测点处的值（可微）。"""
    xy = torch.stack([(p[:,0]+1)*S/2-0.5, (p[:,1]+1)*S/2-0.5], -1)
    grid = xy.view(-1,1,1,2)/ (S/2) - 1
    return F.grid_sample(dmap.unsqueeze(1), grid, align_corners=False).view(-1)

class Turtle(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4,24,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,192), nn.GELU())
        self.step = nn.Embedding(120,32)
        self.pt = nn.Sequential(nn.Linear(192+2+32,256), nn.GELU(), nn.Linear(256,2))
        self.lift = nn.Sequential(nn.Linear(192+2+32+1,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, state, prev, t, pen_down):
        st = self.step(t.clamp(max=119))
        x = torch.cat([self.enc(state), prev, st], -1)
        return torch.tanh(self.pt(x)), self.lift(torch.cat([x, pen_down], -1))

m = Turtle(); opt = torch.optim.AdamW(m.parameters(), lr=5e-4)

def rollout(j, steps=24):
    strokes = seqs[j]
    tgt = cv2.resize(cv2.resize(render_all(strokes), (8,8), interpolation=cv2.INTER_AREA), (S,S))
    canvas = np.zeros((S,S), np.float32); prev_pt=np.zeros(2,np.float32); first=True
    with torch.no_grad():
        for t in range(steps):
            cov = cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            diff = np.clip(tgt-canvas,0,1)
            st = np.stack([canvas,cov,tgt,diff]); st = st/(st.max(axis=(1,2),keepdims=True)+1e-6)
            p, lift = m(torch.from_numpy(st)[None].float(), torch.from_numpy(prev_pt)[None],
                        torch.tensor([t]), torch.tensor([[0.0]]))
            p = p.clamp(-1,1)[0].numpy()
            if not first and lift.item() <= 0:
                a=to_px(prev_pt); b=to_px(p)
                cv2.line(canvas,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
            prev_pt=p; first=False
    return canvas

def iou(a,b,thr=0.3):
    A=a>thr; B=b>thr; return float((A&B).sum()/max((A|B).sum(),1))

for step in range(1, 501):
    j = random.randrange(len(seqs))
    st, pv, pt, lf, ts, dm = build(j, max_steps=random.randint(20, 60))
    pred, lift = m(st, pv, ts, lf.unsqueeze(-1))
    l_pt = F.smooth_l1_loss(pred, pt)
    l_lift = 0.3*F.binary_cross_entropy_with_logits(lift.squeeze(-1), lf, pos_weight=torch.tensor(4.0))
    l_cov = sample_dist(dm, pred).mean() * 2.0          # 朝"还没画的目标"走
    loss = l_pt + l_lift + l_cov
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step % 50 == 0:
        sc=[iou(rollout(jj), render_all(seqs[jj])) for jj in random.sample(range(len(seqs)), 8)]
        print(f"step {step} loss {loss.item():.4f} (pt {l_pt.item():.3f} cov {l_cov.item():.3f}) IoU {np.mean(sc)*100:.1f}%", flush=True)
        torch.save({"model": m.state_dict()}, ROOT/"ckpt/turtle7.pt")

OUT=ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for j in random.sample(range(len(seqs)),3):
    mine=rollout(j); tgt=render_all(seqs[j])
    tiles.append(np.hstack([cv2.cvtColor((255*(1-mine)).astype(np.uint8),cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor((255*(1-tgt)).astype(np.uint8),cv2.COLOR_GRAY2BGR)]))
cv2.imwrite(str(OUT/"compare7.png"), np.vstack(tiles))
print("图:", OUT/"compare7.png")
