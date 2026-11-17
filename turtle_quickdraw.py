#!/usr/bin/env python3
"""Turtle 手在 QuickDraw 上训练：学人类真实简笔画（每类给一个类别向量作意图）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, random
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64
d = np.load(ROOT/"data/quickdraw.npz", allow_pickle=True)
seqs, words = list(d["seqs"]), list(d["words"])
cats = sorted(set(words)); c2i = {c:i for i,c in enumerate(cats)}
print("样本", len(seqs), "类别", cats)

def render(strokes, size=S):
    img = np.zeros((size,size), np.float32)
    for s in strokes:
        pts=[((p[0]+1)*size/2-0.5,(p[1]+1)*size/2-0.5) for p in s]
        for a,b in zip(pts,pts[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.0,1)
    return img

class Turtle(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,24,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8,192), nn.GELU())
        self.cls = nn.Embedding(ncls, 64)
        self.step = nn.Embedding(80,32)
        self.pt = nn.Sequential(nn.Linear(192+64+2+32,256), nn.GELU(), nn.Linear(256,2))
        self.lift = nn.Sequential(nn.Linear(192+64+2+32+1,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, state, cls, prev, t, pd):
        st = self.step(t.clamp(max=79))
        x = torch.cat([self.enc(state), self.cls(cls), prev, st], -1)
        return torch.tanh(self.pt(x)), self.lift(torch.cat([x, pd], -1))

m = Turtle(len(cats)); opt = torch.optim.AdamW(m.parameters(), lr=8e-4)

def build(j):
    strokes = seqs[j]; canvas = np.zeros((S,S), np.float32)
    states, prevs, pts, lifts, steps = [], [], [], [], []
    prev = np.zeros(2, np.float32); t=0
    for s in strokes:
        for pi,p in enumerate(s):
            cov = cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            st = np.stack([canvas, cov, canvas*0+ (canvas.max()>0)])   # 画布 / 覆盖 / 已开始标记
            states.append(st/(st.max(axis=(1,2),keepdims=True)+1e-6)); prevs.append(prev.copy())
            pts.append(p); lifts.append(1.0 if pi==0 else 0.0); steps.append(t)
            if pi>0:
                cv2.line(canvas,(int((prev[0]+1)*S/2),int((prev[1]+1)*S/2)),
                                (int((p[0]+1)*S/2),int((p[1]+1)*S/2)),1.0,1)
            prev=p; t+=1
    return (torch.from_numpy(np.stack(states)).float(), torch.from_numpy(np.stack(prevs)).float(),
            torch.from_numpy(np.stack(pts)).float(), torch.from_numpy(np.array(lifts)).float(),
            torch.arange(len(steps)))

for step in range(1, 801):
    j = random.randrange(len(seqs))
    st, pv, pt, lf, ts = build(j)
    cls = torch.full((st.size(0),), c2i[words[j]])
    pred, lift = m(st, cls, pv, ts, lf.unsqueeze(-1))
    loss = F.smooth_l1_loss(pred, pt) + 0.3*F.binary_cross_entropy_with_logits(
        lift.squeeze(-1), lf, pos_weight=torch.tensor(4.0))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step % 25 == 0:
        # 评测：每类画一张，看像不像该类
        print(f"step {step} loss {loss.item():.4f}", flush=True)
        torch.save({"model": m.state_dict(), "cats": cats}, ROOT/"ckpt/turtle_qd.pt")

# 每类生成一张
OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
with torch.no_grad():
    for c in cats:
        canvas=np.zeros((S,S),np.float32); prev=np.zeros(2,np.float32); first=True
        for t in range(24):
            cov=cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            st=np.stack([canvas,cov,(canvas.max()>0)*np.ones((S,S))]); st=st/(st.max(axis=(1,2),keepdims=True)+1e-6)
            p,lift=m(torch.from_numpy(st)[None].float(), torch.tensor([c2i[c]]),
                     torch.from_numpy(prev)[None], torch.tensor([t]), torch.tensor([[0.0]]))
            p=p.clamp(-1,1)[0].numpy()
            if not first and lift.item()<=0:
                cv2.line(canvas,(int((prev[0]+1)*S/2),int((prev[1]+1)*S/2)),
                                (int((p[0]+1)*S/2),int((p[1]+1)*S/2)),1.0,1)
            prev=p; first=False
        tiles.append((255*(1-canvas)).astype(np.uint8))
        cv2.imwrite(str(OUT/f"qd_{c}.png"), tiles[-1])
cv2.imwrite(str(OUT/"quickdraw_gen.png"), np.hstack(tiles))
print("图:", OUT/"quickdraw_gen.png")
