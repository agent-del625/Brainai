#!/usr/bin/env python3
"""Turtle 在 QuickDraw 上的加强版：20000 条、类别嵌入 256、渲染比对损失、64 步生成。"""
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
        self.cls = nn.Sequential(nn.Embedding(ncls, 256), nn.GELU(), nn.Linear(256,128), nn.GELU())
        self.step = nn.Embedding(100,32)
        self.pt = nn.Sequential(nn.Linear(192+128+2+32,256), nn.GELU(), nn.Linear(256,2))
        self.lift = nn.Sequential(nn.Linear(192+128+2+32+1,128), nn.GELU(), nn.Linear(128,1))
        # 渲染比对：把画布编码成 96 维，和目标的 96 维比
        self.rend = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(48*8*8,96), nn.GELU())
    def forward(self, state, cls, prev, t, pd):
        st = self.step(t.clamp(max=99))
        x = torch.cat([self.enc(state), self.cls(cls), prev, st], -1)
        return torch.tanh(self.pt(x)), self.lift(torch.cat([x, pd], -1))
    def embed_render(self, canvas):
        return self.rend(canvas)

m = Turtle(len(cats)); opt = torch.optim.AdamW(m.parameters(), lr=8e-4)

def build(j):
    strokes = seqs[j]; canvas = np.zeros((S,S), np.float32)
    states, prevs, pts, lifts, steps = [], [], [], [], []
    prev = np.zeros(2, np.float32); t=0
    for s in strokes:
        for pi,p in enumerate(s):
            cov = cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            st = np.stack([canvas, cov, (canvas.max()>0)*np.ones((S,S))])
            states.append(st/(st.max(axis=(1,2),keepdims=True)+1e-6)); prevs.append(prev.copy())
            pts.append(p); lifts.append(1.0 if pi==0 else 0.0); steps.append(t)
            if pi>0:
                cv2.line(canvas,(int((prev[0]+1)*S/2),int((prev[1]+1)*S/2)),
                                (int((p[0]+1)*S/2),int((p[1]+1)*S/2)),1.0,1)
            prev=p; t+=1
    return (torch.from_numpy(np.stack(states)).float(), torch.from_numpy(np.stack(prevs)).float(),
            torch.from_numpy(np.stack(pts)).float(), torch.from_numpy(np.array(lifts)).float(),
            torch.arange(len(steps)), render(strokes))

def rollout(cls_id, steps=64):
    canvas=np.zeros((S,S),np.float32); prev=np.zeros(2,np.float32); first=True
    with torch.no_grad():
        for t in range(steps):
            cov=cv2.resize(cv2.resize(canvas,(16,16),interpolation=cv2.INTER_AREA),(S,S))
            st=np.stack([canvas,cov,(canvas.max()>0)*np.ones((S,S))]); st=st/(st.max(axis=(1,2),keepdims=True)+1e-6)
            p,lift=m(torch.from_numpy(st)[None].float(), torch.tensor([cls_id]),
                     torch.from_numpy(prev)[None], torch.tensor([t]), torch.tensor([[0.0]]))
            p=p.clamp(-1,1)[0].numpy()
            if not first and lift.item()<=0:
                cv2.line(canvas,(int((prev[0]+1)*S/2),int((prev[1]+1)*S/2)),
                                (int((p[0]+1)*S/2),int((p[1]+1)*S/2)),1.0,1)
            prev=p; first=False
    return canvas

for step in range(1, 501):
    j = random.randrange(len(seqs))
    st, pv, pt, lf, ts, tgt_img = build(j)
    cls = torch.full((st.size(0),), c2i[words[j]])
    pred, lift = m(st, cls, pv, ts, lf.unsqueeze(-1))
    loss = F.smooth_l1_loss(pred, pt) + 0.3*F.binary_cross_entropy_with_logits(
        lift.squeeze(-1), lf, pos_weight=torch.tensor(4.0))
    # 渲染比对：每 10 步做一次（rollout 成本高）
    if step % 10 == 0:
        cid = c2i[words[j]]
        mine = rollout(cid, 48)
        e_m = m.embed_render(torch.from_numpy(mine)[None,None].float())
        e_t = m.embed_render(torch.from_numpy(tgt_img)[None,None].float())
        loss = loss + 0.3*(1 - F.cosine_similarity(e_m, e_t, dim=-1)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step % 25 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)
        torch.save({"model": m.state_dict(), "cats": cats}, ROOT/"ckpt/turtle_qd2.pt")

OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for c in cats:
    im = rollout(c2i[c], 64)
    cv2.imwrite(str(OUT/f"qd2_{c}.png"), (255*(1-im)).astype(np.uint8))
    tiles.append((255*(1-im)).astype(np.uint8))
cv2.imwrite(str(OUT/"quickdraw2.png"), np.hstack(tiles))
print("图:", OUT/"quickdraw2.png")
