#!/usr/bin/env python3
"""给模型一只 Turtle 手：每一步看自己画到哪了，再决定下一笔怎么画。

状态 = 当前画布（真实渲染的图）+ 意图向量（概念）
动作 = 下一个点 (x, y)（连续笔画）
训练 = 用参考线稿矢量化出的笔画序列做教师强制
"""
import math, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
import onnxruntime as ort

ROOT = Path("/srv/brainai"); S = 64
sk = ort.InferenceSession(str(ROOT/"data/downloads/sketch.onnx"), providers=["CPUExecutionProvider"])
si, so = sk.get_inputs()[0].name, sk.get_outputs()[0].name

def to_lines(img_bgr, size=256):
    x = cv2.resize(img_bgr, (256,256)).astype(np.float32)/255.0
    x = (x-0.5)/0.5
    out = sk.run([so], {si: x.transpose(2,0,1)[None]})[0][0,0]
    out = (out-out.min())/(out.max()-out.min()+1e-6)
    if out.mean() > 0.5: out = 1-out
    return (out*255).clip(0,255).astype(np.uint8)

def vectorize(line_img, max_strokes=6, max_points=12):
    """线稿 → 笔画序列（每条笔画若干点），坐标归一到 [-1,1]。"""
    bw = (line_img > 90).astype(np.uint8)*255
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: -cv2.arcLength(c, False))[:max_strokes]
    strokes = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 3.0, False).reshape(-1, 2)
        if len(c) < 3: continue
        idx = np.linspace(0, len(c)-1, min(max_points, len(c))).astype(int)
        pts = c[idx].astype(np.float32)
        pts[:,0] = pts[:,0]/line_img.shape[1]*2-1
        pts[:,1] = pts[:,1]/line_img.shape[0]*2-1
        strokes.append(pts)
    return strokes

def render(strokes, size=S):
    """把笔画画到画布上（模拟 turtle 的笔迹）。"""
    img = np.full((size,size), 255, np.uint8)
    for s in strokes:
        pts = [((p[0]+1)*size/2, (p[1]+1)*size/2) for p in s]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), 0, 1)
    return img

# 数据：23 个概念 → 线稿 → 笔画
from collections import defaultdict
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
data = []
for c in cfeat:
    gal = sorted((ROOT/"data/gallery").rglob(f"{c}_*.jpg"))[:2]
    for p in gal:
        strokes = vectorize(to_lines(cv2.imread(str(p))))
        if strokes: data.append((c, strokes))
print("样本", len(data), "个（概念×图）")

class Drawer(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.canvas = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8, 128))
        self.intent = nn.Linear(512, 128)
        self.ctx = nn.GRU(2, 64, batch_first=True)
        self.step = nn.Embedding(32, 32)
        self.out = nn.Sequential(nn.Linear(128+128+2+32, 192), nn.GELU(), nn.Linear(192, 2))

    def forward(self, canvas, intent, prev_pt, step):
        cf = self.canvas(canvas)
        it = torch.relu(self.intent(intent))
        st = self.step(torch.full((canvas.size(0),), min(int(step), 31), dtype=torch.long))
        return torch.tanh(self.out(torch.cat([cf, it, prev_pt, st], -1)))

m = Drawer(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for step in range(1, 401):
    c, strokes = random.choice(data)
    intent = cfeat[c]
    drawn = [[]]
    loss = 0.0; n = 0
    prev = torch.zeros(1, 2); tstep = 0
    for s in strokes:
        for p in s:
            canvas = torch.from_numpy(render(drawn)).float().unsqueeze(0).unsqueeze(0)/255.0
            pred = m(canvas, intent[None], prev, tstep)
            tgt = torch.tensor([[p[0], p[1]]], dtype=torch.float32)
            loss = loss + F.smooth_l1_loss(pred, tgt); n += 1
            prev = tgt; tstep += 1
            drawn[-1].append(p)
    loss = loss/max(n,1)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0: print(f"step {step} loss {loss.item():.4f}")

# 推理：按自己的想法画（只给意图，画布从空白开始）
OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles = []
with torch.no_grad():
    for c in ["dog","cat","train","bird"]:
        if c not in cfeat: continue
        drawn = [[]]; prev = torch.zeros(1,2); intent = cfeat[c][None]
        for t in range(16):
            canvas = torch.from_numpy(render(drawn)).float().unsqueeze(0).unsqueeze(0)/255.0
            p = m(canvas, intent, prev, t)[0].numpy()
            drawn[-1].append(p); prev = torch.tensor(p, dtype=torch.float32)[None]
        img = render(drawn, 160)
        cv2.imwrite(str(OUT/f"draw_{c}.png"), img)
        tiles.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
cv2.imwrite(str(OUT/"turtle.png"), np.hstack(tiles))
print("图:", OUT/"turtle.png")
