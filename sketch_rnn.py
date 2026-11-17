#!/usr/bin/env python3
"""Sketch-RNN：QuickDraw 上被验证过的画图模型（LSTM + 混合密度网络，非 Transformer）。

输入：每一步的位移 (dx, dy) + 笔状态；输出：下一步的位移分布 + 抬笔概率。
这是 Google 2017 年那套思路，专门为简笔画设计。
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, math
from pathlib import Path
ROOT = Path("/srv/brainai")
M = 20          # 高斯分量数
H = 384         # 隐藏层
MAXLEN = 90

d = np.load(ROOT/"data/quickdraw.npz", allow_pickle=True)
seqs, words = list(d["seqs"]), list(d["words"])
cats = sorted(set(words)); c2i = {c:i for i,c in enumerate(cats)}
print("样本", len(seqs), cats)

def to_strokes(seq, maxlen=MAXLEN):
    """QuickDraw 绝对坐标 → 归一化的 (dx, dy, pen) 序列。"""
    pts = []
    for si, s in enumerate(seq):
        for pi, p in enumerate(s):
            pts.append((p[0], p[1], 1.0 if si == 0 and pi == 0 else 0.0))
    if len(pts) < 4: return None
    pts = pts[:maxlen]
    arr = np.zeros((len(pts), 5), np.float32)   # dx, dy, p1(落笔), p2(抬笔), p3(结束)
    for i, (x, y, _) in enumerate(pts):
        if i == 0: dx = dy = 0.0
        else: dx, dy = x-pts[i-1][0], y-pts[i-1][1]
        arr[i, 0], arr[i, 1] = dx*10, dy*10
        # 抬笔：该点是某个笔画的第一点
        is_start = any(abs(pts[i][0]-s[0][0])<1e-6 and abs(pts[i][1]-s[0][1])<1e-6 for s in seq if len(s)>0)
        arr[i, 2] = 0.0 if (i > 0 and is_start) else 1.0
        arr[i, 3] = 1.0 if (i > 0 and is_start) else 0.0
        arr[i, 4] = 0.0
    arr[-1, 4] = 1.0
    return arr

data = []
for i, s in enumerate(seqs):
    a = to_strokes(s)
    if a is not None: data.append((i, a))
print("有效", len(data))

class MDN(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.lstm = nn.LSTM(5 + 64, H, num_layers=2, batch_first=True)
        self.cls = nn.Embedding(ncls, 64)
        self.z = nn.Linear(H, M*6)      # pi, mu1, mu2, sigma1, sigma2, rho
        self.pen = nn.Linear(H, 3)
    def forward(self, x, c):
        cc = self.cls(c).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.lstm(torch.cat([x, cc], -1))
        z = self.z(h); pen = self.pen(h)
        pi, mu1, mu2, s1, s2, rho = z.split(M, -1)
        pi = F.log_softmax(pi, -1)
        s1 = torch.exp(s1); s2 = torch.exp(s2); rho = torch.tanh(rho)
        return pi, mu1, mu2, s1, s2, rho, pen

m = MDN(len(cats)); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

def batch_of(ids):
    B = len(ids); L = MAXLEN
    X = torch.zeros(B, L, 5); C = torch.zeros(B, dtype=torch.long)
    for k, j in enumerate(ids):
        a = data[j][1]
        X[k, :len(a)] = torch.from_numpy(a)
        C[k] = c2i[words[data[j][0]]]
    return X, C

def mdn_loss(pi, mu1, mu2, s1, s2, rho, pen_logits, X):
    """pi/mu/sigma: (B,L,M) → 预测第 t+1 步的位移；目标是 X[:,1:]."""
    pi, mu1, mu2, s1, s2, rho, pen_logits = [t[:, :-1] for t in (pi, mu1, mu2, s1, s2, rho, pen_logits)]
    dx = X[:, 1:, 0].unsqueeze(-1)          # (B,L-1,1)
    dy = X[:, 1:, 1].unsqueeze(-1)
    pen_t = X[:, 1:, 2:].argmax(-1)         # (B,L-1)
    n1 = (dx - mu1) / s1
    n2 = (dy - mu2) / s2
    z = n1**2 + n2**2 - 2*rho*n1*n2
    G = -z/(2*(1-rho**2)) - torch.log(s1*s2*torch.sqrt(1-rho**2)) - math.log(2*math.pi)
    l_mdn = -torch.logsumexp(pi + G, -1).mean()
    l_pen = F.cross_entropy(pen_logits.reshape(-1,3), pen_t.reshape(-1))
    return l_mdn + l_pen, l_mdn, l_pen

for step in range(1, 8001):
    ids = random.sample(range(len(data)), 32)
    X, C = batch_of(ids)
    pi, mu1, mu2, s1, s2, rho, pen = m(X, C)
    loss, l1, l2 = mdn_loss(pi, mu1, mu2, s1, s2, rho, pen, X)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.3f} (mdn {l1.item():.3f} pen {l2.item():.3f})", flush=True)
        torch.save({"model": m.state_dict(), "cats": cats}, ROOT/"ckpt/sketchrnn.pt")
