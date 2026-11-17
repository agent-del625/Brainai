#!/usr/bin/env python3
"""第 2 段升级：用小模型在内容词之间插入功能词（替代统计法）。"""
import re
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
cnt = Counter(w for c in caps for w in words_of(c))
content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
c2i = {w: i for i, w in enumerate(content)}
cset = set(content)

# 收集"间隙模式"
gaps = Counter()
samples = []
for c in caps:
    ws = words_of(c); idx = [i for i, w in enumerate(ws) if w in cset]
    if len(idx) < 2: continue
    seq = [ws[i] for i in idx]
    gs = [" ".join(ws[a+1:b]) for a, b in zip(idx, idx[1:])]
    lead = " ".join(ws[:idx[0]])
    if len(lead.split()) > 3: continue
    samples.append((seq, gs, lead))
    gaps.update(gs); gaps[lead] += 1
patterns = ["<none>"] + [p for p, n in gaps.most_common(40) if p and n >= 5]
p2i = {p: i for i, p in enumerate(patterns)}
print(f"内容词 {len(content)}，间隙模式 {len(patterns)}，样本 {len(samples)}")

class GapModel(nn.Module):
    def __init__(self, d=96, n=0):
        super().__init__()
        self.emb = nn.Embedding(len(content), d)
        self.pos = nn.Embedding(12, d)
        self.net = nn.Sequential(nn.Linear(d*5, 192), nn.GELU(), nn.Linear(192, len(patterns)))
        self.head = nn.Sequential(nn.Linear(d*5, 192), nn.GELU(), nn.Linear(192, len(patterns)))
    def forward(self, a, b, pa, pb, i):
        x = torch.cat([self.emb(a), self.emb(b), self.emb(pa), self.emb(pb), self.pos(i)], -1)
        return self.net(x)

    def lead(self, first, second):
        x = torch.cat([self.emb(first), self.emb(second), self.emb(first), self.emb(second), self.pos(torch.tensor([0]))], -1)
        return self.head(x)

def pad(i, seq): return c2i[seq[max(0, min(i, len(seq)-1))]]
m = GapModel(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
n = len(samples)
for step in range(1, 1201):
    idx = torch.randint(0, n, (64,))
    A, B, PA, PB, I, Y = [], [], [], [], [], []
    for r in idx.tolist():
        seq, gs, _ = samples[r]
        for j, g in enumerate(gs[:11]):
            A.append(pad(j, seq)); B.append(pad(j+1, seq))
            PA.append(pad(j-1, seq)); PB.append(pad(j+2, seq)); I.append(j)
            Y.append(p2i.get(g, 0))
    loss = F.cross_entropy(m(torch.tensor(A), torch.tensor(B), torch.tensor(PA), torch.tensor(PB), torch.tensor(I)), torch.tensor(Y))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0: print(f"step {step} ce {loss.item():.3f}")

def fill(seq):
    out = []
    for j in range(len(seq) - 1):
        p = patterns[int(m(torch.tensor([pad(j,seq)]), torch.tensor([pad(j+1,seq)]),
                             torch.tensor([pad(j-1,seq)]), torch.tensor([pad(j+2,seq)]),
                             torch.tensor([j])).argmax())]
        out.append(seq[j])
        if p != "<none>": out += p.split()
    out.append(seq[-1])
    return out

# 用 gold 内容词顺序测
f1s = []
for seq, gs, lead in samples[:400]:
    pred = fill(seq)
    gold = " ".join(lead.split() + [x for pair in zip(seq, gs) for x in ([pair[0]] + (pair[1].split() if pair[1] else []))] + [seq[-1]]).split()
    tp = sum((Counter(pred) & Counter(gold)).values())
    f1s.append(2*tp/max(len(pred)+len(gold),1))
print(f"第 2 段（gold 顺序）：词级 F1 {np.mean(f1s)*100:.0f}%")
torch.save({"model": m.state_dict(), "patterns": patterns, "content": content}, ROOT/"ckpt/stage2_gap.pt")
