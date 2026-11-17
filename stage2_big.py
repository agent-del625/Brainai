#!/usr/bin/env python3
"""第 2 段（神经网络加强版）：更深更宽，同时预测句首和每个间隙。"""
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
c2i = {w: i for i, w in enumerate(content)}; cset = set(content)

samples, gaps, leads = [], Counter(), Counter()
for c in caps:
    ws = words_of(c); idx = [i for i, w in enumerate(ws) if w in cset]
    if len(idx) < 2: continue
    seq = [ws[i] for i in idx]
    gs = [" ".join(ws[a+1:b]) for a, b in zip(idx, idx[1:])]
    lead = " ".join(ws[:idx[0]])
    if len(lead.split()) > 3: continue
    samples.append((seq, gs, lead)); gaps.update(gs); leads[lead] += 1
patterns = ["<none>"] + [p for p, n in gaps.most_common(60) if p and n >= 3]
p2i = {p: i for i, p in enumerate(patterns)}
leadp = ["<none>"] + [p for p, n in leads.most_common(40) if p and n >= 3]
l2i = {p: i for i, p in enumerate(leadp)}
print(f"样本 {len(samples)}｜间隙模式 {len(patterns)}｜句首模式 {len(leadp)}")

D, H = 128, 512
class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(content), D)
        self.pos = nn.Embedding(14, D)
        self.gap = nn.Sequential(
            nn.Linear(D*5, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
            nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
            nn.Linear(H, H//2), nn.GELU(),
            nn.Linear(H//2, len(patterns)))
        self.lead = nn.Sequential(
            nn.Linear(D*3, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
            nn.Linear(H, H//2), nn.GELU(),
            nn.Linear(H//2, len(leadp)))

def P(i, seq): return c2i[seq[max(0, min(i, len(seq)-1))]]

m = Big(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
n = len(samples)
for step in range(1, 3001):
    idx = torch.randint(0, n, (64,))
    A,B,PA,PB,I,Y,LX,LY = [],[],[],[],[],[],[],[]
    for r in idx.tolist():
        seq, gs, lead = samples[r]
        LX.append(P(0,seq)); LY.append(l2i.get(lead, 0))
        for j, g in enumerate(gs[:12]):
            A.append(P(j,seq)); B.append(P(j+1,seq)); PA.append(P(j-1,seq)); PB.append(P(j+2,seq))
            I.append(j); Y.append(p2i.get(g, 0))
    lg = m.gap(torch.cat([m.emb(torch.tensor(A)), m.emb(torch.tensor(B)), m.emb(torch.tensor(PA)),
                          m.emb(torch.tensor(PB)), m.pos(torch.tensor(I))], -1))
    ll = m.lead(torch.cat([m.emb(torch.tensor(LX)), m.emb(torch.tensor(LX)), m.emb(torch.tensor(LX))], -1))
    loss = F.cross_entropy(lg, torch.tensor(Y)) + 0.5*F.cross_entropy(ll, torch.tensor(LY))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 600 == 0: print(f"step {step} loss {loss.item():.3f}")

def fill(seq):
    out = []
    lead = leadp[int(m.lead(torch.cat([m.emb(torch.tensor([P(0,seq)])),
                                       m.emb(torch.tensor([P(0,seq)])),
                                       m.emb(torch.tensor([P(0,seq)]))], -1)).argmax())]
    if lead != "<none>": out += lead.split()
    for j in range(len(seq)-1):
        x = torch.cat([m.emb(torch.tensor([P(j,seq)])), m.emb(torch.tensor([P(j+1,seq)])),
                       m.emb(torch.tensor([P(j-1,seq)])), m.emb(torch.tensor([P(j+2,seq)])),
                       m.pos(torch.tensor([j]))], -1)
        p = patterns[int(m.gap(x).argmax())]
        out.append(seq[j])
        if p != "<none>": out += p.split()
    out.append(seq[-1]); return out

f1s = []
for seq, gs, lead in samples[:500]:
    pred = fill(seq)
    gold = " ".join(lead.split() + [x for pair in zip(seq, gs) for x in ([pair[0]] + (pair[1].split() if pair[1] else []))] + [seq[-1]]).split()
    tp = sum((Counter(pred) & Counter(gold)).values())
    f1s.append(2*tp/max(len(pred)+len(gold),1))
print(f"第 2 段（加强版，gold 顺序）：词级 F1 {np.mean(f1s)*100:.0f}%")
torch.save({"model": m.state_dict(), "patterns": patterns, "leadp": leadp, "content": content}, ROOT/"ckpt/stage2_big.pt")
