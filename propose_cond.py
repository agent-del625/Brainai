#!/usr/bin/env python3
"""条件化提案：给定 1-2 个种子概念，让概念 LM 补全场景，再看接受率。"""
import re, random
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
co = defaultdict(int); seen = set()
for c in caps:
    ws = sorted(set(words_of(c)))
    for a in range(len(ws)):
        for b in range(a+1, len(ws)): co[(ws[a], ws[b])] += 1
    seen.add(" ".join(ws))
def plaus(ws):
    ws = sorted(set(ws)); v = [co.get((ws[a], ws[b]), 0) for a in range(len(ws)) for b in range(a+1, len(ws))]
    return float(np.mean(np.log1p(v))) if v else 0.0

s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
vocab = s2["content"]; ci = {w:i for i,w in enumerate(vocab)}; inv = {i:w for w,i in ci.items()}; V = len(vocab)+2
tri = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
concepts = [w for w in tri["names"] if w in ci]

class LM(nn.Module):
    def __init__(self, d=96, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.gru = nn.GRU(d, h, batch_first=True); self.out = nn.Linear(h, V)
    def forward(self, x):
        h,_ = self.gru(self.emb(x)); return self.out(h)

ck = torch.load(ROOT/"ckpt/concept_lm.pt", map_location="cpu", weights_only=False)
m = LM(); m.load_state_dict(ck["model"]); m.eval()

def sample_cond(seed_words, n=60, temp=0.8, maxlen=6):
    out = []
    ids0 = [ci[w] for w in seed_words if w in ci]
    with torch.no_grad():
        for _ in range(n):
            ids = list(ids0)
            x = torch.tensor([[V-2] + ids])
            for _ in range(maxlen - len(ids)):
                p = torch.softmax(m(x)[:, -1]/temp, -1)
                nxt = int(torch.multinomial(p, 1))
                if nxt == V-1 or nxt not in inv: break
                ids.append(nxt); x = torch.cat([x, torch.tensor([[nxt]])], 1)
            if len(ids) >= len(ids0) + 1: out.append(ids)
    return out

def accept_rate(cands):
    ok = 0
    for ids in cands:
        ws = [inv[i] for i in ids]
        if not any(w in concepts for w in ws): continue
        key = " ".join(sorted(set(ws)))
        if key in seen: continue
        if plaus(ws) > 0.7: ok += 1
    return ok, len(cands)

random.seed(1)
tot_ok = tot_n = 0
examples = []
for seed in random.sample(concepts, 5):
    cands = sample_cond([seed], 60)
    ok, n = accept_rate(cands)
    tot_ok += ok; tot_n += n
    if ok: examples.append((seed, ok))
    print(f"种子 {seed:12s}：采样 {n} → 接受 {ok}")
print(f"\n条件化接受率 {tot_ok/tot_n*100:.1f}%（无条件时约 0.7%）")

# 打印几条通过的例子
for seed in [e[0] for e in examples[:3]]:
    for ids in sample_cond([seed], 20)[:20]:
        ws = [inv[i] for i in ids]
        if plaus(ws) > 0.7 and " ".join(sorted(set(ws))) not in seen:
            print("  例:", " · ".join(ws)); break
