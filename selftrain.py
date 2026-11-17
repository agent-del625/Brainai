#!/usr/bin/env python3
"""自训练闭环：采样 → 三重检验 → 回灌 → 重训，看每轮生成质量是否变化。"""
import re, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
feats = torch.from_numpy(z["feats"]).float()

s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
vocab = s2["content"]; ci = {w:i for i,w in enumerate(vocab)}; V = len(vocab)+2
tri = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
concepts = set(tri["names"])

# 共现先验
co = defaultdict(int)
for c in caps:
    ws = sorted(set(words_of(c)))
    for a in range(len(ws)):
        for b in range(a+1, len(ws)):
            co[(ws[a], ws[b])] += 1
def plaus(ws):
    ws = sorted(set(ws)); vals = [co.get((ws[a], ws[b]), 0) for a in range(len(ws)) for b in range(a+1, len(ws))]
    return float(np.mean(np.log1p(vals))) if vals else 0.0
seen = {" ".join(sorted(set(words_of(c)))) for c in caps}

class LM(nn.Module):
    def __init__(self, d=96, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.gru = nn.GRU(d, h, batch_first=True); self.out = nn.Linear(h, V)
    def forward(self, x):
        h,_ = self.gru(self.emb(x)); return self.out(h)

seqs = [[V-2]+[ci[w] for w in words_of(c) if w in ci][:6]+[V-1] for c in caps if len([w for w in words_of(c) if w in ci])>=2]
print(f"初始训练序列 {len(seqs)}")
L = 8
def to_X(ss): return torch.tensor([(s[:L]+[0]*(L-len(s))) for s in ss])

m = LM(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
def train(steps=800):
    X = to_X(seqs)
    for _ in range(steps):
        idx = torch.randint(0, X.size(0), (128,))
        logits = m(X[idx][:, :-1])
        loss = F.cross_entropy(logits.reshape(-1,V), X[idx][:,1:].reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

inv = {i:w for w,i in ci.items()}
def sample(n=200, temp=0.9):
    out = []
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor([[V-2]]); ids = []
            for _ in range(6):
                p = torch.softmax(m(x)[:, -1]/temp, -1)
                nxt = int(torch.multinomial(p, 1))
                if nxt == V-1 or nxt not in inv: break
                ids.append(nxt); x = torch.cat([x, torch.tensor([[nxt]])], 1)
            if len(ids) >= 2: out.append(ids)
    return out

def evaluate(cands):
    ok, novel = 0, 0
    for ids in cands:
        ws = [inv[i] for i in ids]
        if not any(w in concepts for w in ws): continue
        key = " ".join(sorted(set(ws)))
        is_novel = key not in seen
        score = plaus(ws)
        if score > 0.7 and is_novel:
            ok += 1; novel += 1
            if key not in seen:
                seqs.append([V-2]+ids+[V-1]); seen.add(key)
    return ok, novel

for rnd in range(1, 4):
    loss = train(600)
    cands = sample(300)
    ok, novel = evaluate(cands)
    print(f"第 {rnd} 轮：CE {loss:.3f}  采样 300 → 接受 {ok}（新组合 {novel}）  训练集 {len(seqs)}")
torch.save({"model": m.state_dict(), "vocab": vocab}, ROOT/"ckpt/concept_lm_self.pt")
