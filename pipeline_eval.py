#!/usr/bin/env python3
"""端到端：第 1 段排内容词 → 第 2 段插功能词 → 和参考答案比。"""
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch

from stage1_order import Orderer, words_of, FUNC, K, MAXLEN

ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float(); caps = [str(c) for c in z["captions"]]
st = torch.load(ROOT/"ckpt/stage1_order.pt", map_location="cpu", weights_only=False)
content = st["content"]; c2i = {w: i for i, w in enumerate(content)}
m = Orderer(n=len(content)); m.load_state_dict(st["model"]); m.eval()

# 第 2 段统计表
cset = set(content)
mid = defaultdict(Counter); lead = Counter()
for c in caps:
    ws = words_of(c); idx = [i for i, w in enumerate(ws) if w in cset]
    if not idx: continue
    lead[" ".join(ws[:idx[0]])] += 1
    for a, b in zip(idx, idx[1:]):
        gap = " ".join(ws[a+1:b])
        if len(gap.split()) <= 3: mid[(ws[a], ws[b])][gap] += 1
def fill(seq):
    if not seq: return []
    out = lead.most_common(1)[0][0].split() if lead else []
    out.append(seq[0])
    for a, b in zip(seq, seq[1:]):
        g = mid.get((a, b))
        if g and g.most_common(1)[0][0]: out += g.most_common(1)[0][0].split()
        out.append(b)
    return out

def order(i):
    with torch.no_grad():
        h0 = m.enc(feats[i:i+1])
        ws = [w for w in words_of(caps[i]) if w in c2i][:K]
        rows = [c2i[w] for w in ws] + [0]*(K-len(ws))
        bag = m.emb(torch.tensor([rows])); h = torch.zeros(1,128); prev = torch.zeros(1,128); out=[]
        used = torch.zeros(1,K,dtype=torch.bool); pad = torch.tensor(rows) == 0
        for t in range(MAXLEN):
            lg, h = m.step(h0, bag, prev, h, t)
            if lg[:, -1].item() > 0 or bool((used|pad)[0].all()): break
            j = int(lg[:, :K].masked_fill(used|pad, -1e4).argmax())
            out.append(content[rows[j]]); used[0, j] = True
            prev = m.emb(torch.tensor([rows[j]]))
        return out

f1s, exact = [], []
for i in range(300, 500):
    pred = fill(order(i)); gold = words_of(caps[i])
    tp = sum((Counter(pred) & Counter(gold)).values())
    f1s.append(2*tp/max(len(pred)+len(gold),1)); exact.append(pred == gold)
print(f"端到端：词级 F1 {np.mean(f1s)*100:.0f}%   整句完全一致 {np.mean(exact)*100:.0f}%")
for i in range(300, 304):
    print(f"  生成: {' '.join(fill(order(i)))[:60]}")
    print(f"  参考: {caps[i][:60]}")
