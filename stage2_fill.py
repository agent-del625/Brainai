#!/usr/bin/env python3
"""第 2 段：给排好的内容词插入功能词（统计法，不训练）。"""
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
cnt = Counter(w for c in caps for w in words_of(c))
content = set(w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC)

# 统计：两个相邻内容词之间最常见的功能词串
mid = defaultdict(Counter)
lead = Counter()
for c in caps:
    ws = words_of(c)
    idx = [i for i, w in enumerate(ws) if w in content]
    if not idx:
        continue
    lead[" ".join(ws[:idx[0]])] += 1
    for a, b in zip(idx, idx[1:]):
        gap = " ".join(ws[a+1:b])
        if len(gap.split()) <= 3:
            mid[(ws[a], ws[b])][gap] += 1

def fill(seq):
    if not seq: return []
    out = []
    head = lead.most_common(1)[0][0]
    if head: out += head.split()
    out.append(seq[0])
    for a, b in zip(seq, seq[1:]):
        gap = mid.get((a, b))
        if gap:
            g = gap.most_common(1)[0][0]
            if g: out += g.split()
        out.append(b)
    return out

# 用 gold 内容词顺序测试插入效果（隔离第 2 段）
sample = caps[:200]
f1s = []
for c in sample:
    gold = words_of(c)
    seq = [w for w in gold if w in content]
    pred = fill(seq)
    inter = Counter(pred) & Counter(gold)
    tp = sum(inter.values())
    f1s.append(2*tp/max(len(pred)+len(gold), 1))
print(f"仅插入功能词（内容词顺序给正确答案）：词级 F1 {np.mean(f1s)*100:.0f}%")
for c in sample[:4]:
    seq = [w for w in words_of(c) if w in content]
    print(f"  生成: {' '.join(fill(seq))}")
    print(f"  参考: {c}")
