#!/usr/bin/env python3
"""QuickDraw → 笔画序列数据集（真实人类简笔画，正是 Turtle 手该学的东西）。"""
import json, glob
import numpy as np
from pathlib import Path
ROOT = Path("/srv/brainai"); S = 64
MAX_STROKES, MAX_PTS = 6, 14

def norm(drawing):
    """QuickDraw: [[x...],[y...]] per stroke，坐标 0-255 → [-1,1]，限制笔画/点数。"""
    out = []
    for st in drawing[:MAX_STROKES]:
        xs, ys = st[0], st[1]
        if len(xs) < 2: continue
        idx = np.linspace(0, len(xs)-1, min(MAX_PTS, len(xs))).astype(int)
        pts = np.stack([np.array(xs)[idx], np.array(ys)[idx]], 1).astype(np.float32)
        pts = pts/255.0*2-1
        out.append(pts)
    return out if len(out) >= 1 else None

seqs, words, keys = [], [], []
for f in sorted(glob.glob(str(ROOT/"data/downloads/quickdraw_*.ndjson"))):
    word = Path(f).stem.replace("quickdraw_","")
    n = 0
    for line in open(f, encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        if not d.get("recognized"): continue
        s = norm(d["drawing"])
        if s: seqs.append(s); words.append(word); n += 1
        if n >= 10000: break
    print(word, n)
np.savez_compressed(ROOT/"data/quickdraw.npz", seqs=np.array(seqs, dtype=object),
                    words=np.array(words), allow_pickle=True)
print("总数", len(seqs))
