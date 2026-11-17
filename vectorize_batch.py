#!/usr/bin/env python3
"""批量把图片转成笔画序列，存成训练数据。"""
import numpy as np, cv2, onnxruntime as ort
from pathlib import Path

ROOT = Path("/srv/brainai")
import onnxruntime as _ort
_sk = _ort.InferenceSession(str(ROOT/"data/downloads/sketch.onnx"), providers=["CPUExecutionProvider"])
_si, _so = _sk.get_inputs()[0].name, _sk.get_outputs()[0].name

def to_lines(img_bgr, size=256):
    x = cv2.resize(img_bgr, (256,256)).astype(np.float32)/255.0
    x = (x - 0.5)/0.5
    o = _sk.run([_so], {_si: x.transpose(2,0,1)[None]})[0][0,0]
    o = (o - o.min())/(o.max()-o.min()+1e-6)
    if o.mean() > 0.5: o = 1 - o
    return (o*255).clip(0,255).astype(np.uint8)

def vectorize(line_img, max_strokes=6, max_points=12):
    bw = (line_img > 90).astype(np.uint8)*255
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda c: -cv2.arcLength(c, False))[:max_strokes]
    out = []
    for c in cnts:
        c = cv2.approxPolyDP(c, 3.0, False).reshape(-1,2)
        if len(c) < 3: continue
        idx = np.linspace(0, len(c)-1, min(max_points, len(c))).astype(int)
        pts = c[idx].astype(np.float32)
        pts[:,0] = pts[:,0]/line_img.shape[1]*2-1
        pts[:,1] = pts[:,1]/line_img.shape[0]*2-1
        out.append(pts)
    return out

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = z["feats"]
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
N = min(1600, len(imgs))
seqs, idxs = [], []
for i in range(N):
    bgr = cv2.cvtColor(cv2.resize(imgs[i].transpose(1,2,0), (256,256)), cv2.COLOR_RGB2BGR)
    lines = to_lines(bgr)
    st = vectorize(lines)
    if len(st) >= 2:
        seqs.append(st); idxs.append(i)
    if (i+1) % 100 == 0: print(f"{i+1}/{N}", flush=True)
print("有效样本", len(seqs))
import time
shard = []
for k in range(0, len(seqs), 200):
    part = seqs[k:k+200]; pidx = idxs[k:k+200]
    np.savez_compressed(ROOT/f"data/strokes_{int(time.time())}_{k}.npz",
                        idx=np.array(pidx), seqs=np.array(part, dtype=object), allow_pickle=True)
    shard.append(len(part))
print("分片保存:", shard)
print("保存 data/strokes.npz")