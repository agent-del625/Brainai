#!/usr/bin/env python3
"""先验证「积木法」的上限：用 K 个基本笔画重建 QuickDraw，能还原到什么程度。

这一步不训练任何模型，只回答一个问题：
    给定积木库，真实简笔画的还原度天花板是多少？
如果天花板就低，那后面的训练注定白费。
"""
import numpy as np, cv2, random
from pathlib import Path
from collections import Counter
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])

def resample(stroke, n=16):
    """把一条笔画重采样成 n 个点（弧长等距），用于比较形状。"""
    p=np.asarray(stroke,dtype=np.float32)
    if len(p)<2: return None
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if seg.sum()<1e-6: return None
    t=np.concatenate([[0],np.cumsum(seg)]); t/=t[-1]
    u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,p[:,0]), np.interp(u,t,p[:,1])],1)

def norm_shape(pts):
    """平移到中心、按包围盒尺度归一化，使形状可比。"""
    c=pts.mean(0); q=pts-c
    sc=np.abs(q).max()+1e-6
    return q/sc

# 收集所有笔画（每类采样，控制规模）
random.seed(0)
idx=random.sample(range(len(seqs)), 6000)
shapes=[]; meta=[]
for i in idx:
    for st in seqs[i]:
        r=resample(st)
        if r is None: continue
        shapes.append(norm_shape(r)); meta.append((i,))
shapes=np.stack(shapes)
print("笔画总数", len(shapes))

# k-means 聚类成积木库
from numpy.random import default_rng
rng=default_rng(0)
K=24
cent=shapes[rng.choice(len(shapes), K, replace=False)]
for it in range(12):
    dist=((shapes[:,None,:,:]-cent[None])**2).sum((-1,-2))
    lab=dist.argmin(1)
    for k in range(K):
        m=lab==k
        if m.sum()>0: cent[k]=shapes[m].mean(0)
print("聚类完成，各类样本数", np.bincount(lab, minlength=K).tolist())

# 每条笔画 → 最近积木 + 位置/尺度，再还原回画布，算 IoU
def render_strokes(strokes, size=S):
    img=np.zeros((size,size),np.float32)
    for st in strokes:
        p=np.asarray(st,dtype=np.float32)
        q=np.stack([(p[:,0]+1)*size/2-0.5,(p[:,1]+1)*size/2-0.5],1)
        for a,b in zip(q,q[1:]):
            cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1.,1)
    return img

ious=[]
for i in random.sample(idx, 400):
    strokes=seqs[i]
    orig=render_strokes(strokes)
    rebuilt=[]
    for st in strokes:
        r=resample(st)
        if r is None: continue
        s=norm_shape(r)
        k=int(((s[None]-cent)**2).sum((-1,-2)).argmin())
        c=r.mean(0); sc=np.abs(r-c).max()+1e-6
        rebuilt.append(cent[k]*sc+c)
    rec=render_strokes(rebuilt)
    A=orig>0.3; B=rec>0.3
    ious.append((A&B).sum()/max((A|B).sum(),1))
ious=np.array(ious)
print(f"积木重建 IoU：均值 {ious.mean()*100:.1f}%  中位 {np.median(ious)*100:.1f}%  >40% 的占比 {(ious>0.4).mean()*100:.0f}%")

# 每幅画的积木数分布
cnt=[len(seqs[i]) for i in idx]
print("每幅画笔数：均值 %.1f 中位 %d 最大 %d" % (np.mean(cnt), np.median(cnt), max(cnt)))

# 可视化：真实 vs 积木重建（各 4 幅）
rows=[]
for i in random.sample(idx,4):
    strokes=seqs[i]; orig=render_strokes(strokes)
    rebuilt=[]
    for st in strokes:
        r=resample(st)
        if r is None: continue
        s=norm_shape(r); k=int(((s[None]-cent)**2).sum((-1,-2)).argmin())
        c=r.mean(0); sc=np.abs(r-c).max()+1e-6
        rebuilt.append(cent[k]*sc+c)
    rec=render_strokes(rebuilt)
    rows.append(np.hstack([(255*(1-orig)).astype(np.uint8),np.full((S,4),200,np.uint8),
                           (255*(1-rec)).astype(np.uint8)]))
cv2.imwrite(str(ROOT/"logs/turtle/prim_probe.png"), np.vstack(rows))
np.savez_compressed(ROOT/"data/primitives.npz", cent=cent)
print("图:", ROOT/"logs/turtle/prim_probe.png")
