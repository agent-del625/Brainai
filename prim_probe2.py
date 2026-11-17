#!/usr/bin/env python3
"""积木法上限测量 v2：换更合理的度量（膨胀 IoU + 倒角距离），并扫 K。

坑 1：1px 线条的像素 IoU 会严重低估相似度（差 1 像素就归零）。
"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])

def resample(st,n=16):
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2: return None
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if seg.sum()<1e-6: return None
    t=np.concatenate([[0],np.cumsum(seg)]); t/=t[-1]
    u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,p[:,0]),np.interp(u,t,p[:,1])],1)

def nshape(q):
    c=q.mean(0); z=q-c
    return z/(np.abs(z).max()+1e-6)

def render(strokes,size=S):
    img=np.zeros((size,size),np.uint8)
    for st in strokes:
        p=np.asarray(st,dtype=np.float32)
        q=np.stack([(p[:,0]+1)*size/2-0.5,(p[:,1]+1)*size/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img

def chamfer(a,b):
    ya,xa=np.nonzero(a>0); yb,xb=np.nonzero(b>0)
    if len(xa)==0 or len(xb)==0: return 99.
    A=np.stack([xa,ya],1).astype(np.float32); B=np.stack([xb,yb],1).astype(np.float32)
    D=((A[:,None,:]-B[None])**2).sum(-1)
    return float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2

random.seed(1)
idx=random.sample(range(len(seqs)), 5000)
shapes=[]
for i in idx:
    for st in seqs[i]:
        r=resample(st)
        if r is not None: shapes.append(nshape(r))
shapes=np.stack(shapes)
test=random.sample(idx,300)

print(f"{'K':>4} {'膨胀IoU':>9} {'倒角(px)':>10} {'最差类别':>10}")
for K in (12,24,48,96):
    rng=np.random.default_rng(0)
    cent=shapes[rng.choice(len(shapes),K,replace=False)].copy()
    for _ in range(10):
        lab=((shapes[:,None]-cent[None])**2).sum((-1,-2)).argmin(1)
        for k in range(K):
            m=lab==k
            if m.sum(): cent[k]=shapes[m].mean(0)
    ious=[]; cham=[]; per={}
    for i in test:
        orig=render(seqs[i]); rebuilt=[]
        for st in seqs[i]:
            r=resample(st)
            if r is None: continue
            k=int(((nshape(r)[None]-cent)**2).sum((-1,-2)).argmin())
            c=r.mean(0); sc=np.abs(r-c).max()+1e-6
            rebuilt.append(cent[k]*sc+c)
        rec=render(rebuilt)
        o=cv2.dilate(orig,np.ones((3,3),np.uint8)); rr=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o>0; B=rr>0
        ious.append((A&B).sum()/max((A|B).sum(),1)); cham.append(chamfer(orig,rec))
        per.setdefault(words[i],[]).append(ious[-1])
    worst=min(per.items(), key=lambda kv: np.mean(kv[1]))
    print(f"{K:>4} {np.mean(ious)*100:>8.1f}% {np.mean(cham):>10.2f} {worst[0]+' '+str(round(np.mean(worst[1])*100,1))+'%':>10}")
    if K==48:
        np.savez_compressed(ROOT/"data/primitives.npz", cent=cent)
print("（膨胀 IoU：线条加粗 3px 后比较；倒角距离：越小越好，单位像素）")
