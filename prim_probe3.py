#!/usr/bin/env python3
"""积木法上限 v3：先按曲率/长度把笔画切成小段，再量化——看还原度能到多少。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])

def resample(st,n=12):
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2: return None
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if seg.sum()<1e-6: return None
    t=np.concatenate([[0],np.cumsum(seg)]); t/=t[-1]
    u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,p[:,0]),np.interp(u,t,p[:,1])],1)

def split(st, parts=3):
    """把一条笔画按弧长等分成若干小段（每段 ≥3 点）。"""
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2*parts: return [p]
    out=[]
    for k in range(parts):
        a=int(len(p)*k/parts); b=int(len(p)*(k+1)/parts)+1
        seg=p[a:min(b,len(p))]
        if len(seg)>=2: out.append(seg)
    return out

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

random.seed(2)
idx=random.sample(range(len(seqs)), 4000)
test=random.sample(idx,250)

for parts in (1,2,3,4):
    shapes=[]
    for i in idx:
        for st in seqs[i]:
            for seg in split(st,parts):
                r=resample(seg)
                if r is not None: shapes.append(nshape(r))
    shapes=np.stack(shapes)
    K=48; rng=np.random.default_rng(0)
    cent=shapes[rng.choice(len(shapes),K,replace=False)].copy()
    for _ in range(8):
        lab=((shapes[:,None]-cent[None])**2).sum((-1,-2)).argmin(1)
        for k in range(K):
            m=lab==k
            if m.sum(): cent[k]=shapes[m].mean(0)
    ious=[]
    for i in test:
        orig=render(seqs[i]); rebuilt=[]
        for st in seqs[i]:
            for seg in split(st,parts):
                r=resample(seg)
                if r is None: continue
                k=int(((nshape(r)[None]-cent)**2).sum((-1,-2)).argmin())
                c=r.mean(0); sc=np.abs(r-c).max()+1e-6
                rebuilt.append(cent[k]*sc+c)
        rec=render(rebuilt)
        o=cv2.dilate(orig,np.ones((3,3),np.uint8)); rr=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o>0; B=rr>0
        ious.append((A&B).sum()/max((A|B).sum(),1))
    print(f"每笔切 {parts} 段：膨胀IoU {np.mean(ious)*100:.1f}%  每幅平均积木数 {np.mean([sum(len(split(st,parts)) for st in seqs[i]) for i in test]):.1f}")
