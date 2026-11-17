#!/usr/bin/env python3
"""建积木数据集：图像 → (积木ID, 网格位置, 尺度) 固定 12 槽。"""
import numpy as np, cv2, random, json
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=12; G=8; SB=3
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs,words=list(d["seqs"]),list(d["words"])
cent=np.load(ROOT/"data/primitives.npz")["cent"]; K=len(cent)

def resample(st,n=16):
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2: return None
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if seg.sum()<1e-6: return None
    t=np.concatenate([[0],np.cumsum(seg)]); t/=t[-1]
    u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,p[:,0]),np.interp(u,t,p[:,1])],1)

def split(st,parts=3):
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

def encode_drawing(i):
    """返回 (slots[N,3] = [pid, gx*G+gy, sbin], 原图)"""
    items=[]
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            pid=int(((nshape(r)[None]-cent)**2).sum((-1,-2)).argmin())
            c=r.mean(0); half=np.abs(r-c).max()+1e-6
            gx=int(np.clip((c[0]+1)/2*G,0,G-1)); gy=int(np.clip((c[1]+1)/2*G,0,G-1))
            sb=int(np.clip(half*SB/0.6,0,SB-1))     # 尺度分档
            items.append((gy*G+gx, pid, gy*G+gx, sb))
    items.sort(key=lambda t: t[0])                   # 规范顺序：自上而下、自左而右
    items=items[:N]
    slots=np.zeros((N,3),np.int64); mask=np.zeros(N,np.int64)
    for k,(_,pid,pos,sb) in enumerate(items):
        slots[k]=(pid,pos,sb); mask[k]=1
    return slots, mask, render(seqs[i])

random.seed(3)
idx=random.sample(range(len(seqs)), 6000)
X=[]; M=[]; IMG=[]
for i in idx:
    s,mk,img=encode_drawing(i)
    X.append(s); M.append(mk); IMG.append(img*255)
np.savez_compressed(ROOT/"data/prim_ds.npz",
                    slots=np.stack(X), mask=np.stack(M),
                    imgs=np.stack(IMG).astype(np.uint8),
                    words=np.array([words[i] for i in idx]),
                    cent=cent)
print("数据集", np.stack(X).shape, "平均有效槽位", np.stack(M).sum(1).mean())
print("积木词表", K, "→ data/prim_ds.npz")
