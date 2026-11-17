#!/usr/bin/env python3
"""积木数据集 v2：格子(16×16) + 格内细分(4×4) + 尺度 16 档，按位置规范排序。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=14; G=16; RES=4; SB=16
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs=list(d["seqs"]); cent=np.load(ROOT/"data/primitives.npz")["cent"]; K=len(cent)
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
random.seed(3); idx=random.sample(range(len(seqs)),8000)
X=[];M=[];IMG=[]
for i in idx:
    items=[]
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            pid=int(((nshape(r)[None]-cent)**2).sum((-1,-2)).argmin())
            c=r.mean(0); half=float(np.abs(r-c).max())
            fx=(c[0]+1)/2*G; fy=(c[1]+1)/2*G
            gx=int(np.clip(np.floor(fx),0,G-1)); gy=int(np.clip(np.floor(fy),0,G-1))
            rx=int(np.clip((fx-gx)*RES,0,RES-1)); ry=int(np.clip((fy-gy)*RES,0,RES-1))
            sb=int(np.clip(half*SB/0.6,0,SB-1))
            items.append((gy*G+gx,pid,gx*G+gy,rx*RES+ry,sb))
    items.sort(key=lambda t:t[0]); items=items[:N]
    sl=np.zeros((N,4),np.int64); mk=np.zeros(N,np.int64)
    for k,(_,pid,cell,res,sb) in enumerate(items):
        sl[k]=(pid,cell,res,sb); mk[k]=1
    X.append(sl);M.append(mk);IMG.append(render(seqs[i])*255)
np.savez_compressed(ROOT/"data/prim_ds2.npz", slots=np.stack(X), mask=np.stack(M),
                    imgs=np.stack(IMG).astype(np.uint8), cent=cent,
                    cfg=np.array([N,G,RES,SB]))
print("数据集", np.stack(X).shape, "平均槽位", np.stack(M).sum(1).mean(), "cfg N,G,RES,SB =", N,G,RES,SB)
