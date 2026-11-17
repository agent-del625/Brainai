#!/usr/bin/env python3
"""加上旋转/缩放后，积木法的还原上限能到多少（回答"积木必须能转能缩放"）。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True); seqs=list(d["seqs"])
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
def rot(q,th):
    c,s=np.cos(th),np.sin(th)
    return q@np.array([[c,s],[-s,c]],dtype=np.float32)

random.seed(7); idx=random.sample(range(len(seqs)),2500); test=random.sample(idx,220)
# 建库：形状 + 主方向（PCA），朝向归一化后再聚类
shapes=[]
for i in idx:
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            z=nshape(r)
            u,s,vt=np.linalg.svd(z,full_matrices=False)
            z=z@vt.T                      # 主方向对齐
            shapes.append(z)
shapes=np.stack(shapes)
K=96; rng=np.random.default_rng(0)
cent=shapes[rng.choice(len(shapes),K,replace=False)].copy()
for _ in range(8):
    lab=((shapes[:,None]-cent[None])**2).sum((-1,-2)).argmin(1)
    for k in range(K):
        m=lab==k
        if m.sum(): cent[k]=shapes[m].mean(0)

def render_set(items):
    img=np.zeros((S,S),np.uint8)
    for pid,th,sc,cx,cy in items:
        sh=rot(cent[pid],th)*sc+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img

for NROT,NSC,NPOS in [(1,1,0),(16,1,0),(16,16,0),(16,16,1),(16,32,1),(32,32,1)]:
    ious=[];chs=[]
    for i in test:
        gt=np.zeros((S,S),np.uint8)
        for st in seqs[i]:
            q=np.stack([(np.asarray(st)[:,0]+1)*S/2-0.5,(np.asarray(st)[:,1]+1)*S/2-0.5],1)
            for a,b in zip(q,q[1:]): cv2.line(gt,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
        items=[]
        for st in seqs[i]:
            for seg in split(st,3):
                r=resample(seg)
                if r is None: continue
                z=nshape(r); u,s,vt=np.linalg.svd(z,full_matrices=False)
                th=np.arctan2(vt[0,1],vt[0,0]); zz=z@vt.T
                k=int(((zz[None]-cent)**2).sum((-1,-2)).argmin())
                c=r.mean(0); sc=float(np.abs(r-c).max()+1e-6)
                if NROT>1: th=round(th/(np.pi/NROT))*(np.pi/NROT)
                else: th=0.0
                if NSC>1: sc=round(sc*NSC/0.6)/NSC*0.6
                cx,cy=c
                if NPOS==0:
                    cx=round(cx*8)/8; cy=round(cy*8)/8      # 1/8 网格吸附
                items.append((k,th,sc,cx,cy))
        rec=render_set(items)
        o=cv2.dilate(gt,np.ones((3,3),np.uint8)); rr=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o>0;B=rr>0; ious.append((A&B).sum()/max((A|B).sum(),1))
        ya,xa=np.nonzero(gt>0); yb,xb=np.nonzero(rec>0)
        if len(xa) and len(xb):
            A2=np.stack([xa,ya],1).astype(np.float32); B2=np.stack([xb,yb],1).astype(np.float32)
            D=((A2[:,None]-B2[None])**2).sum(-1)
            chs.append(float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())/2)
    print(f"旋转 {NROT:>2} 档 缩放 {NSC:>2} 档 位置吸附 {'无' if NPOS==0 else '1/8格'} → IoU {np.mean(ious)*100:>5.1f}%  倒角 {np.mean(chs):.2f}px")
