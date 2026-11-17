#!/usr/bin/env python3
"""积木数据集 v6：11 类 QuickDraw，64×64，K=32，带类别标签（供概念→积木生成）。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=14; G=16; RES=4; SB=16; RB=16; K=32
d=np.load(ROOT/"data/quickdraw12.npz",allow_pickle=True)
seqs=list(d["seqs"]); words=[str(w) for w in d["words"]]
cats=sorted(set(words)); c2i={c:i for i,c in enumerate(cats)}
print("类别",cats,"样本",len(seqs))
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
    return [p[int(len(p)*k/parts):min(int(len(p)*(k+1)/parts)+1,len(p))] for k in range(parts)]
def nshape(q):
    c=q.mean(0); z=q-c
    return z/(np.abs(z).max()+1e-6)
def oriented(z):
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    return z@vt.T, np.arctan2(vt[0,1],vt[0,0])
def render(seq):
    img=np.zeros((S,S),np.uint8)
    for st in seq:
        q=np.stack([(np.asarray(st)[:,0]+1)*S/2-0.5,(np.asarray(st)[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img
random.seed(31); idx=random.sample(range(len(seqs)),min(16000,len(seqs)))
shapes=[]
for i in idx[:5000]:
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is not None: shapes.append(oriented(nshape(r))[0])
shapes=np.stack(shapes)
rng=np.random.default_rng(0); cent=shapes[rng.choice(len(shapes),K,replace=False)].copy()
for _ in range(10):
    lab=((shapes[:,None]-cent[None])**2).sum((-1,-2)).argmin(1)
    for k in range(K):
        m=lab==k
        if m.sum(): cent[k]=shapes[m].mean(0)
print("词表 K=%d"%K)
X=[];M=[];IMG=[];Y=[]
for i in idx:
    items=[]
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            z,th=oriented(nshape(r))
            pid=int(((z[None]-cent)**2).sum((-1,-2)).argmin())
            c=r.mean(0); half=float(np.abs(r-c).max())
            fx=(c[0]+1)/2*G; fy=(c[1]+1)/2*G
            gx=int(np.clip(np.floor(fx),0,G-1)); gy=int(np.clip(np.floor(fy),0,G-1))
            rx=int(np.clip((fx-gx)*RES,0,RES-1)); ry=int(np.clip((fy-gy)*RES,0,RES-1))
            sb=int(np.clip(half*SB/0.6,0,SB-1)); rb=int(((th+np.pi)/(2*np.pi)*RB))%RB
            items.append((gy*G+gx,pid,gx*G+gy,rx*RES+ry,sb,rb))
    items.sort(key=lambda t:t[0]); items=items[:N]
    sl=np.zeros((N,5),np.int64); mk=np.zeros(N,np.int64)
    for k,(_,pid,cell,res,sb,rb) in enumerate(items): sl[k]=(pid,cell,res,sb,rb); mk[k]=1
    X.append(sl);M.append(mk);IMG.append(render(seqs[i])*255);Y.append(c2i[words[i]])
X=np.stack(X);M=np.stack(M);IMG=np.stack(IMG);Y=np.array(Y)
p=np.random.permutation(len(X)); a,b=int(len(X)*0.85),int(len(X)*0.93)
np.savez_compressed(ROOT/"data/prim_v6_train.npz",slots=X[p[:a]],mask=M[p[:a]],imgs=IMG[p[:a]],y=Y[p[:a]],cent=cent,cfg=np.array([N,G,RES,SB,RB,K]))
np.savez_compressed(ROOT/"data/prim_v6_val.npz",slots=X[p[a:b]],mask=M[p[a:b]],imgs=IMG[p[a:b]],y=Y[p[a:b]])
np.savez_compressed(ROOT/"data/prim_v6_test.npz",slots=X[p[b:]],mask=M[p[b:]],imgs=IMG[p[b:]],y=Y[p[b:]])
np.save(ROOT/"data/prim_v6_cats.npy", np.array(cats))
print("划分",a,b-a,len(X)-b,"类别数",len(cats))
