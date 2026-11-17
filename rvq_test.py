#!/usr/bin/env python3
"""对比两种声音编码的信息保真度：①单码本 k-means ②残差量化 RVQ（多码本）。"""
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"]; Y=d["target"]; cats=[str(c) for c in d["cats"]]
N=len(M)
# patch 化（标准化后）
P=np.zeros((N,40,256),dtype=np.float32)
for i,m in enumerate(M):
    m=(m-m.mean())/max(m.std(),1e-6)
    k=0
    for a in range(5):
        for b in range(8):
            P[i,k]=m[a*16:(a+1)*16,b*16:(b+1)*16].ravel(); k+=1
flat=P.reshape(-1,256)
def kmeans(X,K,iters=15,seed=0):
    rng=np.random.default_rng(seed); C=X[rng.choice(len(X),K,replace=False)].copy()
    for _ in range(iters):
        idx=np.empty(len(X),dtype=np.int32)
        for s in range(0,len(X),20000):
            ch=X[s:s+20000]; idx[s:s+20000]=((ch[:,None,:]-C[None,:,:])**2).sum(-1).argmin(1)
        for k in range(K):
            sel=X[idx==k]
            if len(sel): C[k]=sel.mean(0)
    return C,idx
def rvq(X,codebooks=(64,64,64),seed=0):
    R=X.copy(); Cs=[]; idxs=[]
    for K in codebooks:
        C,idx=kmeans(R,K,seed=seed); Cs.append(C); idxs.append(idx)
        R=R-C[idx]
    return Cs,idxs,R
for name,cfg in (("单码本 256",[256]),("单码本 64",[64]),("RVQ 64+64",[64,64]),("RVQ 64+64+64",[64,64,64])):
    if len(cfg)==1:
        C,idx=kmeans(flat,cfg[0]); recon=C[idx]
    else:
        Cs,idxs,R=rvq(flat,cfg); recon=sum(C[idx] for C,idx in zip(Cs,idxs))
    mse=float(((flat-recon)**2).mean()); bits=sum(np.log2(k) for k in cfg)
    print(f"{name:14s} 每块 {bits:.0f} bit  重建 MSE {mse:.4f}  40 块合计 {40*bits/8:.0f} 字节")
