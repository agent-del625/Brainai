#!/usr/bin/env python3
"""把声音也变成离散 token：mel 切成 patch，k-means 量化成码字（和文字的 token 同一套机制）。"""
import numpy as np
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"]; cats=np.array([str(c) for c in d["cats"]])
print("mel",M.shape)
# patch: 80x128 → 5x8 个 16x16 块
P=[]
for m in M:
    m=(m-m.mean())/max(m.std(),1e-6)          # ★先标准化，否则静音块主导聚类
    patches=[]
    for i in range(5):
        for j in range(8):
            patches.append(m[i*16:(i+1)*16, j*16:(j+1)*16].ravel())
    P.append(patches)
P=np.array(P)                     # (N, 40, 256)
print("patch",P.shape)
flat=P.reshape(-1,256)
# k-means 128 码
from numpy.random import default_rng
rng=default_rng(0)
K=256
cent=flat[rng.choice(len(flat),K,replace=False)]
for it in range(25):
    dist=((flat[:,None,:]-cent[None,:,:])**2).sum(-1) if len(flat)<50000 else None
    if dist is None:
        idx=np.empty(len(flat),dtype=np.int32)
        for s in range(0,len(flat),20000):
            chunk=flat[s:s+20000]
            d2=((chunk[:,None,:]-cent[None,:,:])**2).sum(-1)
            idx[s:s+20000]=d2.argmin(1)
    else:
        idx=dist.argmin(1)
    for k in range(K):
        sel=flat[idx==k]
        if len(sel)>0: cent[k]=sel.mean(0)
    if it%5==0: print("  iter",it,flush=True)
codes=idx.reshape(len(M),40)
np.savez_compressed(ROOT/"data/esc_audio_tokens2.npz", codes=codes, cent=cent, cats=cats,
                    target=d["target"])
print("码表",cent.shape,"每条音频 40 个音频 token，示例",codes[0][:12])
