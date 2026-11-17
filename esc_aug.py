#!/usr/bin/env python3
"""声音数据增强：时间平移 + 噪声 + 增益，把 2000 条扩到 10000 条（用于统一模型）。"""
import numpy as np
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"]; Y=d["target"]; CATS=[str(c) for c in d["cats"]]
out=[]; yy=[]; cc=[]
rng=np.random.default_rng(0)
for i,m in enumerate(M):
    out.append(m); yy.append(Y[i]); cc.append(CATS[i])
    for k in range(4):                       # 每条再生成 4 个变体
        v=m.copy()
        sh=int(rng.integers(-12,13)); v=np.roll(v,sh,axis=1)
        if k%2==0:
            v=v+rng.normal(0,0.02,v.shape).astype(np.float32)
        v=v*(0.85+0.3*rng.random())
        out.append(v.astype(np.float32)); yy.append(Y[i]); cc.append(CATS[i])
M2=np.stack(out)
print("增强后",M2.shape)
np.savez_compressed(ROOT/"data/esc_mels_aug.npz", mels=M2, target=np.array(yy), cats=np.array(cc))
print("已存 esc_mels_aug.npz")
