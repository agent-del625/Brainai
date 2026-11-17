#!/usr/bin/env python3
"""用已验证 CNN 编码器给增强后的 10000 条声音提取 256 维语义向量。"""
import numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT=Path("/srv/brainai"); DEV="cpu"
class Clf(nn.Module):
    def __init__(self,NC=50):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.GELU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Dropout(0.3),nn.Linear(256,NC))
    def features(self,x):
        h=x.unsqueeze(1)
        for layer in list(self.net)[:-3]: h=layer(h)
        return h.flatten(1)
m=Clf(); m.load_state_dict(torch.load("/tmp/mel_clf.pt",map_location="cpu")["model"]); m.eval()
d=np.load(ROOT/"data/esc_mels_aug.npz",allow_pickle=True)
X=torch.from_numpy(d["mels"]).float()
feats=[]
with torch.no_grad():
    for s in range(0,len(X),200): feats.append(m.features(X[s:s+200]))
F=torch.cat(feats).numpy()
print("语义向量",F.shape)
np.savez_compressed(ROOT/"data/esc_semvec_aug.npz", vec=F, target=d["target"], cats=d["cats"])
print("已存 esc_semvec_aug.npz")
