#!/usr/bin/env python3
"""用已验证 75.4% 的 CNN 分类器提取声音语义向量（去掉分类头）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path(r"C:\work"); DEV="cuda"
# 载入已验证的分类器结构
class Clf(nn.Module):
    def __init__(self,NC=50):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.GELU(),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Dropout(0.3),nn.Linear(256,NC))
    def forward(self,x): return self.net(x.unsqueeze(1))
    def features(self,x):
        h=x.unsqueeze(1)
        for layer in list(self.net)[:-3]:      # 到 AdaptiveAvgPool 为止，去掉 Flatten/Dropout/Linear
            h=layer(h)
        return h.flatten(1)                    # (B,256)
m=Clf(); m.load_state_dict(torch.load(ROOT/"mel_clf.pt",map_location="cpu")["model"]); m=m.to(DEV).eval()
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
X=torch.from_numpy(d["mels"]).float(); Y=torch.from_numpy(d["target"]).long()
print("mel",tuple(X.shape),flush=True)
with torch.no_grad():
    feats=[]; logits=[]
    for s0 in range(0,len(X),200):
        xb=X[s0:s0+200].to(DEV)
        feats.append(m.features(xb)); logits.append(m(xb))
    feat=torch.cat(feats); logits=torch.cat(logits)
acc=(logits.argmax(-1)==Y.to(DEV)).float().mean().item()
print(f"原始 mel 全量分类准确率 {acc*100:.1f}%",flush=True)
print(f"语义向量维度 {feat.shape}（从 80×128=10240 维压到 256 维）",flush=True)
# 保存特征与码（用 k-means 便于离散化测试）
np.savez_compressed(ROOT/"data/esc_semvec.npz", vec=feat.cpu().numpy(),
                    target=d["target"], cats=d["cats"])
# 量化后分类准确率（检验 256 维连续向量的可分性）
f=feat.cpu().numpy(); y=d["target"]
np.random.seed(0); idx=np.random.permutation(len(f)); ntr=int(len(f)*0.8)
tr,te=idx[:ntr],idx[ntr:]
# 简单线性分类器检验可分性（闭式最小二乘）
Xt=np.hstack([f[tr],np.ones((len(tr),1))]); Ytr=np.eye(50)[y[tr]]
W=np.linalg.solve(Xt.T@Xt+1e-3*np.eye(Xt.shape[1]),Xt.T@Ytr)
Xe=np.hstack([f[te],np.ones((len(te),1))])
acc2=float(((Xe@W).argmax(1)==y[te]).mean())
print(f"256 维语义向量 → 线性分类准确率 {acc2*100:.1f}%",flush=True)
