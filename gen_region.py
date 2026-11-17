#!/usr/bin/env python3
"""区域式生成：先把画面量化成 K 个"色区"（天空/水/植被/地面…），
   再用模型预测每个格子的区域类别，最后用区域色填充 → 干净的色块图。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = torch.from_numpy(imgs).float()/255.0
N = len(imgs); G = 16

# 1) 在像素上做 k-means（带一点纵向位置信息，让"天空"倾向在上）
pix = imgs.permute(0,2,3,1).reshape(-1,3).numpy()
idx = np.random.choice(len(pix), 40000, replace=False)
sample = pix[idx]
per_img = 64*64
ys = (np.arange(len(pix)) % per_img // 64) / 63.0
ys = ys[idx][:, None]
X = np.concatenate([sample, 0.15*ys], 1)
K = 6
rng = np.random.default_rng(0); cent = X[rng.choice(len(X), K, replace=False)]
for _ in range(25):
    d = ((X[:,None,:]-cent[None])**2).sum(-1); lab = d.argmin(1)
    for k in range(K):
        if (lab==k).sum()>0: cent[k] = X[lab==k].mean(0)
cent_rgb = torch.tensor(cent[:,:3], dtype=torch.float32)
print("色区中心:", cent_rgb.numpy().round(2))

# 2) 每张图 → 16×16 区域标签
low = F.adaptive_avg_pool2d(imgs, G).permute(0,2,3,1)          # (N,16,16,3)
dist = ((low[:,:, :,None,:] - cent_rgb[None,None,None])**2).sum(-1)
labels = dist.argmin(-1)                                        # (N,16,16)
print("标签分布:", np.bincount(labels.numpy().ravel(), minlength=K)/labels.numel())

class Region(nn.Module):
    def __init__(self, d=512, ch=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d,512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, ch*2*2))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch,96,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(96,64,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,K,3,padding=1))
    def forward(self, z):
        return self.net(self.fc(z).view(-1,128,2,2))

m = Region(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for step in range(1, 1501):
    i = torch.randint(0, N, (32,))
    logits = m(feats[i])                     # (B,K,16,16)
    loss = F.cross_entropy(logits, labels[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        acc = (logits.argmax(1) == labels[i]).float().mean().item()*100
        print(f"step {step} ce {loss.item():.3f} 格点准确率 {acc:.0f}%")

cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
with torch.no_grad():
    for combo in combos:
        ids = [c for c in combo if c in cfeat]
        zz = torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)
        lab = m(zz).argmax(1)[0]                       # (16,16)
        rgb = cent_rgb[lab].numpy()
        big = cv2.resize((rgb*255).clip(0,255).astype(np.uint8), (192,192), interpolation=cv2.INTER_NEAREST)
        tiles.append(big)
cv2.imwrite(str(OUT/"regions.png"), np.hstack(tiles))
print("图:", OUT/"regions.png")
torch.save({"model": m.state_dict(), "cent": cent_rgb}, ROOT/"ckpt/region.pt")
