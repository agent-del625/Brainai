#!/usr/bin/env python3
"""简笔画能力：从概念直接"画"出边缘线条（不是像素回归，是笔画/轮廓）。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = torch.from_numpy(imgs).float()/255.0
print("图像", imgs.shape)

def edges(x):
    """灰度 Sobel 幅度 → 归一化边缘图（作为简笔画监督）。"""
    g = x.mean(1, keepdim=True)
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
    gx = F.conv2d(g, kx, padding=1); gy = F.conv2d(g, ky, padding=1)
    m = (gx**2 + gy**2).sqrt()
    m = m / (m.amax(dim=(2,3), keepdim=True) + 1e-6)
    return (m > 0.25).float()

class Sketcher(nn.Module):
    def __init__(self, d=512, ch=192):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d,512), nn.GELU(), nn.Linear(512, ch*4*4))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch,128,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128,64,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,32,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32,1,3,padding=1))
    def forward(self, z):
        h = self.fc(z).view(-1,192,4,4)
        return self.net(h)          # 输出 logits

m = Sketcher(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for step in range(1, 1201):
    idx = torch.randint(0, len(feats), (16,))
    tgt = edges(imgs[idx])
    out = m(feats[idx])
    bce = F.binary_cross_entropy_with_logits(out, tgt, pos_weight=torch.tensor(12.0))
    p = torch.sigmoid(out)
    dice = 1 - (2*(p*tgt).sum()+1) / (p.sum()+tgt.sum()+1)
    loss = bce + 0.5*dice
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0: print(f"step {step} bce {loss.item():.4f}")

cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
with torch.no_grad():
    for combo in combos:
        ids = [c for c in combo if c in cfeat]
        zz = torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)
        e = (torch.sigmoid(m(zz))[0,0].numpy()*255).astype(np.uint8)
        e = (e > 90).astype(np.uint8)*255          # 二值化 → 线稿
        e = 255 - e                                 # 黑线白底
        cv2.imwrite(str(OUT/f"sketch_{'_'.join(ids)}.png"), e)
        tiles.append(cv2.resize(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR), (192,192), interpolation=cv2.INTER_NEAREST))
cv2.imwrite(str(OUT/"sketches.png"), np.hstack(tiles))
print("图:", OUT/"sketches.png")
torch.save({"model": m.state_dict()}, ROOT/"ckpt/sketcher.pt")
