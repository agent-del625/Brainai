#!/usr/bin/env python3
"""图像生成器升级：32×32 → 64×64，并加梯度（边缘）损失，避免只生成平均色。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = torch.from_numpy(imgs).float()/255.0
print("图像", imgs.shape)

def grad(x):
    gx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    gy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return gx + gy

class ImgGen(nn.Module):
    def __init__(self, d=512, ch=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, ch*4*4))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch, 192, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(48, 3, 4, 2, 1))
    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return torch.sigmoid(self.net(h))

g = ImgGen(); opt = torch.optim.AdamW(g.parameters(), lr=2e-3)
for step in range(1, 2001):
    idx = torch.randint(0, len(feats), (32,))
    out = g(feats[idx]); tgt = imgs[idx]
    loss = F.mse_loss(out, tgt) + 0.5*F.l1_loss(grad(out), grad(tgt))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"step {step} mse {F.mse_loss(out,tgt).item():.4f} edge {grad(out).item():.4f}/{grad(tgt).item():.4f}")

cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
with torch.no_grad():
    for combo in combos:
        ids = [c for c in combo if c in cfeat]
        zz = torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)
        im = (g(zz)[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        tiles.append(cv2.resize(im, (192,192), interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(OUT/f"gen_{'_'.join(ids)}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/"generated64.png"), np.hstack(tiles))
print("图:", OUT/"generated64.png")
torch.save({"img": g.state_dict()}, ROOT/"ckpt/img64.pt")
