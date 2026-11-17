#!/usr/bin/env python3
"""布局式生成：先预测 16×16 的"色块地图"，再放大成图。

和之前的解码器区别：不做像素级回归（会糊），而是先定结构（哪块是天、哪块是水/地），
边界由色块交界自然产生。
"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2

ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = torch.from_numpy(imgs).float()/255.0
G = 16
low = F.adaptive_avg_pool2d(imgs, G)      # (N,3,16,16) 目标色块图
print("图", imgs.shape, "→ 布局", low.shape)

class Layout(nn.Module):
    def __init__(self, d=512, ch=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d,512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, ch*2*2))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch,96,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(96,64,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,3,3,padding=1))
    def forward(self, z):
        h = self.fc(z).view(-1,128,2,2)
        return torch.sigmoid(self.net(h))

m = Layout(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for step in range(1, 1501):
    idx = torch.randint(0, len(feats), (32,))
    out = m(feats[idx]); tgt = low[idx]
    # 色块损失 + 边界一致性（相邻格差异应与真图一致）
    dx_o = (out[:,:,:,1:]-out[:,:,:,:-1]).abs().mean(); dx_t = (tgt[:,:,:,1:]-tgt[:,:,:,:-1]).abs().mean()
    dy_o = (out[:,:,1:,:]-out[:,:,:-1,:]).abs().mean(); dy_t = (tgt[:,:,1:,:]-tgt[:,:,:-1,:]).abs().mean()
    loss = F.mse_loss(out, tgt) + 0.5*(F.l1_loss(dx_o, dx_t) + F.l1_loss(dy_o, dy_t))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 400 == 0:
        print(f"step {step} mse {F.mse_loss(out,tgt).item():.4f} edge {dx_o.item():.4f}/{dx_t.item():.4f}")

cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
with torch.no_grad():
    for combo in combos:
        ids = [c for c in combo if c in cfeat]
        zz = torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)
        im = (m(zz)[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
        big = cv2.resize(im, (192,192), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(OUT/f"layout_{'_'.join(ids)}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        tiles.append(big)
cv2.imwrite(str(OUT/"layouts.png"), np.hstack(tiles))
print("图:", OUT/"layouts.png")
torch.save({"model": m.state_dict()}, ROOT/"ckpt/layout.pt")
