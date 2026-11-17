#!/usr/bin/env python3
"""用已训练的 region 模型渲染（双线性上采样 + 轻微模糊，让边界自然）。"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, cv2
ROOT = Path("/srv/brainai")
ck = torch.load(ROOT/"ckpt/region.pt", map_location="cpu", weights_only=False)
cent = ck["cent"]; K = cent.shape[0]
class Region(nn.Module):
    def __init__(self, d=512, ch=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d,512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, ch*2*2))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch,96,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(96,64,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,K,3,padding=1))
    def forward(self, z): return self.net(self.fc(z).view(-1,128,2,2))
m = Region(); m.load_state_dict(ck["model"]); m.eval()
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
OUT = ROOT/"logs/gen"
tiles = []
for combo in [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]:
    ids = [c for c in combo if c in cfeat]
    with torch.no_grad():
        lab = m(torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)).argmax(1)[0].numpy()
    rgb = (cent[lab].numpy()*255).clip(0,255).astype(np.uint8)
    big = cv2.resize(rgb, (192,192), interpolation=cv2.INTER_LINEAR)
    tiles.append(big)
cv2.imwrite(str(OUT/"regions_smooth.png"), np.hstack(tiles))
print("图:", OUT/"regions_smooth.png")
