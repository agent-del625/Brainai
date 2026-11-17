#!/usr/bin/env python3
"""场景合成：区域背景（天空/水/地面）+ 物体贴片（来自概念图库）。

布局 = 背景区域 + 物体框；边界由区域交界与贴片轮廓共同决定。
"""
from pathlib import Path
import numpy as np, torch, torch.nn as nn, cv2
ROOT = Path("/srv/brainai")
W = 256

# 背景：区域模型
ck = torch.load(ROOT/"ckpt/region.pt", map_location="cpu", weights_only=False)
cent, K = ck["cent"], ck["cent"].shape[0]
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

def background(concepts):
    ids = [c for c in concepts if c in cfeat]
    if not ids:
        return np.full((W,W,3), 180, np.uint8)
    with torch.no_grad():
        lab = m(torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)).argmax(1)[0].numpy()
    rgb = (cent[lab].numpy()*255).clip(0,255).astype(np.uint8)
    return cv2.resize(rgb, (W,W), interpolation=cv2.INTER_LINEAR)

def sticker(concept, size):
    gal = sorted((ROOT/"data/gallery").rglob(f"{concept}_*.jpg"))
    if not gal: return None
    im = cv2.imread(str(gal[0])); im = cv2.resize(im, (size,size))
    h, w = im.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    cv2.ellipse(mask, (w//2, h//2), (int(w*0.42), int(h*0.42)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31,31), 0).astype(np.float32)[...,None]/255.0
    return im.astype(np.float32), mask

OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
slots = [(0.30, 0.62), (0.70, 0.55)]     # (x, y) 归一化位置
tiles = []
for combo in combos:
    bg = background([c for c in combo if c in cfeat]).astype(np.float32)
    objs = [c for c in combo if sorted((ROOT/"data/gallery").rglob(f"{c}_*.jpg"))]
    for i, c in enumerate(objs[:2]):
        st = sticker(c, int(W*0.42))
        if st is None: continue
        sim, mask = st
        x, y = slots[i]
        cx, cy = int(x*W), int(y*W); half = sim.shape[0]//2
        x0, y0 = max(0, cx-half), max(0, cy-half)
        x1, y1 = min(W, x0+sim.shape[0]), min(W, y0+sim.shape[1])
        sub = sim[:y1-y0, :x1-x0]; msub = mask[:y1-y0, :x1-x0]
        bg[y0:y1, x0:x1] = bg[y0:y1, x0:x1]*(1-msub) + sub*msub
    out = bg.clip(0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/f"scene_{'_'.join(combo)}.png"), out)
    tiles.append(out)
cv2.imwrite(str(OUT/"scenes.png"), np.hstack(tiles))
print("图:", OUT/"scenes.png")
