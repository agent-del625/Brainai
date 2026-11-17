#!/usr/bin/env python3
"""场景合成 v2：取色真实化 + 简笔画贴片 + 位置由数据决定。"""
from pathlib import Path
import numpy as np, torch, cv2
ROOT = Path("/srv/brainai"); W = 256

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs_t = torch.from_numpy(imgs).float()/255.0
low = torch.nn.functional.adaptive_avg_pool2d(imgs_t, 16)     # (N,3,16,16) 真实取色
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}

def background(concepts, k=5):
    """用最相似的 K 张真实图的低分辨率颜色当背景（颜色真实、结构保留）。"""
    ids = [c for c in concepts if c in cfeat]
    if not ids: return np.full((W,W,3), 180, np.uint8)
    zz = torch.stack([cfeat[c] for c in ids]).mean(0)
    sim = (torch.nn.functional.normalize(feats, dim=-1) @ torch.nn.functional.normalize(zz, dim=-1))
    top = sim.topk(k).indices
    avg = low[top].mean(0).permute(1,2,0).numpy()
    return cv2.resize((avg*255).clip(0,255).astype(np.uint8), (W,W), interpolation=cv2.INTER_LINEAR)

def sketch(concept, size):
    gal = sorted((ROOT/"data/gallery").rglob(f"{concept}_*.jpg"))
    if not gal: return None, None
    im = cv2.imread(str(gal[0]))
    im = cv2.resize(im, (size,size))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 9, 75, 75)            # 先去噪，减少碎边
    g = cv2.GaussianBlur(g, (5,5), 0)
    e = cv2.Canny(g, 110, 220)                        # 只保留强边缘 → 细线
    e = cv2.dilate(e, np.ones((2,2), np.uint8), 1)
    line = 255 - e                                   # 白底黑线
    # 软遮罩：椭圆 + 边缘再软化
    h,w = line.shape
    mask = np.zeros((h,w), np.float32)
    cv2.ellipse(mask, (w//2,h//2), (int(w*0.46),int(h*0.46)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21,21), 0)[...,None]
    return line, mask

def slot_of(concept):
    """位置由数据决定：算概念图库的边缘密度重心。"""
    gal = sorted((ROOT/"data/gallery").rglob(f"{concept}_*.jpg"))[:3]
    if not gal: return (0.5, 0.6)
    cx = cy = wsum = 0.0
    for p in gal:
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, 60, 160).astype(np.float32)
        ys, xs = np.nonzero(e > 0)
        if len(xs) == 0: continue
        w_ = e.sum()
        cx += xs.mean()/g.shape[1]*w_; cy += ys.mean()/g.shape[0]*w_; wsum += w_
    if wsum == 0: return (0.5, 0.6)
    return (cx/wsum, max(0.35, min(0.8, cy/wsum)))

OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
for combo in combos:
    canvas = background([c for c in combo if c in cfeat]).astype(np.float32)
    for c in combo:
        line, mask = sketch(c, int(W*0.45))
        if line is None: continue
        cx, cy = slot_of(c)
        h, w = line.shape
        x0, y0 = int(cx*W - w/2), int(cy*W - h/2)
        x1, y1 = min(W, x0+w), min(W, y0+h)
        x0c, y0c = max(0, x0), max(0, y0)
        sub = line[y0c-y0:y1-y0, x0c-x0:x1-x0].astype(np.float32)
        msub = mask[y0c-y0:y1-y0, x0c-x0:x1-x0]
        region = canvas[y0c:y1, x0c:x1]
        for ch in range(3):
            region[:,:,ch] = region[:,:,ch]*(1-msub[:,:,0]) + sub*msub[:,:,0]
        canvas[y0c:y1, x0c:x1] = region
    out = canvas.clip(0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/f"scene2_{'_'.join(combo)}.png"), out)
    tiles.append(out)
cv2.imwrite(str(OUT/"scenes2.png"), np.hstack(tiles))
print("图:", OUT/"scenes2.png")
for c in ["dog","cat","train"]:
    print(f"  槽位 {c}: {tuple(round(v,2) for v in slot_of(c))}")
