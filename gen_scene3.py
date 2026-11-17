#!/usr/bin/env python3
"""场景合成 v3：预训练 sketch ONNX 做线稿 + 检索取色背景 + 数据决定槽位。"""
from pathlib import Path
import numpy as np, torch, cv2
import onnxruntime as ort
ROOT = Path("/srv/brainai"); W = 256

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
low = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(imgs).float()/255.0, 16)
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cfeat = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}

sk = ort.InferenceSession(str(ROOT/"data/downloads/sketch.onnx"), providers=["CPUExecutionProvider"])
si, so = sk.get_inputs()[0].name, sk.get_outputs()[0].name

def to_sketch(img_bgr, size=256):
    x = cv2.resize(img_bgr, (256,256)).astype(np.float32)/255.0
    x = (x - 0.5)/0.5
    out = sk.run([so], {si: x.transpose(2,0,1)[None]})[0][0,0]
    out = (out - out.min())/(out.max()-out.min()+1e-6)
    if out.mean() > 0.5:          # 多数像素是亮的 → 黑底白线，取反成"线条强度"
        out = 1 - out
    lines = (out*255).clip(0,255).astype(np.uint8)     # 线条强度（0=无线，255=线）
    return cv2.resize(lines, (size,size)) if size != 256 else lines

def background(concepts, k=5):
    ids = [c for c in concepts if c in cfeat]
    if not ids: return np.full((W,W,3), 180, np.uint8)
    zz = torch.stack([cfeat[c] for c in ids]).mean(0)
    sim = torch.nn.functional.normalize(feats, dim=-1) @ torch.nn.functional.normalize(zz, dim=-1)
    avg = low[sim.topk(k).indices].mean(0).permute(1,2,0).numpy()
    return cv2.resize((avg*255).clip(0,255).astype(np.uint8), (W,W), interpolation=cv2.INTER_LINEAR)

def sticker(concept, size):
    gal = sorted((ROOT/"data/gallery").rglob(f"{concept}_*.jpg"))
    if not gal: return None, None
    lines = to_sketch(cv2.imread(str(gal[0])), size)
    h,w = lines.shape
    ellipse = np.zeros((h,w), np.float32)
    cv2.ellipse(ellipse, (w//2,h//2), (int(w*0.46),int(h*0.46)), 0,0,360,1.0,-1)
    ellipse = cv2.GaussianBlur(ellipse,(21,21),0)
    alpha = (lines.astype(np.float32)/255.0) * ellipse       # 只画线，不盖背景
    return alpha, alpha

OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
tiles = []
for combo in [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]:
    canvas = background([c for c in combo if c in cfeat]).astype(np.float32)
    for i, c in enumerate(combo):
        s, mask = sticker(c, int(W*0.5))
        if s is None: continue
        h,w = s.shape
        cx = int(W*(0.32 if i==0 else 0.7)); cy = int(W*0.55)
        x0,y0 = max(0,cx-w//2), max(0,cy-h//2); x1,y1 = min(W,x0+w), min(W,y0+h)
        msub = mask[y0-(cy-h//2):y1-(cy-h//2), x0-(cx-w//2):x1-(cx-w//2)]
        reg = canvas[y0:y1, x0:x1]
        for ch in range(3):
            reg[:,:,ch] = reg[:,:,ch]*(1-msub) + 0.0*msub      # 线条画成黑色
        canvas[y0:y1, x0:x1] = reg
    img = canvas.clip(0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/f"scene3_{'_'.join(combo)}.png"), img)
    tiles.append(img)
cv2.imwrite(str(OUT/"scenes3.png"), np.hstack(tiles))
print("图:", OUT/"scenes3.png")
