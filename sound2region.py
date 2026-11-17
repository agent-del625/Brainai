#!/usr/bin/env python3
"""通感（改进）：听声成像——声音 → 6 色区布局 → 彩色画面。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai")

rk = torch.load(ROOT/"ckpt/region.pt", map_location="cpu", weights_only=False)
cent, K = rk["cent"], rk["cent"].shape[0]          # 6 个色区中心
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = [str(c) for c in ep["cats"]], ep["mels"]
ck = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
names = [str(c) for c in ck["names"]]
pairs = [(n, emels[ecats.index(n)]) for n in names if n in ecats]
imgs = {str(c): torch.tensor(f).float() for c, f in zip(cp["cats"], cp["feats"])}
print("概念", len(pairs))

# 目标：该概念对应真实图的 16×16 色区标签（用 region 模型的色区中心量化）
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
raw = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
low = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(raw).float()/255.0, 16).permute(0,2,3,1)
labels = []
for c, _ in pairs:
    sim = torch.nn.functional.normalize(feats,dim=-1) @ torch.nn.functional.normalize(imgs[c],dim=-1)
    img = low[sim.argmax()]                                   # (16,16,3)
    d = ((img[:,:,None,:] - cent[None,None])**2).sum(-1)
    labels.append(d.argmin(-1))
T = torch.stack(labels)                                        # (N,16,16)
print("目标标签分布", np.bincount(T.numpy().ravel(), minlength=K)/T.numel())

class S2R(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,24,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(96*10*16, 512), nn.GELU(), nn.Linear(512, 16*16*K))
    def forward(self,m): return self.net(m).view(-1,K,16,16)

m = S2R(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
Y = torch.tensor(np.stack([em for _, em in pairs])).float()
for step in range(1, 1201):
    logits = m(Y); loss = F.cross_entropy(logits, T)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0:
        acc = (logits.argmax(1)==T).float().mean().item()*100
        print(f"step {step} ce {loss.item():.3f} 区域准确率 {acc:.0f}%", flush=True)

OUT = ROOT/"logs/crossmodal"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
with torch.no_grad():
    for c, em in pairs[:4]:
        lab = m(torch.tensor(em)[None]).argmax(1)[0].numpy()
        rgb = (cent[lab].numpy()*255).clip(0,255).astype(np.uint8)
        tiles.append(cv2.resize(rgb,(128,128),interpolation=cv2.INTER_NEAREST))
        print(f"  {c}: 听到声音 → 生成画面")
cv2.imwrite(str(OUT/"sound2region.png"), np.hstack(tiles))
print("图:", OUT/"sound2region.png")
