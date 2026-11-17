#!/usr/bin/env python3
"""通感：看图出声、听声成像（概念级的跨模态生成，不是检索）。"""
import wave
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from audio_feat import mel_to_wav

ROOT = Path("/srv/brainai")
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = [str(c) for c in ep["cats"]], ep["mels"]
ck = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
names = [str(c) for c in ck["names"]]
pairs = [(n, emels[ecats.index(n)]) for n in names if n in ecats]
print("通感训练对", len(pairs))

imgs = {}   # 概念 → CLIP 原型
for c, f in zip([str(x) for x in cp["cats"]], cp["feats"]):
    imgs[c] = torch.tensor(f).float()
X = torch.stack([imgs[c] for c, _ in pairs])                  # (N,512)
Y = torch.tensor(np.stack([m for _, m in pairs])).float()      # (N,1,80,128)

class Img2Sound(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(512,512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512,96*5*8))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(96,64,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(32,16,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(16,1,4,2,1))
    def forward(self, z):
        h = self.fc(z).view(-1,96,5,8)
        return F.adaptive_avg_pool2d(torch.sigmoid(self.net(h)), (80,128))

class Sound2Img(nn.Module):
    """声音 → 16×16 区域布局 → 上色块（复用 6 色区概念）。"""
    def __init__(self, K=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,24,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(24,48,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48,96,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(96*10*16, 512), nn.GELU(), nn.Linear(512, 16*16))
    def forward(self, m):
        return self.net(m).view(-1,16,16)

g1 = Img2Sound(); o1 = torch.optim.AdamW(g1.parameters(), lr=2e-3)
for step in range(1, 801):
    out = g1(X); loss = F.mse_loss(out, Y)
    o1.zero_grad(); loss.backward(); o1.step()
    if step % 200 == 0: print(f"[图→声] step {step} mse {loss.item():.4f}")

# 声音 → 图像：目标用该概念所对应真实图的 16×16 颜色
g2 = Sound2Img(); o2 = torch.optim.AdamW(g2.parameters(), lr=2e-3)
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs_all = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
low = torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(imgs_all).float()/255.0, 16)
# 每个概念用"最相似图片"的低分辨率颜色当目标（灰度化后作为形状代理）
tgt_maps = []
for c, _ in pairs:
    sim = torch.nn.functional.normalize(feats,dim=-1) @ torch.nn.functional.normalize(imgs[c],dim=-1)
    tgt_maps.append(low[sim.argmax()].mean(0))      # (16,16) 亮度图
T = torch.stack(tgt_maps).float()
for step in range(1, 801):
    out = g2(Y); loss = F.mse_loss(out, T)
    o2.zero_grad(); loss.backward(); o2.step()
    if step % 200 == 0: print(f"[声→图] step {step} mse {loss.item():.4f}")

OUT = ROOT/"logs/crossmodal"; OUT.mkdir(parents=True, exist_ok=True)
tiles = []
with torch.no_grad():
    for c, m in pairs[:4]:
        mel = g1(imgs[c][None])[0,0].numpy()
        wav = mel_to_wav(mel)
        with wave.open(str(OUT/f"{c}_seen.wav"), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(wav.tobytes())
        img = g2(torch.tensor(m)[None])[0].numpy()
        vis = (img*255).clip(0,255).astype(np.uint8)
        tiles.append(cv2.resize(vis, (128,128), interpolation=cv2.INTER_NEAREST))
        print(f"  {c}: 看了照片→生成声音 {c}_seen.wav；听了声音→生成画面 {vis.shape}")
cv2.imwrite(str(OUT/"heard_images.png"), np.hstack(tiles))
print("图:", OUT/"heard_images.png")
