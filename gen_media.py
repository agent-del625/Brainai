#!/usr/bin/env python3
"""从概念生成图像和声音（不是检索，是解码器生成）。"""
import wave
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import cv2

from audio_feat import mel_to_wav
ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
imgs = np.concatenate([np.load(f, allow_pickle=True)["images"]
                       for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = torch.from_numpy(imgs).float()/255.0
print("图像", imgs.shape, "特征", feats.shape)

class ImgGen(nn.Module):
    """CLIP 特征 → 32×32 RGB 图像。"""
    def __init__(self, d=512, ch=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, ch*4*4))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch, 96, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(48, 3, 4, 2, 1))
    def forward(self, z):
        h = self.fc(z).view(-1, 128, 4, 4)
        return torch.sigmoid(self.net(h))

g = ImgGen(); opt = torch.optim.AdamW(g.parameters(), lr=2e-3)
target = F.interpolate(imgs, size=(32,32), mode="bilinear")
for step in range(1, 1501):
    idx = torch.randint(0, len(feats), (32,))
    out = g(feats[idx]); loss = F.mse_loss(out, target[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0: print(f"[图像] step {step} mse {loss.item():.4f}")

# 声音生成：概念原型 → 梅尔谱
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = [str(c) for c in ep["cats"]], ep["mels"]
ck = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
names = ck["names"]
sub = [(n, emels[ecats.index(n)]) for n in names if n in ecats]
print("声音概念", len(sub))

class AudGen(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 96*5*8))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1))
    def forward(self, z):
        h = self.fc(z).view(-1, 96, 5, 8)
        return F.adaptive_avg_pool2d(torch.sigmoid(self.net(h)), (80, 128))

ag = AudGen(); opta = torch.optim.AdamW(ag.parameters(), lr=2e-3)
cfeat = {}
for c, f in zip([str(x) for x in cp["cats"]], cp["feats"]): cfeat[c] = torch.tensor(f).float()
X = torch.stack([cfeat[n] for n, _ in sub]); Y = torch.tensor(np.stack([m for _, m in sub])).float()
for step in range(1, 1501):
    out = ag(X); loss = F.mse_loss(out, Y)
    opta.zero_grad(); loss.backward(); opta.step()
    if step % 500 == 0: print(f"[声音] step {step} mse {loss.item():.4f}")

# 生成演示：组合概念 → 图像 + 声音
OUT = ROOT/"logs/gen"; OUT.mkdir(parents=True, exist_ok=True)
combos = [["beach","dog"], ["river","cat"], ["train"], ["forest","bird"]]
tiles = []
with torch.no_grad():
    for combo in combos:
        ids = [c for c in combo if c in cfeat]
        if not ids: continue
        zz = torch.stack([cfeat[c] for c in ids]).mean(0, keepdim=True)
        im = g(zz)[0].permute(1,2,0).numpy()
        tiles.append(cv2.resize((im*255).astype(np.uint8), (160,160), interpolation=cv2.INTER_NEAREST))
        mel = ag(zz)[0,0].numpy()
        wav = mel_to_wav(mel)
        p = OUT/f"{'_'.join(ids)}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(wav.tobytes())
        print(f"生成 {ids} → 图像+声音 {p.name}")
if tiles:
    cv2.imwrite(str(OUT/"generated.png"), np.hstack(tiles))
    print("图:", OUT/"generated.png")
torch.save({"img": g.state_dict(), "aud": ag.state_dict()}, ROOT/"ckpt/gen_media.pt")
