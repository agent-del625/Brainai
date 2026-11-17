#!/usr/bin/env python3
"""中文长描述：图像 → 概念 + 场景属性 → 组织成一段话（不是单词拼贴）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai")

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
raw = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = raw.astype(np.float32)/255.0
print("图像", imgs.shape)

# ---- 属性标签（用图像统计量当监督，便宜且客观）----
lum = imgs.mean(axis=(1,2,3))
bright = np.digitize(lum, [0.35, 0.6])                       # 0 暗 / 1 中 / 2 亮
top = imgs[:, :, :20].mean(axis=(1,2,3)); bot = imgs[:, :, 40:].mean(axis=(1,2,3))
sky_like = (top > bot + 0.08).astype(int)                     # 上亮下暗 → 像天空
rm = imgs[:,0].mean(axis=(1,2)); bm = imgs[:,2].mean(axis=(1,2)); gm = imgs[:,1].mean(axis=(1,2))
warm = (rm - bm) > 0.03
green = (gm > rm + 0.02) & (gm > bm + 0.02)
print("亮/中/暗", np.bincount(bright), "天空感", sky_like.sum(), "暖色", warm.sum(), "绿意", green.sum())

class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(512,384), nn.GELU(), nn.LayerNorm(384))
        self.b = nn.Linear(384,3); self.s = nn.Linear(384,2); self.w = nn.Linear(384,2); self.g = nn.Linear(384,2)
    def forward(self,z):
        h = self.trunk(z)
        return self.b(h), self.s(h), self.w(h), self.g(h)

m = Heads(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
B=torch.tensor(bright); S=torch.tensor(sky_like); W=torch.tensor(warm.astype(int)); G=torch.tensor(green.astype(int))
for step in range(1, 1201):
    i = torch.randint(0,len(feats),(64,))
    lb,ls,lw,lg = m(feats[i])
    loss = (F.cross_entropy(lb,B[i]) + F.cross_entropy(ls,S[i]) + F.cross_entropy(lw,W[i]) + F.cross_entropy(lg,G[i]))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0: print(f"step {step} loss {loss.item():.3f}")

# ---- 概念识别（用之前的 23 概念原型）----
cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cnames = [str(c) for c in cp["cats"]]; cfeat = torch.tensor(cp["feats"]).float()
CN = {"dog":"一只狗","cat":"一只猫","bird":"一只鸟","train":"一列火车","airplane":"一架飞机",
      "frog":"一只青蛙","cow":"一头牛","sheep":"一只羊","hen":"一只母鸡","pig":"一头猪",
      "rooster":"一只公鸡","insects":"一群昆虫","chainsaw":"一把电锯","clock_alarm":"一个闹钟",
      "keyboard_typing":"一副键盘","church_bells":"一座教堂","rain":"一场雨","beach":"一片海滩",
      "ocean":"一片大海","river":"一条河流","waterfall":"一道瀑布","forest":"一片森林",
      "crickets":"一片草丛","chirping_birds":"一片鸟鸣"}

OUT = ROOT/"logs/cn"; OUT.mkdir(parents=True, exist_ok=True)
m.eval(); tiles=[]
with torch.no_grad():
    for k in range(6):
        i = k*37
        f = feats[i:i+1]
        lb,ls,lw,lg = m(f)
        b = int(lb.argmax()); s = int(ls.argmax()); w = int(lw.argmax()); g = int(lg.argmax())
        sim = (F.normalize(cfeat,dim=-1) @ F.normalize(f[0],dim=-1))
        top3 = [cnames[j] for j in sim.topk(3).indices.tolist()]
        obj = CN.get(top3[0], top3[0])
        scene = "绿色的草坪" if g else ("开阔的场地" if s else "室内空间")
        light = ["光线偏暗","光线柔和","阳光明媚"][b]
        sky = "，远处是蓝天白云" if s else ""
        extra = "，画面整体偏暖" if w else ""
        sent = f"这是一张照片，{obj}位于{scene}上，{light}{sky}{extra}。"
        print(f"  {sent}")
        tiles.append(cv2.resize(raw[i].transpose(1,2,0), (128,128), interpolation=cv2.INTER_NEAREST))
cv2.imwrite(str(OUT/"cn_samples.png"), np.hstack([cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in tiles]))
print("图:", OUT/"cn_samples.png")
