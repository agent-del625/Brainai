#!/usr/bin/env python3
"""中文长描述 v2：色区构成 + 概念（带阈值）+ 语法模板。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai")

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float()
raw = np.concatenate([np.load(f, allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])
imgs = raw.astype(np.float32)/255.0

# 色区模型（已训练）
rk = torch.load(ROOT/"ckpt/region.pt", map_location="cpu", weights_only=False)
cent, K = rk["cent"], rk["cent"].shape[0]
class Region(nn.Module):
    def __init__(self, d=512, ch=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d,512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, ch*2*2))
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch,96,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(96,64,3,padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64,K,3,padding=1))
    def forward(self, z): return self.net(self.fc(z).view(-1,128,2,2))
reg = Region(); reg.load_state_dict(rk["model"]); reg.eval()

# 给色区中心起中文名（按颜色）
def region_name(c):
    r,g,b = [float(x) for x in c]
    if r>0.55 and g>0.5 and b>0.45: return "明亮的天空"
    if b>r+0.08 and b>0.35: return "蓝色的水面或天空"
    if g>r+0.03 and g>b+0.03: return "绿色的植被"
    if r>g+0.05 and r>b+0.05: return "偏暖的地面"
    if max(r,g,b) < 0.3: return "较暗的区域"
    return "浅色的地面或天空"
RN = [region_name(c) for c in cent.numpy()]
print("色区命名:", RN)

cp = np.load(ROOT/"data/concept_all.npz", allow_pickle=True)
cnames = [str(c) for c in cp["cats"]]; cfeat = torch.tensor(cp["feats"]).float()
CN = {"dog":"一只狗","cat":"一只猫","bird":"一只鸟","train":"一列火车","airplane":"一架飞机",
      "frog":"一只青蛙","cow":"一头牛","sheep":"一只羊","hen":"一只母鸡","pig":"一头猪",
      "rooster":"一只公鸡","insects":"一群昆虫","chainsaw":"一把电锯","clock_alarm":"一个闹钟",
      "keyboard_typing":"一副键盘","church_bells":"一座教堂"}
PLACE = {"beach":"一片海滩","ocean":"一片大海","river":"一条河流","waterfall":"一道瀑布",
         "forest":"一片森林","rain":"一场雨","crickets":"一片草丛","chirping_birds":"鸟鸣声传来的地方"}

def caption(i, thr=0.26):
    f = feats[i:i+1]
    sim = (F.normalize(cfeat,dim=-1) @ F.normalize(f[0],dim=-1))
    top = sim.topk(3)
    with torch.no_grad():
        lab = reg(f).argmax(1)[0].numpy()          # (16,16)
    frac = np.bincount(lab.ravel(), minlength=K)/lab.size
    order = frac.argsort()[::-1]
    sky_row = lab[:4].ravel(); ground_row = lab[8:].ravel()
    top_name = RN[np.bincount(sky_row, minlength=K).argmax()]
    bot_name = RN[np.bincount(ground_row, minlength=K).argmax()]

    parts = []
    obj = next((CN[cnames[j]] for j,v in zip(top.indices.tolist(), top.values.tolist())
                if v > thr and cnames[j] in CN), None)
    place = next((PLACE[cnames[j]] for j,v in zip(top.indices.tolist(), top.values.tolist())
                  if v > 0.22 and cnames[j] in PLACE), None)
    if obj: parts.append(f"画面中{obj}")
    if place: parts.append(f"整体像是{place}")
    parts.append(f"上方是{top_name}，下方是{bot_name}")
    return "，".join(parts) + "。"

OUT = ROOT/"logs/cn"; OUT.mkdir(parents=True, exist_ok=True)
tiles=[]
for k in range(6):
    i = k*37
    print("  " + caption(i))
    tiles.append(cv2.cvtColor(cv2.resize(raw[i].transpose(1,2,0), (128,128)), cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/"cn2.png"), np.hstack(tiles))
print("图:", OUT/"cn2.png")
