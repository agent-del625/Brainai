#!/usr/bin/env python3
"""中文长描述 v3：用 caption 监督的概念分类器 + 色区构成 + 语法模板。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai")

z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float()
raw=np.concatenate([np.load(f,allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])

ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384), nn.GELU(), nn.LayerNorm(384), nn.Dropout(0.1),
                               nn.Linear(384,192), nn.GELU(), nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()

rk=torch.load(ROOT/"ckpt/region.pt",map_location="cpu",weights_only=False)
cent,K=rk["cent"],rk["cent"].shape[0]
class Region(nn.Module):
    def __init__(self,d=512,ch=128):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(d,512),nn.GELU(),nn.LayerNorm(512),nn.Linear(512,ch*2*2))
        self.net=nn.Sequential(nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(ch,96,3,padding=1),nn.GELU(),
                               nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(96,64,3,padding=1),nn.GELU(),
                               nn.Upsample(scale_factor=2,mode="nearest"),nn.Conv2d(64,K,3,padding=1))
    def forward(self,z): return self.net(self.fc(z).view(-1,128,2,2))
reg=Region(); reg.load_state_dict(rk["model"]); reg.eval()

def rname(c):
    r,g,b=[float(x) for x in c]
    if r>0.55 and g>0.5 and b>0.45: return "明亮的天空"
    if b>r+0.08 and b>0.35: return "蓝色的水面"
    if g>r+0.03 and g>b+0.03: return "绿色的植被"
    if r>g+0.05 and r>b+0.05: return "偏暖的地面"
    if max(r,g,b)<0.3: return "较暗的区域"
    return "浅色的地面"
RN=[rname(c) for c in cent.numpy()]

CN={"dog":"一只狗","cat":"一只猫","bird":"一只鸟","train":"一列火车","airplane":"一架飞机",
    "frog":"一只青蛙","cow":"一头牛","sheep":"一只羊","hen":"一只母鸡","pig":"一头猪",
    "rooster":"一只公鸡","insects":"一些昆虫","chainsaw":"一把电锯","clock_alarm":"一个闹钟",
    "keyboard_typing":"一副键盘","church_bells":"一座教堂"}
PLACE={"beach":"海滩","ocean":"大海","river":"河流","waterfall":"瀑布","forest":"森林",
       "crickets":"草丛","chirping_birds":"林间"}

def caption(i):
    f=feats[i:i+1]
    with torch.no_grad():
        p=torch.sigmoid(clf(f))[0]
        lab=reg(f).argmax(1)[0].numpy()
    idx=p.argsort(descending=True).tolist()
    obj=next((CN[cnames[j]] for j in idx if cnames[j] in CN and p[j]>0.5), None)
    place=next((PLACE[cnames[j]] for j in idx if cnames[j] in PLACE and p[j]>0.35), None)
    top_r=lab[:5].ravel(); bot_r=lab[9:].ravel()
    tn=RN[np.bincount(top_r,minlength=K).argmax()]; bn=RN[np.bincount(bot_r,minlength=K).argmax()]
    parts=[]
    if obj: parts.append(f"画面主体是{obj}")
    if place: parts.append(f"场景看起来像{place}")
    parts.append(f"上方是{tn}，下方是{bn}")
    sent="，".join(parts)
    return f"这是一张照片：{sent}。"

OUT=ROOT/"logs/cn"; OUT.mkdir(parents=True,exist_ok=True)
tiles=[]
for k in range(6):
    i=k*37
    print("  "+caption(i))
    tiles.append(cv2.cvtColor(cv2.resize(raw[i].transpose(1,2,0),(128,128)),cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/"cn3.png"),np.hstack(tiles))
print("图:",OUT/"cn3.png")
