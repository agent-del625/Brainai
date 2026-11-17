#!/usr/bin/env python3
"""中文长描述 v4：概念 + 动作 + 地面 + 天空/天气，拼成一段完整描述。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, re
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
raw=np.concatenate([np.load(f,allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])

ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating"]
GRD=["grass","sand","water","road","snow","beach","field"]
SKY=["clouds","sky","sunny","night","tree","trees"]
CN_ACT={"running":"奔跑","jumping":"跳跃","sitting":"坐着","standing":"站着","walking":"行走",
        "playing":"玩耍","swimming":"游泳","riding":"骑行","holding":"拿着东西","eating":"吃东西"}
CN_GRD={"grass":"绿色的草坪","sand":"沙地","water":"水边","road":"道路","snow":"雪地","beach":"海滩","field":"田野"}
CN_SKY={"clouds":"蓝天白云","sky":"开阔的天空","sunny":"阳光明媚","night":"夜色","tree":"树木","trees":"树木"}

ALL = ACT + GRD + SKY

def lab(words):
    W=set(words); return [1.0 if w in W else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in caps])
print("动作/地面/天空 正样本:", LA.sum(0).int().tolist())

class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
m=Attr(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(1,2001):
    i=torch.randint(0,len(feats),(128,))
    loss=F.binary_cross_entropy_with_logits(m(feats[i]), LA[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"step {step} bce {loss.item():.4f}")
m.eval()

ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()
CN={"dog":"一只狗","cat":"一只猫","bird":"一只鸟","train":"一列火车","airplane":"一架飞机",
    "cow":"一头牛","sheep":"一只羊","hen":"一只母鸡","pig":"一头猪","frog":"一只青蛙",
    "beach":"海滩","ocean":"大海","river":"河流","waterfall":"瀑布","forest":"森林"}

def caption(i):
    f=feats[i:i+1]
    with torch.no_grad():
        pc=torch.sigmoid(clf(f))[0]; pa=torch.sigmoid(m(f))[0]
    obj_top=[j for j in pc.argsort(descending=True).tolist() if cnames[j] in CN][:1]
    obj = CN[cnames[obj_top[0]]] if obj_top else None
    def best(mapping, thr=0.0):
        for j in pa.argsort(descending=True).tolist():
            if ALL[j] in mapping and pa[j] > thr: return mapping[ALL[j]]
        return None
    act=best(CN_ACT); grd=best(CN_GRD); sky=best(CN_SKY)
    PLACES = {"海滩","大海","河流","瀑布","森林"}
    if obj in PLACES:
        s = f"这是一张{obj}的照片"
        if grd and grd not in obj: s += f"，画面下方是{grd}"
    else:
        s = f"这是一张照片：{obj or '画面中有一个主体'}"
        if act: s += f"正在{act}"
        if grd: s += f"，位于{grd}上"
    if sky: s += f"，背景是{sky}"
    return s + "。"

OUT=ROOT/"logs/cn"; OUT.mkdir(parents=True,exist_ok=True)
tiles=[]
for k in range(6):
    i=k*37
    print("  "+caption(i))
    tiles.append(cv2.cvtColor(cv2.resize(raw[i].transpose(1,2,0),(128,128)),cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/"cn4.png"),np.hstack(tiles))
print("图:",OUT/"cn4.png")
