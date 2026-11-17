#!/usr/bin/env python3
"""中文长描述 v5：主体-动作相容性 + 人数 + 时间。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2, re
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
raw=np.concatenate([np.load(f,allow_pickle=True)["images"] for f in sorted((ROOT/"data/image").glob("img_*.npz"))])

ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"]
SKY=["clouds","sky","sunny","night","trees","sunset"]
CNT=["one","two","three","group","crowd"]
CN_ACT={"running":"奔跑","jumping":"跳跃","sitting":"坐着","standing":"站着","walking":"行走",
        "playing":"玩耍","swimming":"游泳","riding":"骑行","holding":"拿着东西","eating":"吃东西",
        "driving":"行驶","flying":"飞行"}
CN_GRD={"grass":"绿色的草坪","sand":"沙地","water":"水边","road":"道路","snow":"雪地",
        "beach":"海滩","field":"田野","street":"街道"}
CN_SKY={"clouds":"蓝天白云","sky":"开阔的天空","sunny":"阳光明媚","night":"夜色","trees":"树木","sunset":"夕阳"}
CN_CNT={"one":"一个人","two":"两个人","three":"几个人","group":"一群人","crowd":"熙攘的人群"}

# 主体-动作相容性：从 caption 统计
CONCEPT_WORDS={"dog":"dog","cat":"cat","bird":"bird","train":"train","airplane":"airplane",
               "cow":"cow","sheep":"sheep","hen":"hen","pig":"pig","frog":"frog"}
compat=defaultdict(Counter)
for c in caps:
    ws=set(re.findall(r"[a-z']+",c.lower()))
    for cw in CONCEPT_WORDS.values():
        if cw in ws:
            for a in ACT:
                if a in ws: compat[cw][a]+=1
print("相容性示例 dog:", compat["dog"].most_common(4))

ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in caps])

class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
m=Attr(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(1,1501):
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
PLACES={"海滩","大海","河流","瀑布","森林"}

def pick(props, mapping, thr=0.10):
    for j in props.argsort(descending=True).tolist():
        if ALL[j] in mapping and props[j] > thr: return ALL[j], mapping[ALL[j]]
    return None, None

def caption(i):
    f=feats[i:i+1]
    with torch.no_grad():
        pc=torch.sigmoid(clf(f))[0]; pa=torch.sigmoid(m(f))[0]
    cw=None
    for j in pc.argsort(descending=True).tolist():
        if cnames[j] in CN: cw=cnames[j]; break
    obj=CN.get(cw)
    # 动作：概率 × 相容性
    act_word, act_cn = None, None
    if cw in CONCEPT_WORDS:
        best=None
        for a in ACT:
            j=ALL.index(a)
            bonus = 0.25 if compat[CONCEPT_WORDS[cw]].get(a,0) > 5 else 0.0
            sc = float(pa[j]) + bonus
            if best is None or sc > best[0]: best=(sc,a)
        if best: act_word=best[1]; act_cn=CN_ACT[act_word]
    else:
        act_word, act_cn = pick(pa, CN_ACT)
    _, grd_cn = pick(pa, CN_GRD)
    _, sky_cn = pick(pa, CN_SKY, 0.12)
    _, cnt_cn = pick(pa, CN_CNT, 0.12)

    if obj in PLACES:
        s=f"这是一张{obj}的照片"
        if grd_cn and grd_cn not in obj: s+=f"，画面下方是{grd_cn}"
    else:
        s=f"这是一张照片：{obj or '画面中有一个主体'}"
        if act_cn: s+=f"正在{act_cn}"
        if grd_cn: s+=f"，位于{grd_cn}上"
    if sky_cn: s+=f"，背景是{sky_cn}"
    if cnt_cn: s+=f"，远处能见到{cnt_cn}"
    return s+"。"

OUT=ROOT/"logs/cn"; OUT.mkdir(parents=True,exist_ok=True)
tiles=[]
for k in range(6):
    i=k*37
    print("  "+caption(i))
    tiles.append(cv2.cvtColor(cv2.resize(raw[i].transpose(1,2,0),(128,128)),cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/"cn5.png"),np.hstack(tiles))
print("图:",OUT/"cn5.png")
