#!/usr/bin/env python3
"""60 个「可看 + 可画 + 可出声」的概念：分类器 + 覆盖表。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
# key, 中文, QuickDraw 类别或 None, ESC-50 类别或 None
TABLE=[
 ("dog","狗","dog","dog"),("cat","猫","cat","cat"),("bird","鸟","bird","chirping_birds"),
 ("fish","鱼","fish",None),("horse","马",None,None),("cow","牛",None,"cow"),
 ("sheep","羊",None,"sheep"),("hen","母鸡",None,"hen"),("pig","猪",None,"pig"),
 ("frog","青蛙",None,"frog"),("rooster","公鸡",None,"rooster"),("insect","昆虫",None,"insects"),
 ("car","汽车","car",None),("train","火车","train","train"),("airplane","飞机",None,"airplane"),
 ("boat","船",None,None),("bicycle","自行车",None,None),("motorcycle","摩托车",None,"engine"),
 ("tree","树","tree",None),("flower","花","flower",None),("grass","草地",None,None),
 ("mountain","山",None,None),("snow","雪",None,"wind"),("sand","沙",None,None),
 ("rock","岩石",None,None),("water","水",None,"water_drops"),("ocean","海",None,"sea_waves"),
 ("river","河",None,"pouring_water"),("pool","泳池",None,None),("rain","雨",None,"rain"),
 ("sky","天空",None,None),("sun","太阳","sun",None),("star","星星","star",None),
 ("cloud","云",None,None),("house","房子","house",None),("building","建筑",None,None),
 ("street","街道",None,"footsteps"),("bridge","桥",None,None),("fence","栅栏",None,None),
 ("window","窗户",None,None),("door","门",None,"door_wood_knock"),("wall","墙",None,None),
 ("table","桌子",None,None),("chair","椅子",None,None),("bed","床",None,"snoring"),
 ("ball","球",None,None),("bat","球棒",None,None),("racket","球拍",None,None),
 ("surfboard","冲浪板",None,None),("skateboard","滑板",None,None),("umbrella","雨伞",None,None),
 ("bench","长椅",None,None),("backpack","背包",None,None),("bottle","瓶子",None,None),
 ("cup","杯子",None,"drinking_sipping"),("camera","相机",None,None),("guitar","吉他",None,None),
 ("phone","手机",None,"clock_tick"),("clock","时钟",None,"clock_alarm"),("fire","火",None,"crackling_fire"),
]
names=[t[0] for t in TABLE]
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
ALIAS={"insect":["insect","bug","bee"],"bicycle":["bicycle","bike"],"ocean":["ocean","sea"],
       "car":["car","truck"],"hen":["hen","chicken"],"rock":["rock","stone"],"boat":["boat","canoe"]}
w2i={w:i for i,w in enumerate(names)}
Y=torch.zeros(len(CAPS),len(names))
for k,c in enumerate(CAPS):
    ws=set(re.findall(r"[a-z]+",c.lower()))
    for w,al in ALIAS.items():
        if ws & set(al): Y[k,w2i[w]]=1
    for w in ws:
        if w in w2i: Y[k,w2i[w]]=1
print("概念",len(names),"正样本",int(Y.sum()),"平均每句 %.2f"%float(Y.sum(1).mean()))
class Clf(nn.Module):
    def __init__(self,d=512,h=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.1),
                               nn.Linear(h,h),nn.GELU(),nn.Linear(h,len(names)))
    def forward(self,x): return self.net(x)
m=Clf(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
nimg=len(FE0); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img=set(perm[:ntr].tolist()); te_img=set(perm[ntr:].tolist())
tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]
te=[i for i in range(len(CAPS)) if int(IMG[i]) in te_img]
for step in range(1,4001):
    i=torch.tensor(np.random.choice(tr,256))
    loss=F.binary_cross_entropy_with_logits(m(FE0[IMG[i]]), Y[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%800==0: print(f"step {step} bce {loss.item():.4f}", flush=True)
m.eval(); hit=n=0; pn=pd=0
with torch.no_grad():
    for i in te[:2000]:
        p=torch.sigmoid(m(FE0[IMG[i]:IMG[i]+1]))[0]
        gold=Y[i].nonzero().flatten().tolist()
        if not gold: continue
        n+=1; pred=p.argmax().item(); hit+=int(pred in gold)
        for j in (p>0.5).nonzero().flatten().tolist():
            pd+=1; pn+=int(j in gold)
print(f"top-1 命中率 {hit/max(n,1)*100:.0f}%（{n} 张，随机 {100/len(names):.1f}%） 精确率 {pn/max(pd,1)*100:.0f}%")
torch.save({"model":m.state_dict(),"names":names,"table":TABLE}, ROOT/"ckpt/concept60.pt")
cov=[t for t in TABLE if t[2] and t[3]]
print(f"三路俱全（能画+出声）：{len(cov)} 个；能画：{sum(1 for t in TABLE if t[2])}；能出声：{sum(1 for t in TABLE if t[3])}")
