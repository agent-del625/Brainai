#!/usr/bin/env python3
"""用结构化标签合成中文描述（多样句式），供中文生成模型训练。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random
from pathlib import Path
import sys; sys.path.insert(0,"/srv/brainai")
from zh_words import ZH as ZH_FULL
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
ZH=ZH_FULL
def zh(w): return ZH.get(w, w)
T1=["照片里{s}正在{a}。","{s}在{g}上{a}。","这是{s}，正在{a}，背景是{g}。",
    "画面中{s}{a}，周围是{g}。","一张{s}{a}的照片。","{s}{a}，看起来很专注。"]
T2=["照片中是{s}。","画面里出现了{s}。","这是一张{s}的照片。","{s}占据了画面中央。"]
T3=["{s}在{g}上。","{s}停在{g}边。","{s}位于{g}附近。","{s}出现在{g}上。"]
ACT_Z={"running":"奔跑","jumping":"跳跃","sitting":"坐着","standing":"站着","walking":"行走","playing":"玩耍",
 "swimming":"游泳","riding":"骑行","holding":"拿着东西","eating":"吃东西","driving":"行驶","flying":"飞行"}
GRD_Z={"grass":"绿色的草地","sand":"沙地","water":"水边","road":"道路","snow":"雪地","beach":"海滩","field":"田野","street":"街道"}
SKY_Z={"clouds":"蓝天白云","sky":"开阔的天空","sunny":"阳光","night":"夜色","trees":"树木","sunset":"夕阳"}
def synth(subject, others, act, grd, sky, rng):
    s=zh(subject); g=GRD_Z.get(grd,"空地")
    if act and rng.random()<0.7:
        a=ACT_Z.get(act,"活动")
        t=rng.choice(T1).format(s=s,a=a,g=g)
    elif g and rng.random()<0.6:
        t=rng.choice(T3).format(s=s,g=g)
    else:
        t=rng.choice(T2).format(s=s)
    if sky and rng.random()<0.45:
        t=t[:-1]+"，"+rng.choice([f"背景是{SKY_Z.get(sky,sky)}",f"远处有{SKY_Z.get(sky,sky)}",f"上方是{SKY_Z.get(sky,sky)}"])+"。"
    if others and rng.random()<0.4:
        t=t[:-1]+f"，画面里还能看到{zh(others[0])}"+"。"
    return t
if __name__=="__main__":
    ck=torch.load(ROOT/"ckpt/concept170.pt",map_location="cpu",weights_only=False)
    names=[str(x) for x in ck["names"]]; K=len(names)
    class Net(nn.Module):
        def __init__(self,dd=512,h=768,k=150):
            super().__init__()
            self.trunk=nn.Sequential(nn.Linear(dd,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),nn.Linear(h,h),nn.GELU())
            self.multi=nn.Linear(h,k); self.sal=nn.Linear(h,k)
        def forward(self,x):
            z=self.trunk(x); return self.multi(z),self.sal(z)
    m=Net(k=K); m.load_state_dict(ck["model"]); m.eval()
    old=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
    on=[str(c) for c in old["names"]]
    class OClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(on)))
        def forward(self,x): return self.net(x)
    oclf=OClf(); oclf.load_state_dict(old["model"]); oclf.eval()
    ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
    GRD=["grass","sand","water","road","snow","beach","field","street"];SKY=["clouds","sky","sunny","night","trees","sunset"];CNT=["one","two","three","group","crowd"]
    ALL=ACT+GRD+SKY+CNT
    def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
    LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
    class Attr(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
        def forward(self,x): return self.net(x)
    at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
    perm=torch.randperm(len(FE0)); ntr=int(len(FE0)*0.9); tr_img=set(perm[:ntr].tolist())
    tr=[k for k in range(len(CAPS)) if int(IMG[k]) in tr_img]
    for s in range(1500):
        i=torch.tensor(np.random.choice(tr,256))
        l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
    at.eval()
    with torch.no_grad():
        multi,sal=m(FE); mp=torch.sigmoid(multi); ap=torch.sigmoid(at(FE))
    rng=random.Random(0); texts=[]
    for i in range(len(FE0)):
        cand=[j for j in range(K) if float(mp[i,j])>0.15] or [int(mp[i].argmax())]
        cand=[j for j in cand if names[j] in ZH_FULL]
        if not cand: cand=[int(mp[i].argmax())]
        subj=names[max(cand,key=lambda j: float(sal[i,j]))]
        others=[names[j] for j in sorted(cand,key=lambda j:-float(mp[i,j])) if names[j]!=subj and names[j] in ZH_FULL][:2]
        act=ALL[int(ap[i,:len(ACT)].argmax())] if float(ap[i,:len(ACT)].max())>0.12 else None
        grd=ALL[len(ACT)+int(ap[i,len(ACT):len(ACT)+8].argmax())] if float(ap[i,len(ACT):len(ACT)+8].max())>0.10 else None
        sky=ALL[len(ACT)+8+int(ap[i,len(ACT)+8:len(ACT)+14].argmax())] if float(ap[i,len(ACT)+8:len(ACT)+14].max())>0.10 else None
        texts.append(synth(subj,others,act,grd,sky,rng))
    is_tr=np.array([1 if i in tr_img else 0 for i in range(len(FE0))])
    np.savez_compressed(ROOT/"data/zh_synth2.npz", texts=np.array(texts), img=np.arange(len(FE0)),
                        feats=FE0.numpy(), split=is_tr)
    print("生成中文描述",len(texts),"示例:",texts[0],"/",texts[1],"/",texts[2])
