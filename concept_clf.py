#!/usr/bin/env python3
"""概念分类器（用 caption 做弱标注）：把 CLIP 特征映射到 23 个概念的概率。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, re
from pathlib import Path
ROOT=Path("/srv/brainai")
z=np.load(ROOT/"data/image/clip_feats.npz",allow_pickle=True)
feats=torch.from_numpy(z["feats"]).float(); caps=[str(c) for c in z["captions"]]
cp=np.load(ROOT/"data/concept_all.npz",allow_pickle=True)
cnames=[str(c) for c in cp["cats"]]
ALIAS={"chirping_birds":["bird","birds"],"crickets":["cricket","grasshopper"],
       "clock_alarm":["clock"],"keyboard_typing":["keyboard"],"church_bells":["church","bell"],
       "insects":["bug","insect","bee"],"train":["train"],"airplane":["plane","airplane"],
       "dog":["dog"],"cat":["cat"],"bird":["bird"],"beach":["beach"],"ocean":["ocean","sea"],
       "river":["river","stream"],"waterfall":["waterfall"],"forest":["forest","woods"],
       "cow":["cow"],"sheep":["sheep"],"frog":["frog"],"hen":["hen","chicken"],"pig":["pig"],
       "rooster":["rooster"],"chainsaw":["chainsaw"]}
Y=torch.zeros(len(caps),len(cnames))
for i,c in enumerate(caps):
    ws=set(re.findall(r"[a-z']+",c.lower()))
    for j,cn in enumerate(cnames):
        if ws & set(ALIAS.get(cn,[cn])): Y[i,j]=1
print("正样本数:", Y.sum(0).int().tolist())

class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384), nn.GELU(), nn.LayerNorm(384), nn.Dropout(0.1),
                               nn.Linear(384,192), nn.GELU(), nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
m=Clf(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
for step in range(1, 2001):
    i=torch.randint(0,len(feats),(128,))
    loss=F.binary_cross_entropy_with_logits(m(feats[i]), Y[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"step {step} bce {loss.item():.4f}")
m.eval()
with torch.no_grad():
    p=torch.sigmoid(m(feats[:2000]))
    hit=n=0
    for i in range(2000):
        gold=Y[i].nonzero().flatten().tolist()
        if not gold: continue
        n+=1
        if int(p[i].argmax()) in gold: hit+=1
    print(f"命中率 {hit/max(n,1)*100:.0f}%（{n} 张有概念的图）")
    freq=torch.bincount(p.argmax(1), minlength=len(cnames))
    top=[(cnames[j], int(freq[j])) for j in freq.argsort(descending=True)[:5]]
    print("预测分布 top5:", top)
torch.save({"model": m.state_dict(), "names": cnames}, ROOT/"ckpt/concept_clf.pt")
