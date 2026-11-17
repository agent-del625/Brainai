#!/usr/bin/env python3
"""把概念分类器从 23 类扩到 100+ 类：词表从 caption 统计自动生成。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
STOP=set("""a an the two three four five several many some few this that these those his her their its my your our
is are was were be been being has have had do does did can could will would may might must
of in on at to from with without by for about above across after against along among around
as into like near over through under up down out off back then there here where when while and or but
man woman boy girl people person men women boys girls child children kid group crowd dog dogs cat cats
white black red blue green yellow brown grey gray orange pink purple dark light young old small large big
sits sit sitting stands stand standing runs run running walks walk walking jumps jump jumping plays play playing
holding holds hold wearing wears wear looking looks look riding rides ride swimming swims swim eating eats
front side top back left right middle end background foreground photo picture image one some
he she it they we you i him them us""".split())
words=Counter()
for c in CAPS:
    for w in re.findall(r"[a-z]+", c.lower()):
        if len(w)<3 or w in STOP: continue
        words[w]+=1
# 取出现 >=80 次的名词候选
cand=[w for w,n in words.most_common(400) if n>=80]
names=cand[:120]
print("概念数",len(names),"示例",names[:15])
word2i={w:i for i,w in enumerate(names)}
Y=torch.zeros(len(CAPS), len(names))
for k,c in enumerate(CAPS):
    ws=set(re.findall(r"[a-z]+", c.lower()))
    for w in ws:
        j=word2i.get(w)
        if j is not None: Y[k,j]=1
print("正样本总数", int(Y.sum()), "平均每句", float(Y.sum(1).mean()))
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
for step in range(1,3001):
    i=torch.tensor(np.random.choice(tr,256))
    loss=F.binary_cross_entropy_with_logits(m(FE0[IMG[i]]), Y[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"step {step} bce {loss.item():.4f}", flush=True)
# 评估：留出图像上的 top-1 命中率（图里确实有该词才算）
m.eval(); hit=n=0; prec_num=0; prec_den=0
with torch.no_grad():
    for i in te[:1500]:
        p=torch.sigmoid(m(FE0[IMG[i]:IMG[i]+1]))[0]
        gold=Y[i].nonzero().flatten().tolist()
        if not gold: continue
        n+=1
        pred=p.argmax().item()
        hit += int(pred in gold)
        for j in (p>0.5).nonzero().flatten().tolist():
            prec_den+=1; prec_num+= int(j in gold)
print(f"留出图像 top-1 命中率 {hit/max(n,1)*100:.0f}%（{n} 张有概念的图，随机约 {100/len(names):.1f}%）")
print(f"阈值 0.5 时的精确率 {prec_num/max(prec_den,1)*100:.0f}%（共 {prec_den} 个预测）")
torch.save({"model":m.state_dict(),"names":names}, ROOT/"ckpt/concept_clf120.pt")
print("已保存 ckpt/concept_clf120.pt")
