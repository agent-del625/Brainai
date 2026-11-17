#!/usr/bin/env python3
"""严格版识物：150 类多标签，只用训练图像训练，留出图像严格评估。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)

BAD=set("""white black red blue green yellow brown grey gray orange pink purple dark light young old small large big
little another together outside inside front back side top bottom left right middle next near far same other
first second third new old good bad nice beautiful pretty happy sad busy empty full open closed
standing running jumping sitting walking playing looking holding wearing riding swimming eating drinking
water air mouth face hair hand head eye eyes leg legs arm arms foot feet body people person man woman
boy girl child children kid group crowd men women boys girls two three four five several many some few
time day night morning evening year way thing things lot lots part piece pair kind sort area place
white black red blue green yellow brown grey gray""".split())
words=Counter(); in_phrase=Counter()
for c in CAPS:
    toks=re.findall(r"[a-z]+", c.lower())
    for i,w in enumerate(toks):
        if len(w)<3 or w in BAD: continue
        words[w]+=1
        if i>0 and toks[i-1] in ("a","an","the","two","three","some","several"): in_phrase[w]+=1
# 只保留"经常跟在冠词/数量词后面"的具体名词
cand=[w for w,n in words.most_common(600) if n>=60 and in_phrase[w]>=n*0.25]
names=cand[:150]
print("概念数",len(names),"示例",names[:12])
w2i={w:i for i,w in enumerate(names)}
ALIAS={"insect":["insect","bug"],"bicycle":["bicycle","bike"],"ocean":["ocean","sea"],"hen":["hen","chicken"],"boat":["boat","canoe"]}
Y=torch.zeros(len(CAPS),len(names))
for k,c in enumerate(CAPS):
    ws=set(re.findall(r"[a-z]+",c.lower()))
    for w in ws:
        if w in w2i: Y[k,w2i[w]]=1
for w,al in ALIAS.items():
    if w not in w2i: continue
    for k,c in enumerate(CAPS):
        ws=set(re.findall(r"[a-z]+",c.lower()))
        if ws & set(al): Y[k,w2i[w]]=1
print("正样本",int(Y.sum()),"平均每句 %.2f 个概念"%float(Y.sum(1).mean()))
class Clf(nn.Module):
    def __init__(self,d=512,h=768,k=150):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),
                               nn.Linear(h,h),nn.GELU(),nn.Linear(h,k))
    def forward(self,x): return self.net(x)
m=Clf(k=len(names)); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
nimg=len(FE0); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img=set(perm[:ntr].tolist()); te_img=set(perm[ntr:].tolist())
tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]     # ★只用训练图像
for step in range(1,5001):
    i=torch.tensor(np.random.choice(tr,256))
    loss=F.binary_cross_entropy_with_logits(m(FE[IMG[i]]), Y[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0: print(f"step {step} bce {loss.item():.4f}", flush=True)
torch.save({"model":m.state_dict(),"names":names}, ROOT/"ckpt/concept150.pt")
# ★严格评估：只用留出图像
m.eval()
te_list=sorted(te_img)
gold_of={}
for i in te_list:
    ks=[k for k in range(len(CAPS)) if int(IMG[k])==i]
    g=torch.zeros(len(names))
    for k in ks: g=torch.maximum(g, Y[k])
    if g.sum()>0: gold_of[i]=g
print("留出图像中有概念的：",len(gold_of))
with torch.no_grad():
    idxs=list(gold_of.keys())
    P=[]
    for s in range(0,len(idxs),128):
        P.append(torch.sigmoid(m(FE[idxs[s:s+128]])))
    P=torch.cat(P)
G=torch.stack([gold_of[i] for i in idxs])
top1=(P.argmax(1).unsqueeze(1)==G).any(1).float().mean().item()
top3=torch.topk(P,3,1).indices
hit3=sum(1 for r in range(len(idxs)) if G[r,top3[r]].sum()>0)/len(idxs)
for thr in (0.3,0.5):
    pred=(P>thr).float()
    tp=(pred*G).sum().item(); fp=(pred*(1-G)).sum().item(); fn=((1-pred)*G).sum().item()
    print(f"阈值 {thr}: 精确率 {tp/max(tp+fp,1)*100:.0f}%  召回率 {tp/max(tp+fn,1)*100:.0f}%  平均每图预测 {pred.sum(1).float().mean():.1f} 个")
print(f"top-1 命中 {top1*100:.0f}%   top-3 命中 {hit3*100:.0f}%   （随机 top-1 {100/len(names):.2f}%）")
torch.save({"model":m.state_dict(),"names":names}, ROOT/"ckpt/concept150.pt")
