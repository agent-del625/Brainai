#!/usr/bin/env python3
"""识物 v2：合并单复数 + 正样本加权 + top-3 输出，严格按图像划分评估。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
BAD=set("""white black red blue green yellow brown grey gray orange pink purple dark light young old small large big
little another together outside inside front back side top bottom left right middle next near far same other
first second third new good bad nice beautiful pretty happy sad busy empty full open closed
standing running jumping sitting walking playing looking holding wearing riding swimming eating drinking
water air mouth face hair hand head eye eyes leg legs arm arms foot feet body people person man woman
boy girl child children kid group crowd men women boys girls two three four five several many some few
time day night morning evening year way thing things lot lots part piece pair kind sort area place""".split())
def norm_word(w):
    if w.endswith("ies") and len(w)>4: return w[:-3]+"y"
    if w.endswith("es") and len(w)>4 and w[-3] in "sxzh": return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w)>3: return w[:-1]
    return w
words=Counter(); in_phrase=Counter()
for c in CAPS:
    toks=[norm_word(t) for t in re.findall(r"[a-z]+", c.lower())]
    for i,w in enumerate(toks):
        if len(w)<3 or w in BAD: continue
        words[w]+=1
        if i>0 and toks[i-1] in ("a","an","the","two","three","some","several"): in_phrase[w]+=1
cand=[w for w,n in words.most_common(800) if n>=60 and in_phrase[w]>=n*0.25]
names=cand[:150]
print("概念数",len(names),"示例",names[:12])
w2i={w:i for i,w in enumerate(names)}
Y=torch.zeros(len(CAPS),len(names))
for k,c in enumerate(CAPS):
    for w in set(norm_word(t) for t in re.findall(r"[a-z]+",c.lower())):
        j=w2i.get(w)
        if j is not None: Y[k,j]=1
pos=Y.sum(0); print("正样本",int(Y.sum()),"最少类样本",int(pos.min()))
class Clf(nn.Module):
    def __init__(self,d=512,h=768,k=150):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),
                               nn.Linear(h,h),nn.GELU(),nn.Linear(h,k))
    def forward(self,x): return self.net(x)
m=Clf(k=len(names)); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
# ★正样本加权（缓解“太保守”）
pw=torch.clamp((len(CAPS)*0.02)/pos.clamp(min=1), 1.0, 12.0)
nimg=len(FE0); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img=set(perm[:ntr].tolist()); te_img=set(perm[ntr:].tolist())
tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]
for step in range(1,6001):
    i=torch.tensor(np.random.choice(tr,256))
    tgt=Y[i]
    loss=F.binary_cross_entropy_with_logits(m(FE[IMG[i]]), tgt, pos_weight=pw)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0: print(f"step {step} bce {loss.item():.4f}", flush=True)
torch.save({"model":m.state_dict(),"names":names}, ROOT/"ckpt/concept150b.pt")
m.eval()
gold_of={}
for i in sorted(te_img):
    g=torch.zeros(len(names))
    for k in range(len(CAPS)):
        if int(IMG[k])==i: g=torch.maximum(g, Y[k])
    if g.sum()>0: gold_of[i]=g
idxs=list(gold_of.keys())
with torch.no_grad():
    P=torch.cat([torch.sigmoid(m(FE[idxs[s:s+128]])) for s in range(0,len(idxs),128)])
G=torch.stack([gold_of[i] for i in idxs])
t1=(P.argmax(1).unsqueeze(1)==G).any(1).float().mean().item()
t3=torch.topk(P,3,1).indices
h3=sum(1 for r in range(len(idxs)) if G[r,t3[r]].sum()>0)/len(idxs)
print(f"top-1 {t1*100:.0f}%   top-3 {h3*100:.0f}%   （{len(idxs)} 张留出图）")
for thr in (0.2,0.3,0.5):
    pred=(P>thr).float()
    tp=(pred*G).sum().item(); fp=(pred*(1-G)).sum().item(); fn=((1-pred)*G).sum().item()
    print(f"阈值 {thr}: 精确率 {tp/max(tp+fp,1)*100:.0f}%  召回率 {tp/max(tp+fn,1)*100:.0f}%  每图 {pred.sum(1).mean():.1f} 个")
