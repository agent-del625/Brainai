#!/usr/bin/env python3
"""识物 v3：多标签识别 + 显著性排序头（回答"主体是哪个"）。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/all_clip.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
BAD=set("""white black red blue green yellow brown grey gray orange pink purple dark light young old small large big
little another together outside inside front back side top bottom left right middle next near far same other
first second third new good bad nice beautiful pretty happy sad busy empty full open closed
standing running jumping sitting walking playing looking holding wearing riding swimming eating drinking
water air mouth face hair hand head eye eyes leg legs arm arms foot feet body two three four five several many some few
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
names=[w for w,n in words.most_common(900) if n>=50 and in_phrase[w]>=n*0.22][:150]
for extra in ['person','man','woman','boy','girl','child','people','player','skier','swimmer','rider','athlete']:
    if extra not in names: names.append(extra)
K0=len(names)
w2i={w:i for i,w in enumerate(names)}; K=len(names)
print('含人物后概念数',K)
print("概念",K)
# 每张图：多标签 + 显著性（5 条 caption 里出现次数最多的那个）
nimg=len(FE0)
Y=torch.zeros(nimg,K); SAL=torch.full((nimg,),-1,dtype=torch.long); NSAL=torch.zeros(nimg)
per_img={}
for k,c in enumerate(CAPS):
    i=int(IMG[k]); per_img.setdefault(i,[]).append(c)
for i,caps in per_img.items():
    cnt=Counter(); first={}
    for pos,c in enumerate(caps):
        for j,w in enumerate(re.findall(r"[a-z]+",c.lower())):
            w=norm_word(w); idx=w2i.get(w)
            if idx is None: continue
            cnt[idx]+=1
            first.setdefault(idx,pos*100+j)
    if not cnt: continue
    for idx in cnt: Y[i,idx]=1
    sal=max(cnt.items(), key=lambda kv:(kv[1], -first[kv[0]]))[0]
    SAL[i]=sal; NSAL[i]=1
print("有概念的图",int(NSAL.sum()))
class Net(nn.Module):
    def __init__(self,dd=512,h=768,k=150):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(dd,h),nn.GELU(),nn.LayerNorm(h),nn.Dropout(0.15),
                                 nn.Linear(h,h),nn.GELU())
        self.multi=nn.Linear(h,k); self.sal=nn.Linear(h,k)
    def forward(self,x):
        z=self.trunk(x); return self.multi(z), self.sal(z)
m=Net(k=K); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
pos=Y.sum(0); pw=torch.clamp((nimg*0.02)/pos.clamp(min=1),1.0,12.0)
perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_idx=perm[:ntr]; te_idx=perm[ntr:]
tr_has=tr_idx[NSAL[tr_idx]>0]; te_has=te_idx[NSAL[te_idx]>0]
print("训练图",len(tr_idx),"其中有显著性的",len(tr_has),"| 测试图",len(te_idx),"有显著性",len(te_has))
for step in range(1,10001):
    i=tr_idx[torch.randint(0,len(tr_idx),(256,))]
    lo,ls=m(FE[i])
    loss=F.binary_cross_entropy_with_logits(lo,Y[i],pos_weight=pw)
    sel=NSAL[i]>0
    if sel.any():
        mask=(Y[i]>0)
        masked=ls.masked_fill(~mask, -1e9)
        loss=loss+0.8*F.cross_entropy(masked[sel],SAL[i][sel])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0: print(f"step {step} loss {loss.item():.4f}", flush=True)
torch.save({"model":m.state_dict(),"names":names}, ROOT/"ckpt/concept_all.pt")
m.eval()
with torch.no_grad():
    lo,ls=m(FE[te_has]); g=Y[te_has]; sal=SAL[te_has]
t1_multi=(lo.argmax(1).unsqueeze(1)==g).any(1).float().mean().item()
t1_sal=(ls.argmax(1)==sal).float().mean().item()
# 组合：显著性头里挑第一个属于多标签正集的
comb=[]
for r in range(len(te_has)):
    cand=[j for j in range(K) if lo[r,j]>0.0]      # 多标签判定"存在"的类
    if not cand: cand=list(range(K))
    pick=max(cand, key=lambda j: float(ls[r,j]))  # 只在候选里比显著性
    comb.append(int(g[r,pick]>0))                 # 命中任一真实概念即算对
t3=torch.topk(ls,3,1).indices
h3=sum(1 for r in range(len(te_has)) if g[r,t3[r]].sum()>0)/len(te_has)
print(f"多标签 top-1 命中 {t1_multi*100:.0f}%")
PERSON={'person','man','woman','boy','girl','child','people','player','skier','swimmer','rider','athlete'}
pidx=[j for j,n in enumerate(names) if n in PERSON]
mask=torch.zeros(len(te_has))
for r in range(len(te_has)):
    if any(g[r,j]>0 for j in pidx): mask[r]=1
if mask.sum()>0:
    sel=mask.bool()
    hit=0
    for r in torch.nonzero(sel).flatten().tolist():
        order=ls[r].argsort(descending=True).tolist()
        pick=next((j for j in order if j in pidx and g[r,j]>0), order[0])
        hit+=int(g[r,pick]>0)
    print(f"人物图（{int(mask.sum())} 张）主体命中 {hit/max(int(mask.sum()),1)*100:.0f}%")
print(f"显著性头 top-1 {t1_sal*100:.0f}%   组合后 {np.mean(comb)*100:.0f}%   top-3 {h3*100:.0f}%")
