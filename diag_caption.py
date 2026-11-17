#!/usr/bin/env python3
"""诊断：为什么描述模型对任何图都输出同一句。

三个测试：
 A. 初始隐状态 h0 是否塌缩（不同图的 h0 余弦相似度）
 B. 换掉图像条件后，教师强制损失变化多少（条件到底有没有被用）
 C. 生成的首词分布（是不是所有图首词一样）
"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()
ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"];SKY=["clouds","sky","sunny","night","trees","sunset"];CNT=["one","two","three","group","crowd"]
ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
for s in range(800):
    i=torch.randint(0,len(CAPS),(256,))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(clf(FE)), torch.sigmoid(at(FE))], -1)
cc=torch.load(ROOT/"ckpt/cap8.pt",map_location="cpu",weights_only=False)
vocab=cc["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=320,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],512),nn.GELU(),nn.LayerNorm(512),nn.Linear(512,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
m=Cap(); m.load_state_dict(cc["model"]); m.eval()

nimg=len(FE)
with torch.no_grad():
    H0=m.h0(COND[:200])[0]                       # (200,320)
    Hn=F.normalize(H0,dim=-1); S=Hn@Hn.T
    off=S[~torch.eye(200,dtype=bool)]
    print(f"[A] h0 余弦相似度：均值 {off.mean():.3f}  最大 {off.max():.3f}  最小 {off.min():.3f}")
    print(f"    h0 标准差（跨样本）：{H0.std(0).mean():.4f}")

# B：教师强制损失，正确条件 vs 打乱条件
def tf_loss(cond_idx):
    i=torch.randint(0,len(CAPS),(64,))
    cond=COND[cond_idx]
    L=38
    def enc(s):
        ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
    X=torch.tensor([enc(CAPS[j]) for j in i])
    cp=m.cproj(cond).unsqueeze(1).expand(-1,X.size(1)-1,-1)
    h0=m.init(m.enc(cond)).unsqueeze(0)
    out,_=m.gru(torch.cat([m.emb(X[:,:-1]),cp],-1),h0)
    return F.cross_entropy(m.out(out).reshape(-1,len(vocab)), X[:,1:].reshape(-1), ignore_index=0).item()
torch.manual_seed(0)
with torch.no_grad():
    a=[tf_loss(IMG[torch.randint(0,len(CAPS),(64,))]) for _ in range(5)]
    b=[tf_loss(IMG[torch.randperm(len(CAPS))[:64]]) for _ in range(5)]
print(f"[B] 教师强制 CE：正确条件 {np.mean(a):.3f}  打乱条件 {np.mean(b):.3f}  差 {np.mean(b)-np.mean(a):+.3f}")

# C：首词分布
with torch.no_grad():
    first={}
    for i in range(0,200):
        c=COND[i:i+1]; h=m.h0(c)
        o,_=m.gru(torch.cat([m.emb(torch.tensor([[1]])), m.cproj(c).unsqueeze(1)],-1),h)
        w=int(m.out(o[:,-1]).argmax())
        first[w]=first.get(w,0)+1
top=sorted(first.items(),key=lambda kv:-kv[1])[:5]
print("[C] 前 200 张图的首词分布：", [(inv[k],v) for k,v in top])
