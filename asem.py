#!/usr/bin/env python3
"""声音语义编码器：mel → 离散 token，判据是「分类准确率不下降」而不是重建误差。"""
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path(r"C:\work"); DEV="cuda"
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"][:,None,:,:]; Y=torch.from_numpy(d["target"]).long()
mu,sd=float(M.mean()),float(M.std()+1e-6)
X=torch.from_numpy((M-mu)/sd).float()
print("数据",tuple(X.shape),flush=True)
NT=40; K=64; NB=2
class Enc(nn.Module):
    """输出 40 个 token 位置 × ZD 维"""
    def __init__(self,zd=128):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.GELU(),nn.Conv2d(32,64,4,2,1),nn.GELU(),
                             nn.Conv2d(64,128,4,2,1),nn.GELU(),nn.Conv2d(128,zd,3,1,1),nn.GELU())
        self.pool=nn.AdaptiveAvgPool2d((5,8))
    def forward(self,x):
        z=self.c(x); z=self.pool(z)                     # (B,zd,5,8)
        B,C,H,W=z.shape
        return z.permute(0,2,3,1).reshape(B,H*W,C)      # (B,40,zd)
class VQ(nn.Module):
    def __init__(self,K,zd):
        super().__init__()
        self.emb=nn.Embedding(K,zd)
    def forward(self,r):
        B,T,C=r.shape
        zf=r.reshape(-1,C)
        e=self.emb.weight
        dist=(zf**2).sum(1,keepdim=True)-2*zf@e.T+(e**2).sum(1)[None]
        idx=dist.argmin(1); q=e[idx]
        vql=((q-zf.detach())**2).mean()+0.25*((zf-q.detach())**2).mean()
        return q.view(B,T,C), idx.view(B,T), vql
ZD=128
enc=Enc(ZD).to(DEV)
vqs=[VQ(K,ZD).to(DEV) for _ in range(NB)]
clf=nn.Sequential(nn.Flatten(),nn.Linear(NT*ZD,512),nn.GELU(),nn.Dropout(0.2),nn.Linear(512,50)).to(DEV)
params=list(enc.parameters())+list(clf.parameters())+[p for v in vqs for p in v.parameters()]
opt=torch.optim.AdamW(params,lr=4e-4,weight_decay=0.01)
n=len(X); perm=torch.randperm(n); ntr=int(n*0.8)
tr=perm[:ntr]; te=perm[ntr:]
def quantize(z):
    r=z; qs=[]; idxs=[]; vql=0.0
    for v in vqs:
        q,idx,l=v(r); qs.append(q); idxs.append(idx); vql=vql+l; r=r-q.detach()
    return sum(qs),idxs,vql
def revive():
    """死码复活：把使用率低的码字替换成当前 batch 的随机构造向量"""
    with torch.no_grad():
        for v in vqs:
            v.emb.weight.data.normal_(0,0.1)
t0=time.time()
for step in range(1,6001):
    i=tr[torch.randint(0,len(tr),(64,))]
    x=X[i].to(DEV); y=Y[i].to(DEV)
    z=enc(x)
    zq,idxs,vql=quantize(z)
    logits=clf(zq.reshape(len(i),-1))
    ce=F.cross_entropy(logits,y)
    loss=ce+1.0*vql
    opt.zero_grad(); loss.backward(); opt.step()
    if step%1000==0:
        enc.eval(); clf.eval()
        with torch.no_grad():
            xv=X[te[:400]].to(DEV); yv=Y[te[:400]].to(DEV)
            zv=enc(xv); zqv,_,_=quantize(zv)
            acc=(clf(zqv.reshape(len(xv),-1)).argmax(-1)==yv).float().mean().item()
        enc.train(); clf.train()
        print(f"  step {step} ce {ce.item():.3f} vq {vql.item():.4f} 测试准确率 {acc*100:.1f}%",flush=True)
        if step==2000 or step==5000:
            with torch.no_grad():
                for bi,ix in enumerate(quantize(enc(X[te[:400]].to(DEV)))[1]):
                    print(f"    码本{bi} 使用 {len(torch.unique(ix))}/{K}",flush=True)
# 最终对比：原始 mel 分类 vs 量化 token 分类
enc.eval(); clf.eval()
with torch.no_grad():
    xv=X[te].to(DEV); yv=Y[te].to(DEV)
    zv=enc(xv); zqv,idxs,_=quantize(zv)
    acc_q=(clf(zqv.reshape(len(xv),-1)).argmax(-1)==yv).float().mean().item()
    # 直接用连续潜变量（不量化）分类：检验量化损失
    acc_c=(clf(zv.reshape(len(xv),-1)).argmax(-1)==yv).float().mean().item()
print(f"最终：连续潜变量 {acc_c*100:.1f}%   量化 token {acc_q*100:.1f}%   压缩率 {NT*math.log2(K):.0f} bit/条",flush=True)
torch.save({"enc":enc.state_dict(),"clf":clf.state_dict(),"vq":[v.state_dict() for v in vqs],
            "K":K,"ZD":ZD,"NT":NT,"mu":mu,"sd":sd},ROOT/"asem.pt")
print(f"用时 {time.time()-t0:.0f}s\nDONE")
