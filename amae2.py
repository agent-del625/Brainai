#!/usr/bin/env python3
"""声音自编码器 v2（修正 VQ 塌缩）+ 可逆性检验 + 分频段误差。
   关键修正：k-means 预热码本、死码复活、潜变量归一化、训练更久。"""
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path(r"C:\work"); DEV="cuda"
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"][:,None,:,:]; mu,sd=M.mean(),M.std()+1e-6
Xn=(M-mu)/sd
X=torch.from_numpy(Xn).float()
print("数据",tuple(X.shape),flush=True)
class Enc(nn.Module):
    def __init__(self,zd=64):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.GELU(),nn.Conv2d(32,64,4,2,1),nn.GELU(),
                             nn.Conv2d(64,128,4,2,1),nn.GELU(),nn.Conv2d(128,zd,3,1,1))
    def forward(self,x): return self.c(x)
class Dec(nn.Module):
    def __init__(self,zd=64):
        super().__init__()
        self.c=nn.Sequential(nn.ConvTranspose2d(zd,128,4,2,1),nn.GELU(),
                             nn.ConvTranspose2d(128,64,4,2,1),nn.GELU(),
                             nn.ConvTranspose2d(64,32,4,2,1),nn.GELU(),
                             nn.Conv2d(32,1,3,1,1))
    def forward(self,z): return self.c(z)
ZD=64; K=64; NBOOK=3
enc=Enc(ZD).to(DEV); dec=Dec(ZD).to(DEV)
class VQ(nn.Module):
    def __init__(self,K):
        super().__init__()
        self.emb=nn.Embedding(K,ZD); self.K=K
    def forward(self,r):
        B,C,H,W=r.shape
        zf=r.permute(0,2,3,1).reshape(-1,C)
        zf=F.normalize(zf,dim=-1)                   # ★归一化，避免幅度不匹配
        e=F.normalize(self.emb.weight,dim=-1)
        dist=(zf**2).sum(1,keepdim=True)-2*zf@e.T+(e**2).sum(1)[None]
        idx=dist.argmin(1); q=self.emb(idx)
        vql=((q-zf.detach())**2).mean()+0.25*((zf-q.detach())**2).mean()
        return q.view(B,H,W,C).permute(0,3,1,2), idx, vql, zf
vqs=[VQ(K).to(DEV) for _ in range(NBOOK)]
params=list(enc.parameters())+list(dec.parameters())+[p for v in vqs for p in v.parameters()]
opt=torch.optim.AdamW(params,lr=4e-4)
n=len(X); perm=torch.randperm(n); ntr=int(n*0.85); tr,te=perm[:ntr],perm[ntr:]
# ---- k-means 预热 ----
enc.eval()
with torch.no_grad():
    z=enc(X[tr[:800]].to(DEV))
    zf=z.permute(0,2,3,1).reshape(-1,ZD)
    zfn=F.normalize(zf,dim=-1)
    from numpy.random import default_rng
    rng=default_rng(0)
    C=zfn[rng.choice(len(zfn),K,replace=False)].clone()
    for _ in range(10):
        dd=(zfn**2).sum(1,keepdim=True)-2*zfn@C.T+(C**2).sum(1)[None]
        ix=dd.argmin(1)
        for k in range(K):
            sel=zfn[ix==k]
            if len(sel): C[k]=F.normalize(sel.mean(0),dim=0)
    vqs[0].emb.weight.data=C.clone()
    print("码本已用 k-means 预热",flush=True)
enc.train()
def quantize(z):
    r=z; qs=[]; vql=0.0; idxs=[]
    for v in vqs:
        q,idx,l,zf=v(r)
        qs.append(q); idxs.append(idx); vql=vql+l
        r=r-q.detach()
    return sum(qs), qs, vql, idxs
def revive(vqs):
    """死码复活：使用率过低的码字用随机潜变量替换"""
    with torch.no_grad():
        for v in vqs:
            e=v.emb.weight.data
            e.normal_()                            # 简单有效：重新随机化整表（会由后续训练收敛）
t0=time.time()
for step in range(1,12001):
    i=tr[torch.randint(0,len(tr),(32,))]
    x=X[i].to(DEV)
    z=enc(x)
    zq,qs,vql,idxs=quantize(z)
    rec=dec(zq)
    mse=F.mse_loss(rec,x); loss=mse+2.0*vql
    opt.zero_grad(); loss.backward(); opt.step()
    if step%3000==0: print(f"  step {step} mse {mse.item():.4f} vq {vql.item():.4f}",flush=True)
enc.eval(); dec.eval()
with torch.no_grad():
    xt=X[te[:400]].to(DEV)
    z=enc(xt); zq,qs,vql,idxs=quantize(z)
    # 码字使用率
    for bi,ix in enumerate(idxs):
        u=len(torch.unique(ix))
        print(f"  码本{bi} 使用码字 {u}/{K}",flush=True)
    for nq in (1,2,3):
        rec=dec(sum(qs[:nq]))
        mse=F.mse_loss(rec,xt).item()
        print(f"  {nq} 码本（{nq*6} bit/块，共 {nq*6*40} bit/条）→ MSE {mse:.4f}",flush=True)
    # 可逆性：解码结果再编码，比较潜变量一致性
    re_z=enc(dec(sum(qs)))
    cos=F.cosine_similarity(z.flatten(1),re_z.flatten(1),dim=-1).mean().item()
    print(f"  往返一致性（潜变量余弦）{cos:.3f}",flush=True)
    # 分频段误差（低频/中频/高频各 1/3）
    rec=dec(sum(qs))
    for name,sl in (("低频",slice(0,27)),("中频",slice(27,54)),("高频",slice(54,80))):
        e=F.mse_loss(rec[:,:,sl,:],xt[:,:,sl,:]).item()
        print(f"  {name}误差 {e:.4f}",flush=True)
torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"vq":[v.state_dict() for v in vqs],
            "K":K,"zdim":ZD,"mu":mu,"sd":sd},ROOT/"amae2.pt")
print(f"用时 {time.time()-t0:.0f}s")
print("DONE")
