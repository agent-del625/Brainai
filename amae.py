#!/usr/bin/env python3
"""声音自编码器 + 残差量化（RVQ）：学出来的表示，对比硬切块。
   目标：同样 12 bit 信息量，重建 MSE 从 0.364 降到 0.15 以下。"""
import math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path(r"C:\work"); DEV="cuda" if torch.cuda.is_available() else "cpu"
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
M=d["mels"][:,None,:,:]                      # (N,1,80,128)
M=(M-M.mean())/(M.std()+1e-6)
X=torch.from_numpy(M).float()
print("数据",tuple(X.shape),"设备",DEV,flush=True)
class Enc(nn.Module):
    def __init__(self,zdim=64):
        super().__init__()
        self.c=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.GELU(),          # 80→40, 128→64
                             nn.Conv2d(32,64,4,2,1),nn.GELU(),          # 20,32
                             nn.Conv2d(64,128,4,2,1),nn.GELU(),         # 10,16
                             nn.Conv2d(128,zdim,3,1,1),nn.GELU())
    def forward(self,x): return self.c(x)
class Dec(nn.Module):
    def __init__(self,zdim=64):
        super().__init__()
        self.c=nn.Sequential(nn.ConvTranspose2d(zdim,128,4,2,1),nn.GELU(),   # 20,32
                             nn.ConvTranspose2d(128,64,4,2,1),nn.GELU(),     # 40,64
                             nn.ConvTranspose2d(64,32,4,2,1),nn.GELU(),      # 80,128
                             nn.Conv2d(32,1,3,1,1))
    def forward(self,z): return self.c(z)
ZD=64
enc=Enc(ZD).to(DEV); dec=Dec(ZD).to(DEV)
class VQ(nn.Module):
    def __init__(self,K,zdim=ZD):
        super().__init__()
        self.emb=nn.Embedding(K,zdim); self.emb.weight.data.uniform_(-1/K,1/K); self.K=K
    def forward(self,z):
        B,C,H,W=z.shape
        zf=z.permute(0,2,3,1).reshape(-1,C)
        d=((zf**2).sum(1,keepdim=True)-2*zf@self.emb.weight.T+(self.emb.weight**2).sum(1)[None])
        idx=d.argmin(1); q=self.emb(idx)
        loss=((q-zf.detach())**2).mean()+0.25*((zf-q.detach())**2).mean()
        q=q.view(B,H,W,C).permute(0,3,1,2)
        zq=z+(q-z).detach()
        return zq,q,loss,idx
NBOOK=3; K=64
vqs=[VQ(K).to(DEV) for _ in range(NBOOK)]
params=list(enc.parameters())+list(dec.parameters())+[p for v in vqs for p in v.parameters()]
opt=torch.optim.AdamW(params,lr=4e-4)
n=len(X); perm=torch.randperm(n); ntr=int(n*0.85); tr,te=perm[:ntr],perm[ntr:]
t0=time.time()
def quantize(z):
    """标准残差量化：每级对上一级的残差做最近邻，累积求和即为量化结果"""
    r=z; qs=[]; vql=0.0; idxs=[]
    for v in vqs:
        B,C,H,W=r.shape
        zf=r.permute(0,2,3,1).reshape(-1,C)
        d=(zf**2).sum(1,keepdim=True)-2*zf@v.emb.weight.T+(v.emb.weight**2).sum(1)[None]
        idx=d.argmin(1); q=v.emb(idx)
        vql=vql+((q-zf.detach())**2).mean()+0.25*((zf-q.detach())**2).mean()
        qv=q.view(B,H,W,C).permute(0,3,1,2)
        qs.append(qv); idxs.append(idx.view(B,H,W)); r=r-qv.detach()
    return sum(qs), qs, vql, idxs
for step in range(1,4001):
    i=tr[torch.randint(0,len(tr),(32,))]
    x=X[i].to(DEV)
    z=enc(x)
    zq,qs,vql,_=quantize(z)
    rec=dec(zq)
    mse=F.mse_loss(rec,x); loss=mse+vql
    opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"  step {step} mse {mse.item():.4f} vq {vql.item():.4f}",flush=True)
enc.eval(); dec.eval()
with torch.no_grad():
    x=X[te[:400]].to(DEV); z=enc(x)
    _,qs,_,_=quantize(z)
    for nq in (1,2,3):
        rec=dec(sum(qs[:nq])); mse=F.mse_loss(rec,x).item()
        print(f"  用 {nq} 个码本（{nq*6:.0f} bit/块）→ MSE {mse:.4f}",flush=True)
print(f"训练用时 {time.time()-t0:.0f}s",flush=True)
torch.save({"enc":enc.state_dict(),"dec":dec.state_dict(),"vq":[v.state_dict() for v in vqs],"K":K,"zdim":ZD},ROOT/"amae.pt")
print("DONE")
