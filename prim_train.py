#!/usr/bin/env python3
"""积木重建模型：图像 → 12 个槽位的 (积木ID, 网格位置, 尺度)，全分类，无坐标回归。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=12; G=8; SB=3
d=np.load(ROOT/"data/prim_ds.npz",allow_pickle=True)
slots=torch.from_numpy(d["slots"]); mask=torch.from_numpy(d["mask"])
imgs=torch.from_numpy(d["imgs"]).float().unsqueeze(1)/255.0
cent=torch.from_numpy(d["cent"]); K=len(cent)
print("样本",len(imgs),"积木",K,"槽位",N)

class PrimNet(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,32,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.GELU(),
            nn.Conv2d(64,96,3,stride=2,padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(96*8*8,h), nn.GELU())
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.att=nn.MultiheadAttention(h, 4, batch_first=True)
        self.id_h=nn.Linear(h,K); self.pos_h=nn.Linear(h,G*G); self.sc_h=nn.Linear(h,SB)
    def forward(self,x):
        f=self.cnn(x)
        q=self.q.unsqueeze(0).expand(x.size(0),-1,-1)
        o,_=self.att(q, f.unsqueeze(1), f.unsqueeze(1))
        o=o+q
        return self.id_h(o), self.pos_h(o), self.sc_h(o)

m=PrimNet(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
n=len(imgs); perm=torch.randperm(n); tr,te=perm[:int(n*0.9)],perm[int(n*0.9):]
def loss_fn(pred, target, msk):
    lid,lpos,lsc=pred
    l=(F.cross_entropy(lid.transpose(1,2), target[:,:,0], reduction='none')*msk).sum()/msk.sum()
    l+=(F.cross_entropy(lpos.transpose(1,2), target[:,:,1], reduction='none')*msk).sum()/msk.sum()
    l+=(F.cross_entropy(lsc.transpose(1,2), target[:,:,2], reduction='none')*msk).sum()/msk.sum()
    return l

for step in range(1, 3001):
    i=tr[torch.randint(0,len(tr),(64,))]
    out=m(imgs[i]); loss=loss_fn(out, slots[i], mask[i])
    opt.zero_grad(); loss.backward(); opt.step()
    if step%200==0: print(f"step {step} loss {loss.item():.3f}", flush=True)

# 评估：渲染预测 vs 原图（膨胀 IoU）
def render_slots(ids,poss,scs):
    img=np.zeros((S,S),np.uint8)
    for k,(pid,pos,sb) in enumerate(zip(ids,poss,scs)):
        gy,gx=divmod(int(pos),G)
        cx=(gx+0.5)/G*2-1; cy=(gy+0.5)/G*2-1
        half=(sb+1)*0.2
        shape=cent[int(pid)].numpy()*half+np.array([cx,cy])
        q=np.stack([(shape[:,0]+1)*S/2-0.5,(shape[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img

m.eval(); ious=[]; rows=[]
with torch.no_grad():
    for i in te[:80].tolist():
        lid,lpos,lsc=m(imgs[i:i+1])
        ids=lid[0].argmax(-1).tolist(); poss=lpos[0].argmax(-1).tolist(); scs=lsc[0].argmax(-1).tolist()
        nsl=int(mask[i].sum())
        rec=render_slots(ids[:nsl],poss[:nsl],scs[:nsl])
        gt=(d["imgs"][i]>0).astype(np.uint8)
        o=cv2.dilate(gt,np.ones((3,3),np.uint8)); r=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o>0;B=r>0
        ious.append((A&B).sum()/max((A|B).sum(),1))
        if len(rows)<4: rows.append(np.hstack([(255*(1-gt)).astype(np.uint8),np.full((S,4),200,np.uint8),
                                               (255*(1-rec)).astype(np.uint8)]))
print(f"重建 膨胀IoU：{np.mean(ious)*100:.1f}%（天花板约 54%）")
cv2.imwrite(str(ROOT/"logs/turtle/prim_recon.png"), np.vstack(rows))
print("图:", ROOT/"logs/turtle/prim_recon.png")
torch.save({"model":m.state_dict(),"cent":cent}, ROOT/"ckpt/primnet.pt")
