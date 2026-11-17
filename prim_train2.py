#!/usr/bin/env python3
"""积木重建 v2：CNN 编码 + GRU 逐步解码槽位（教师强制），四个分类头。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64
d=np.load(ROOT/"data/prim_ds2.npz",allow_pickle=True)
slots=torch.from_numpy(d["slots"]); mask=torch.from_numpy(d["mask"]).float()
imgs=torch.from_numpy(d["imgs"]).float().unsqueeze(1)/255.0
cent=torch.from_numpy(d["cent"]); N,G,RES,SB=[int(x) for x in d["cfg"]]
K=len(cent); print("样本",len(imgs),"积木",K,"槽位",N)

class Net(nn.Module):
    def __init__(self,h=256):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.GELU(), nn.Conv2d(32,64,3,2,1), nn.GELU(),
            nn.Conv2d(64,96,3,2,1), nn.GELU(), nn.Flatten(), nn.Linear(96*8*8,h), nn.GELU())
        self.emb=nn.Embedding(K+1,h)                # +1 = 起始符
        self.gru=nn.GRU(h*2,h,batch_first=True)
        self.id_h=nn.Linear(h,K); self.cell_h=nn.Linear(h,G*G)
        self.res_h=nn.Linear(h,RES*RES); self.sc_h=nn.Linear(h,SB)
    def forward(self,x,tgt):
        f=self.cnn(x)
        bos=torch.full((x.size(0),1),K,dtype=torch.long)
        inp=torch.cat([bos,tgt[:,:,0]],1)[:,:-1]     # 上一槽的积木 ID
        e=self.emb(inp)
        o,_=self.gru(torch.cat([e,f.unsqueeze(1).expand(-1,e.size(1),-1)],-1))
        return self.id_h(o), self.cell_h(o), self.res_h(o), self.sc_h(o)

m=Net(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
n=len(imgs); perm=torch.randperm(n); tr,te=perm[:int(n*0.9)],perm[int(n*0.9):]
def loss_fn(out,tgt,msk):
    a,b,c,e=out; L=0; w=0
    for lg,idx in ((a,0),(b,1),(c,2),(e,3)):
        ce=F.cross_entropy(lg.transpose(1,2), tgt[:,:,idx], reduction='none')
        L=L+(ce*msk).sum(); w+=msk.sum()
    return L/w
START=1
import os
if os.environ.get('RESUME')=='1' and (ROOT/'ckpt/prim2.pt').exists():
    st=torch.load(ROOT/'ckpt/prim2.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(st['model']); START=st.get('step',1)+1; print('续训自',START)
for step in range(START,16001):
    i=tr[torch.randint(0,len(tr),(64,))]
    out=m(imgs[i], slots[i]); loss=loss_fn(out,slots[i],mask[i])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%200==0:
        torch.save({"model":m.state_dict(),"cent":cent,"cfg":d["cfg"],"step":step}, ROOT/"ckpt/prim2.pt")
        print(f"step {step} loss {loss.item():.3f}", flush=True)

def render(ids,cells,ress,scs):
    img=np.zeros((S,S),np.uint8)
    for pid,cell,res,sb in zip(ids,cells,ress,scs):
        gx,gy=divmod(int(cell),G); rx,ry=divmod(int(res),RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6
        sh=cent[int(pid)].numpy()*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img
m.eval(); ious=[]; rows=[]
with torch.no_grad():
    for i in te[:100].tolist():
        a,b,c,e=m(imgs[i:i+1], slots[i:i+1]*0)   # 解码时用自回归
        # 逐步贪心解码
        ids=[];cells=[];ress=[];scs=[];prev=torch.tensor([[K]])
        f=m.cnn(imgs[i:i+1])
        h=None
        for t in range(N):
            inp=m.emb(prev)
            o,h=m.gru(torch.cat([inp,f.unsqueeze(1)],-1),h)
            pid=int(m.id_h(o)[0,0].argmax()); cell=int(m.cell_h(o)[0,0].argmax())
            res=int(m.res_h(o)[0,0].argmax()); sc=int(m.sc_h(o)[0,0].argmax())
            ids.append(pid);cells.append(cell);ress.append(res);scs.append(sc); prev=torch.tensor([[pid]])
            if t+1>=int(mask[i].sum()): break
        rec=render(ids,cells,ress,scs)
        gt=(d["imgs"][i]>0).astype(np.uint8)
        o1=cv2.dilate(gt,np.ones((3,3),np.uint8)); r1=cv2.dilate(rec,np.ones((3,3),np.uint8))
        A=o1>0;B=r1>0
        ious.append((A&B).sum()/max((A|B).sum(),1))
        if len(rows)<5: rows.append(np.hstack([(255*(1-gt)).astype(np.uint8),np.full((S,4),200,np.uint8),(255*(1-rec)).astype(np.uint8)]))
print(f"重建 膨胀IoU：{np.mean(ious)*100:.1f}%（理论上限约 54%）")
cv2.imwrite(str(ROOT/"logs/turtle/prim2.png"), np.vstack(rows))
print("图:",ROOT/"logs/turtle/prim2.png")
torch.save({"model":m.state_dict(),"cent":cent,"cfg":d["cfg"],"step":step}, ROOT/"ckpt/prim2.pt")
