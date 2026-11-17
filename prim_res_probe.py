#!/usr/bin/env python3
"""量化精度对天花板的影响：网格 G 与尺度分档 SB 的扫描。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=12
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True)
seqs=list(d["seqs"]); cent=np.load(ROOT/"data/primitives.npz")["cent"]
def resample(st,n=16):
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2: return None
    seg=np.linalg.norm(np.diff(p,axis=0),axis=1)
    if seg.sum()<1e-6: return None
    t=np.concatenate([[0],np.cumsum(seg)]); t/=t[-1]
    u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,p[:,0]),np.interp(u,t,p[:,1])],1)
def split(st,parts=3):
    p=np.asarray(st,dtype=np.float32)
    if len(p)<2*parts: return [p]
    out=[]
    for k in range(parts):
        a=int(len(p)*k/parts); b=int(len(p)*(k+1)/parts)+1
        seg=p[a:min(b,len(p))]
        if len(seg)>=2: out.append(seg)
    return out
def nshape(q):
    c=q.mean(0); z=q-c
    return z/(np.abs(z).max()+1e-6)
random.seed(3); idx=random.sample(range(len(seqs)),1200)
for G,SB,RES in [(12,3,1),(12,3,2),(12,3,4),(16,3,4)]:
    ious=[]
    for i in idx:
        items=[]
        for st in seqs[i]:
            for seg in split(st,3):
                r=resample(seg)
                if r is None: continue
                pid=int(((nshape(r)[None]-cent)**2).sum((-1,-2)).argmin())
                c=r.mean(0); half=np.abs(r-c).max()+1e-6
                fx=(c[0]+1)/2*G; fy=(c[1]+1)/2*G
                gx=int(np.clip(np.floor(fx),0,G-1)); gy=int(np.clip(np.floor(fy),0,G-1))
                rx=int(np.clip(round((fx-gx)*RES),0,RES-1)); ry=int(np.clip(round((fy-gy)*RES),0,RES-1))
                sb=int(np.clip(half*SB/0.6,0,SB-1))
                items.append((gy*G+gx,pid,gx,gy,sb,rx,ry))
        items.sort(key=lambda t:t[0]); items=items[:N]
        img=np.zeros((S,S),np.uint8)
        for _,pid,gx,gy,sb,rx,ry in items:
            cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1; half=(sb+1)*(0.6/SB)/2+0.05
            sh=cent[pid]*half+np.array([cx,cy])
            q=np.stack([(sh[:,0]+1)*S/2-0.5,(sh[:,1]+1)*S/2-0.5],1)
            for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
        gt=np.zeros((S,S),np.uint8)
        for st in seqs[i]:
            q=np.stack([(np.asarray(st)[:,0]+1)*S/2-0.5,(np.asarray(st)[:,1]+1)*S/2-0.5],1)
            for a,b in zip(q,q[1:]): cv2.line(gt,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
        o=cv2.dilate(gt,np.ones((3,3),np.uint8)); r2=cv2.dilate(img,np.ones((3,3),np.uint8))
        A=o>0;B=r2>0
        ious.append((A&B).sum()/max((A|B).sum(),1))
    print(f"网格 {G}×{G} + 每格细分 {RES}×{RES}（等效 {G*RES}×{G*RES}）→ 天花板 {np.mean(ious)*100:>5.1f}%")
