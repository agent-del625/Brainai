#!/usr/bin/env python3
"""积木数据集 v3：K=64（带朝向归一化）、旋转 16 档、尺度 16 档、位置=格子+细分，
   4 万张图，训练/验证/测试 8:1:1 分开存储。"""
import numpy as np, cv2, random
from pathlib import Path
ROOT=Path("/srv/brainai"); S=64; N=20; G=16; RES=4; SB=16; RB=16; K=64
d=np.load(ROOT/"data/quickdraw.npz",allow_pickle=True); seqs=list(d["seqs"])
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
def oriented(z):
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    return z@vt.T, np.arctan2(vt[0,1],vt[0,0])

random.seed(11)
pool=random.sample(range(len(seqs)), 40000)
# 1) 先用前 6000 张建词表
shapes=[]
for i in pool[:6000]:
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            z,th=oriented(nshape(r)); shapes.append(z)
shapes=np.stack(shapes)
rng=np.random.default_rng(0); cent=shapes[rng.choice(len(shapes),K,replace=False)].copy()
for _ in range(10):
    lab=((shapes[:,None]-cent[None])**2).sum((-1,-2)).argmin(1)
    for k in range(K):
        m=lab==k
        if m.sum(): cent[k]=shapes[m].mean(0)
print("词表 K=%d 建好" % K)
np.savez_compressed(ROOT/"data/cent64.npz", cent=cent)

def render(seq):
    img=np.zeros((S,S),np.uint8)
    for st in seq:
        q=np.stack([(np.asarray(st)[:,0]+1)*S/2-0.5,(np.asarray(st)[:,1]+1)*S/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),1,1)
    return img

def encode(i):
    items=[]
    for st in seqs[i]:
        for seg in split(st,3):
            r=resample(seg)
            if r is None: continue
            z,th=oriented(nshape(r))
            pid=int(((z[None]-cent)**2).sum((-1,-2)).argmin())
            c=r.mean(0); half=float(np.abs(r-c).max())
            fx=(c[0]+1)/2*G; fy=(c[1]+1)/2*G
            gx=int(np.clip(np.floor(fx),0,G-1)); gy=int(np.clip(np.floor(fy),0,G-1))
            rx=int(np.clip((fx-gx)*RES,0,RES-1)); ry=int(np.clip((fy-gy)*RES,0,RES-1))
            sb=int(np.clip(half*SB/0.6,0,SB-1))
            rb=int(((th+np.pi)/(2*np.pi)*RB))%RB
            items.append((gy*G+gx, pid, gx*G+gy, rx*RES+ry, sb, rb))
    items.sort(key=lambda t:t[0]); items=items[:N]
    sl=np.zeros((N,5),np.int64); mk=np.zeros(N,np.int64)
    for k,(_,pid,cell,res,sb,rb) in enumerate(items):
        sl[k]=(pid,cell,res,sb,rb); mk[k]=1
    return sl, mk

X=[];M=[];IMG=[]
for n,i in enumerate(pool):
    sl,mk=encode(i); X.append(sl); M.append(mk); IMG.append(render(seqs[i])*255)
    if (n+1)%5000==0: print("已处理", n+1, flush=True)
X=np.stack(X); M=np.stack(M); IMG=np.stack(IMG)
n=len(X); idx=np.random.permutation(n)
tr,va,te=idx[:int(n*0.8)], idx[int(n*0.8):int(n*0.9)], idx[int(n*0.9):]
np.savez_compressed(ROOT/"data/prim_v7_train.npz", slots=X[tr],mask=M[tr],imgs=IMG[tr],cent=cent,cfg=np.array([N,G,RES,SB,RB,K]))
np.savez_compressed(ROOT/"data/prim_v7_val.npz", slots=X[va], mask=M[va], imgs=IMG[va])
np.savez_compressed(ROOT/"data/prim_v7_test.npz", slots=X[te],mask=M[te],imgs=IMG[te])
print("数据划分 训练 %d 验证 %d 测试 %d，平均槽位 %.2f" % (len(tr),len(va),len(te), M.sum(1).mean()))
