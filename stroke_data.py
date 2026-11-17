#!/usr/bin/env python3
"""把 QuickDraw 原始笔画转成「逐笔 token 序列」+ 64x64 线稿图像。"""
import json, numpy as np, glob, cv2
from pathlib import Path
ROOT=Path("/srv/brainai")
NCLS=11; PER=2500
def _flat2pair(s):
    """QuickDraw 每笔是 [xs, ys] 两个列表（不是交错的坐标对）"""
    if isinstance(s,(list,tuple)) and len(s)==2 and isinstance(s[0],(list,tuple)):
        xs=np.array(s[0],dtype=np.float32); ys=np.array(s[1],dtype=np.float32)
        n=min(len(xs),len(ys))
        return np.stack([xs[:n],ys[:n]],1)
    a=np.array(s,dtype=np.float32).ravel()
    return a.reshape(-1,2)
def norm_drawing(strokes):
    """★整幅图统一归一化（不是每笔各自归一化，否则每笔都会被拉满画布）"""
    strokes=[_flat2pair(s) for s in strokes]
    allp=np.concatenate(strokes,0)
    x0,x1=allp[:,0].min(),allp[:,0].max(); y0,y1=allp[:,1].min(),allp[:,1].max()
    sx=max(x1-x0,1e-3); sy=max(y1-y0,1e-3); sc=max(sx,sy)      # 等比缩放，保持形状
    out=[]
    for a in strokes:
        a=a.copy()
        a[:,0]=(a[:,0]-x0)/sc; a[:,1]=1-(a[:,1]-y0)/sc
        out.append(a)
    return out
def render(strokes,S=64):
    img=np.zeros((S,S),np.uint8)
    for s in strokes:
        for i in range(len(s)-1):
            p1=(int(s[i,0]*(S-1)),int(s[i,1]*(S-1))); p2=(int(s[i+1,0]*(S-1)),int(s[i+1,1]*(S-1)))
            cv2.line(img,p1,p2,255,1)
    return img
def to_tokens(strokes,S=64,BIN=32):
    """每个笔画内部单独算差分，避免跨笔跳变。token: qx*BIN+qy，笔末插 1024。"""
    E=0.35
    def q(v):
        v=np.clip(v,-E,E)
        return np.clip(((v+E)/(2*E)*(BIN-1)).round().astype(int),0,BIN-1)
    out=[]
    for si,s in enumerate(strokes):
        if len(s)<2: continue
        dx=np.diff(s[:,0]); dy=np.diff(s[:,1])
        qx,qy=q(dx),q(dy)
        for a,b in zip(qx,qy): out.append(int(a)*BIN+int(b))
        if si<len(strokes)-1: out.append(BIN*BIN)
    return out
files=sorted(glob.glob(str(ROOT/"data/downloads/quickdraw_*.ndjson")))
seqs=[]; imgs=[]; labels=[]
for fi,f in enumerate(files):
    name=Path(f).stem.replace("quickdraw_","")
    n=0
    with open(f) as fh:
        for line in fh:
            if n>=PER: break
            try: d=json.loads(line)
            except Exception: continue
            strokes=d.get("drawing")
            if not strokes: continue
            strokes=norm_drawing([s for s in strokes if len(s)>1])
            if len(strokes)<2: continue
            t=to_tokens(strokes)
            if t is None or len(t)<6 or len(t)>120: continue
            seqs.append(t); imgs.append(render(strokes)); labels.append(fi); n+=1
    print(name,n,flush=True)
L=max(len(t) for t in seqs)
print("样本",len(seqs),"最长序列",L)
np.savez_compressed(ROOT/"data/stroke_ds.npz", seqs=np.array(seqs,dtype=object),
                    imgs=np.stack(imgs), labels=np.array(labels))
print("已存 data/stroke_ds.npz")
