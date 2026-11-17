#!/usr/bin/env python3
"""COCO 验证集：算 CLIP 特征，构建 (图像, caption) 对，与 Flickr8k 合并。"""
import numpy as np, cv2, glob, pyarrow.parquet as pq, onnxruntime as ort
from pathlib import Path
ROOT=Path("/srv/brainai")
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
per={}
for f in sorted(glob.glob(str(ROOT/"data/downloads/coco_*.parquet"))):
    t=pq.read_table(f).to_pydict()
    for k in range(len(t["cocoid"])):
        cid=int(t["cocoid"][k]); cap=t["caption"][k].strip()
        if cid not in per: per[cid]={"img":t["image"][k]["bytes"],"caps":[]}
        per[cid]["caps"].append(cap)
print("COCO 图像",len(per),"caption 对",sum(len(v["caps"]) for v in per.values()))
ids=sorted(per); feats=[]; caps=[]; idx=[]
B=16; buf=[]; bufc=[]
for n,cid in enumerate(ids):
    b=per[cid]["img"]
    arr=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    rgb=cv2.cvtColor(cv2.resize(arr,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    buf.append(((rgb-mean)/std).transpose(2,0,1)); bufc.append(per[cid]["caps"])
    if len(buf)==B or n==len(ids)-1:
        f=sess.run([oname],{iname: np.stack(buf).astype(np.float32)})[0]
        base=len(feats)
        for row in f: feats.append(row)
        for k,c5 in enumerate(bufc):
            for c in c5: caps.append(c); idx.append(base+k)
        buf=[]; bufc=[]
    if n%1000==0: print(n, flush=True)
F_=np.stack(feats)
np.savez_compressed(ROOT/"data/coco_clip.npz", feats=F_, captions=np.array(caps), img_idx=np.array(idx))
print("完成", F_.shape, len(caps))
