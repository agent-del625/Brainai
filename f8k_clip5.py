#!/usr/bin/env python3
"""从 Flickr8k parquet 重新算 CLIP 特征，并保留每张图的 5 条 caption。"""
import numpy as np, cv2, glob, pyarrow.parquet as pq, onnxruntime as ort
from pathlib import Path
ROOT=Path("/srv/brainai")
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
recs=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    t=pq.read_table(f).to_pydict()
    for k in range(len(t["caption_0"])):
        recs.append((t["image"][k]["bytes"], [t[f"caption_{j}"][k].strip() for j in range(5)]))
print("图像", len(recs))
feats=[]; caps=[]; img_idx=[]
B=16
buf=[]; buf_caps=[]
for n,(b,c5) in enumerate(recs):
    arr=cv2.imdecode(np.frombuffer(b,np.uint8), cv2.IMREAD_COLOR)
    rgb=cv2.cvtColor(cv2.resize(arr,(224,224)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    buf.append(((rgb-mean)/std).transpose(2,0,1)); buf_caps.append(c5)
    if len(buf)==B or n==len(recs)-1:
        f=sess.run([oname],{iname: np.stack(buf).astype(np.float32)})[0]
        feats.append(f)
        base=sum(len(x) for x in feats[:-1])
        for c5 in buf_caps:
            for c in c5: caps.append(c)
        buf=[]; buf_caps=[]
    if n%1000==0: print(n, flush=True)
F_=np.concatenate(feats)
# 重建 img_idx（按顺序）
img_idx=[]
for i in range(len(F_)):
    img_idx.extend([i]*5)
np.savez_compressed(ROOT/"data/f8k_clip5.npz", feats=F_, captions=np.array(caps), img_idx=np.array(img_idx))
print("完成", F_.shape, len(caps))
