#!/usr/bin/env python3
"""给 QuickDraw 简笔画算 CLIP 特征（供「看图作画」训练用）。"""
import numpy as np, cv2, onnxruntime as ort, torch, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
tr=np.load(ROOT/"data/prim_v6_train.npz",allow_pickle=True)
te=np.load(ROOT/"data/prim_v6_test.npz",allow_pickle=True)
imgs=np.concatenate([tr["imgs"],te["imgs"]])
y=np.concatenate([tr["y"],te["y"]]); slots=np.concatenate([tr["slots"],te["slots"]]); mask=np.concatenate([tr["mask"],te["mask"]])
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
out=[]
B=32
for i in range(0,len(imgs),B):
    batch=[]
    for im in imgs[i:i+B]:
        rgb=cv2.cvtColor(255-im, cv2.COLOR_GRAY2RGB)      # 白底黑线（CLIP 期望自然图）
        x=cv2.resize(rgb,(224,224)).astype(np.float32)/255.
        batch.append(((x-mean)/std).transpose(2,0,1))
    f=sess.run([oname], {iname: np.stack(batch).astype(np.float32)})[0]
    out.append(f)
    if (i//B)%40==0: print(i, flush=True)
F_=np.concatenate(out).astype(np.float32)
np.savez_compressed(ROOT/"data/qd_clip.npz", feats=F_, y=y, slots=slots, mask=mask)
print("完成", F_.shape)
