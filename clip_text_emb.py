#!/usr/bin/env python3
"""把概念词编码到 CLIP 文本空间（通感的公共枢纽）。"""
import numpy as np, onnxruntime as ort, json
from pathlib import Path
ROOT=Path("/srv/brainai")
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_text.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
def enc(texts):
    return sess.run([oname],{iname: np.array(texts,dtype=object)})[0]
QD=["bird","car","cat","dog","fish","flower","house","star","sun","train","tree"]
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
ESC=sorted(set(str(c) for c in d["cats"]))
prompts=lambda w: f"the sound of a {w}" 
E_esc=enc([prompts(w) for w in ESC])
E_qd=enc([f"a drawing of a {w}" for w in QD])
np.savez_compressed(ROOT/"data/concept_clip.npz",
                    esc_words=np.array(ESC), esc_emb=E_esc,
                    qd_words=np.array(QD), qd_emb=E_qd)
print("ESC 类别",len(ESC),"QD 类别",len(QD),"维度",E_esc.shape)
print("示例",ESC[:6])
