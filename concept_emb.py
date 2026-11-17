#!/usr/bin/env python3
"""通感枢纽：用我们自己学到的 token 向量表示概念（不依赖外部 tokenizer）。"""
import numpy as np, torch, re
from pathlib import Path
ROOT=Path("/srv/brainai")
ck=torch.load("/tmp/vlm_cont.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; E=ck["model"]["tok.weight"]
print("词表",len(vocab),"向量维度",E.shape[1])
def word_vec(w):
    """整词优先，否则用片段/字符平均"""
    w=w.lower()
    if w in vocab: return E[vocab[w]]
    ids=[]
    i=0
    while i<len(w):
        hit=None
        for L in (4,3,2,1):
            if i+L<=len(w) and w[i:i+L] in vocab: hit=w[i:i+L]; break
        if hit: ids.append(vocab[hit]); i+=len(hit)
        else: i+=1
    if not ids: return E[3]
    return E[ids].mean(0)
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
ESC=sorted(set(str(c) for c in d["cats"]))
QD=["bird","car","cat","dog","fish","flower","house","star","sun","train","tree"]
def emb(words): return np.stack([word_vec(w).numpy() for w in words])
np.savez_compressed(ROOT/"data/concept_emb.npz",
                    esc_words=np.array(ESC), esc_emb=emb(ESC).astype(np.float32),
                    qd_words=np.array(QD), qd_emb=emb(QD).astype(np.float32))
print("ESC",len(ESC),"QD",len(QD),"维度",emb(ESC).shape[1])
print("覆盖检查：", [(w, w in vocab) for w in ESC[:6]])
