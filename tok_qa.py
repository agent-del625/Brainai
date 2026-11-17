#!/usr/bin/env python3
"""重建 token 表：语料 = 图片描述 + 问答（含指令/答案），让词表覆盖问答域。"""
import sys, re, numpy as np
from collections import Counter
sys.path.insert(0,"/srv/brainai")
from tok_pipe import build
from pathlib import Path
ROOT=Path("/srv/brainai")
cap=[str(s).lower() for s in np.load(ROOT/"data/en_big_sents.npy",allow_pickle=True)]
alp=[(str(a).lower(),str(b).lower()) for a,b in np.load(ROOT/"data/alpaca_pairs.npy",allow_pickle=True)]
orc=[(str(a).lower(),str(b).lower()) for a,b in np.load(ROOT/"data/orca_pairs.npy",allow_pickle=True)][:120000]
qa=[x for p in (alp+orc) for x in p]
corpus=cap+qa
print("语料 描述",len(cap),"问答",len(qa),"合计",len(corpus),flush=True)
vocab,imp,seqs=build(corpus, min_cov=8, max_len=10, min_freq=6)
print("词表大小",len(vocab))
np.savez_compressed(ROOT/"data/tok_qa_vocab.npz", vocab=np.array(list(vocab.keys())), imp=imp)
print("已存 data/tok_qa_vocab.npz")
