#!/usr/bin/env python3
"""在 8895 词表上做热/冷划分，并实测树莓派速度。"""
import json, re, time, numpy as np, torch, torch.nn as nn
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
ck=torch.load("/tmp/vlm_cont.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; Vn=len(vocab)
use=Counter()
for s in [str(x).lower() for x in np.load(ROOT/"data/en_big_sents.npy",allow_pickle=True)][:150000]:
    for w in re.findall(r"[a-z']+|[.,]", s): use[w]+=1
V=np.load(ROOT/"data/vtt16.npz",allow_pickle=True)
for cs in V["caps"]:
    for c in cs:
        for w in re.findall(r"[a-z']+|[.,]", str(c).lower()): use[w]+=1
special=["<pad>","<bos>","<eos>","<unk>"]
ranked=sorted(vocab, key=lambda t: -use.get(t,0))
for cap in (6000,8000):
    hot=[t for t in special if t in vocab]+[t for t in ranked if use.get(t,0)>=2][:cap-len(special)]
    hs=set(hot); cold=[t for t in vocab if t not in hs]
    cover=sum(use.get(t,0) for t in hot)/max(sum(use.get(t,0) for t in vocab),1)
    print(f"热表 {len(hot):>5}  冷表 {len(cold):>5}  覆盖率 {cover*100:.1f}%")
    json.dump({"hot":hot,"cold":cold,"cover":cover}, open(ROOT/f"data/hotcold_{cap}.json","w"), ensure_ascii=False)
