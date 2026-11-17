#!/usr/bin/env python3
"""热/冷词表：热表常驻保证速度，冷表存磁盘支持无限增长。"""
import json, numpy as np, torch, torch.nn as nn
from pathlib import Path
ROOT=Path("/srv/brainai")
def split_hotcold(vocab, imp, usage, hot_cap=8000, min_freq=3):
    """按 使用频率×重要性 排序，前 hot_cap 进热表；其余进冷表。"""
    score = {t: float(usage.get(t,0)) * (0.5 + float(imp.get(t,0.0))) for t in vocab}
    ranked = sorted(vocab, key=lambda t: -score[t])
    special = ["<pad>","<bos>","<eos>","<unk>"]
    hot = [t for t in special if t in vocab]
    for t in ranked:
        if len(hot) >= hot_cap: break
        if t not in hot and usage.get(t,0) >= min_freq: hot.append(t)
    hot_set = set(hot)
    cold = [t for t in ranked if t not in hot_set]
    return hot, cold, score
if __name__=="__main__":
    ck=torch.load("/tmp/vlm_gpu.pt",map_location="cpu",weights_only=False)
    vocab=ck["vocab"]
    # 用 en_big_sents + 视频字幕统计使用频次（近似）
    import re
    from collections import Counter
    use=Counter()
    S=[str(x).lower() for x in np.load(ROOT/"data/en_big_sents.npy",allow_pickle=True)][:100000]
    for s in S:
        for w in re.findall(r"[a-z']+|[.,]", s): use[w]+=1
    V=np.load(ROOT/"data/vtt16.npz",allow_pickle=True)
    for cs in V["caps"]:
        for c in cs:
            for w in re.findall(r"[a-z']+|[.,]", str(c).lower()): use[w]+=1
    imp={t:0.5 for t in vocab}
    hot,cold,score=split_hotcold(vocab,imp,use,hot_cap=8000)
    print(f"总词表 {len(vocab)}  热表 {len(hot)}  冷表 {len(cold)}")
    print("冷表示例:", cold[:10])
    print(f"热表覆盖率: {sum(use.get(t,0) for t in hot)/max(sum(use.get(t,0) for t in vocab),1)*100:.1f}% 的 token 出现次数")
    json.dump({"hot":hot,"cold":cold}, open(ROOT/"data/hotcold.json","w"), ensure_ascii=False)
    np.save(ROOT/"data/hotcold_scores.npy", np.array([score[t] for t in hot]))
    print("已存 data/hotcold.json")
