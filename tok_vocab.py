#!/usr/bin/env python3
"""由集合切分结果生成 token 表：重要性 = 覆盖度/(包含它的更长 token 覆盖度)。"""
import sys, numpy as np, json
from collections import Counter, defaultdict
sys.path.insert(0,"/srv/brainai")
from tok_induct import Inducer
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/zh_all.npz",allow_pickle=True)
SENTS=[str(t) for t in d["texts"]]
print("语料句数",len(SENTS))
ind=Inducer(min_cov=8, max_len=8); cov=ind.run(SENTS)
print("候选片段",len(cov))
# 重要性：若存在更长的包含它的片段且覆盖度不小于它 → 被"吸收"
items=sorted(cov.items(), key=lambda kv:(-kv[1], -len(kv[0])))
bylen=defaultdict(list)
for t,c in items: bylen[len(t)].append((t,c))
def importance(t,c):
    """若存在更长的包含它的片段，且覆盖度不低于它 → 它只在词内出现，判定被吸收。"""
    for L in range(len(t)+1, 9):
        for u,cu in bylen.get(L,[]):
            if t in u and cu>=c: return 0.0
    return 1.0
imp={t:importance(t,c) for t,c in items}
keep=[(t,c) for t,c in items if imp[t]>0.0]
absorbed=[(t,c) for t,c in items if imp[t]==0.0]
print(f"独立 token {len(keep)}  被吸收（只在词内才有意义）{len(absorbed)}")
chars=[t for t,_ in absorbed if len(t)==1]
print("被吸收的单字数",len(chars))
print("被吸收示例:", [t for t,_ in absorbed[:10]])
print("保留示例:", [(t,c) for t,c in keep[:12]])
# 贪心最长匹配分词
vocab=[t for t,_ in keep]
vocab.sort(key=len, reverse=True)
def tokenize(s):
    out=[]; i=0
    while i<len(s):
        for L in range(min(8,len(s)-i),0,-1):
            if s[i:i+L] in imp and imp[s[i:i+L]]>0.0:
                out.append(s[i:i+L]); i+=L; break
        else:
            out.append(s[i]); i+=1
    return out
toks=[tokenize(s) for s in SENTS]
lens=[len(t) for t in toks]
covered=sum(sum(len(x) for x in t if len(x)>1) for t in toks)/sum(len(s) for s in SENTS)
print(f"平均 token/句 {np.mean(lens):.1f}（字符平均 {np.mean([len(s) for s in SENTS]):.1f}）  多字 token 覆盖 {covered*100:.0f}% 字符")
freq=Counter(x for t in toks for x in t)
print("最常用 token:", freq.most_common(12))
json.dump({"vocab":[t for t,_ in keep],"imp":{t:imp[t] for t,_ in keep},"cov":{t:c for t,c in keep}},
          open(ROOT/"data/tok_zh.json","w"), ensure_ascii=False)
np.savez_compressed(ROOT/"data/tok_zh_seqs.npz", seqs=np.array(["|".join(t) for t in toks]))
print("已存 data/tok_zh.json 与分词结果")
