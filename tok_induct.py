#!/usr/bin/env python3
"""无规则分词：用集合的并/差递归切分语料，让模型自己长出 token。

算法（对应"我喜欢狗/猫/兔子 → 我喜欢、狗、猫、兔子"）：
  给定一个句子集合 S：
    1) 求全体最长公共前缀 P 与最长公共后缀 Q —— 这是"并集里共同的部分"
    2) 剥掉 P、Q 后剩下的就是"差集"，是不同的部分
    3) 对差集按首字分组递归，直到无法再分出共同片段
  每个产出的片段就是候选 token；同时记录它出现在多少个句子里（覆盖度）。
"""
import sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0,"/srv/brainai")

def lcp(strings):
    if not strings: return ""
    s=min(strings, key=len); 
    for i,ch in enumerate(s):
        if any(x[i]!=ch for x in strings): return s[:i]
    return s
def lcs(strings):
    if not strings: return ""
    s=min(strings, key=len)
    for i in range(1,len(s)+1):
        suf=s[-i:]
        if any(not x.endswith(suf) for x in strings): return suf[1:] if i>1 else ""
    return s

class Inducer:
    def __init__(self, min_cov=8, max_len=8):
        self.cov=Counter()          # token -> 覆盖句子数
        self.min_cov=min_cov; self.max_len=max_len
    def run(self, sents):
        self._rec(set(sents), depth=0)
        return self.cov
    def _emit(self, tok, group):
        if 2 <= len(tok) <= self.max_len:
            self.cov[tok]+=len(group)
    def _rec(self, group, depth):
        if len(group)<2 or depth>12: return
        g=list(group)
        p=lcp(g); q=lcs(g)
        # 剥掉共同前后缀
        rest=set()
        for s in g:
            r=s
            if p and r.startswith(p): r=r[len(p):]
            if q and r.endswith(q) and len(r)>len(q): r=r[:-len(q)]
            rest.add(r)
        if len(p)>=2: self._emit(p, group)
        if len(q)>=2: self._emit(q, group)
        rest={r for r in rest if r}
        if len(rest)<2: return
        # 差集：按首字分组继续
        buckets=defaultdict(set)
        for r in rest: buckets[r[0]].add(r)
        for ch,sub in buckets.items():
            if len(sub)>=2:
                # 该首字在所有子句里都出现 → 它本身也是候选（长度1 的交集）
                self.cov[ch]+=len(sub)
                self._rec(sub, depth+1)
            else:
                for r in sub:
                    self.cov[r[0]]+=1                     # 单字也作为候选
                    for k in range(2,min(len(r),self.max_len)+1):
                        self.cov[r[:k]]+=1
if __name__=="__main__":
    demo=["我喜欢狗","我喜欢猫","我喜欢兔子"]
    ind=Inducer(min_cov=2); cov=ind.run(demo)
    print("小例子产出:",{k:v for k,v in sorted(cov.items(),key=lambda kv:-kv[1])})
