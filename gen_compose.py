#!/usr/bin/env python3
"""生成新答案：从概念库里组合出训练集里没有的句子，再用"评论家"筛选。

流程：采样概念组合 → 第2段填功能词 → 三重打分（概念可渲染性 / 语言流畅度 / 新颖度）
      → 只保留通过的 → 输出可渲染的句子（图 + 声）
"""
import random, re
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was is in on at of and or to with for by from as it its his her their this that there two one three".split())
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
caps = [str(c) for c in z["captions"]]
seen = {" ".join(re.findall(r"[a-z']+", c.lower())) for c in caps}

s1 = torch.load(ROOT/"ckpt/stage1_order.pt", map_location="cpu", weights_only=False)
content = s1["content"]; c2i = {w:i for i,w in enumerate(content)}
s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
patterns, leadp = s2["patterns"], s2["leadp"]
D,H = 128,512
class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(content), D); self.pos = nn.Embedding(14, D)
        self.gap = nn.Sequential(nn.Linear(D*5,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H,H//2), nn.GELU(), nn.Linear(H//2,len(patterns)))
        self.lead = nn.Sequential(nn.Linear(D*3,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                  nn.Linear(H,H//2), nn.GELU(), nn.Linear(H//2,len(leadp)))
m2 = Big(); m2.load_state_dict(s2["model"]); m2.eval()

tri = torch.load(ROOT/"ckpt/tri18.pt", map_location="cpu", weights_only=False)
cnames = tri["names"]
class Tri(nn.Module):
    def __init__(self, d=128, n=0):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Linear(256,d))
        self.aud = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16,32,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,3,padding=1), nn.GELU(), nn.MaxPool2d(2),
                                 nn.Flatten(), nn.Linear(64*10*16,d))
        self.txt = nn.Embedding(n,d)
    def forward(self,i,a,t):
        return (self.img(i), self.aud(a), self.txt(t))
mt = Tri(n=len(cnames)); mt.load_state_dict(tri["model"]); mt.eval()

def compose(seq):
    P = lambda i: c2i[seq[max(0,min(i,len(seq)-1))]]
    out = []
    with torch.no_grad():
        lead = leadp[int(m2.lead(torch.cat([m2.emb(torch.tensor([P(0)])), m2.emb(torch.tensor([P(0)])),
                                            m2.emb(torch.tensor([P(0)]))],-1)).argmax())]
        if lead != "<none>": out += lead.split()
        for j in range(len(seq)-1):
            x = torch.cat([m2.emb(torch.tensor([P(j)])), m2.emb(torch.tensor([P(j+1)])),
                           m2.emb(torch.tensor([P(j-1)])), m2.emb(torch.tensor([P(j+2)])),
                           m2.pos(torch.tensor([j]))],-1)
            p = patterns[int(m2.gap(x).argmax())]
            out.append(seq[j])
            if p != "<none>": out += p.split()
        out.append(seq[-1])
    return out

from collections import defaultdict
co = defaultdict(int); tot = 0
for c in caps:
    ws = sorted(set(re.findall(r"[a-z']+", c.lower())))
    if len(ws) < 2: continue
    for a in range(len(ws)):
        for b in range(a+1, len(ws)):
            co[(ws[a], ws[b])] += 1
    tot += 1
def plausibility(words):
    """平均共现次数（log1p），衡量这个组合像不像真实场景。"""
    ws = sorted(set(words)); vals = []
    for a in range(len(ws)):
        for b in range(a+1, len(ws)):
            vals.append(co.get((ws[a], ws[b]), 0))
    if not vals: return 0.0
    import numpy as np
    return float(np.mean(np.log1p(vals)))

def render_score(words):
    """概念可渲染性：句子里的词有多少能对上概念库。"""
    hit = [w for w in words if w in cnames]
    return len(hit)/max(len(words),1), hit

def fluency(sent):
    toks = sent.split()
    if len(toks) < 2: return 0.0
    P = lambda i: c2i.get(toks[max(0,min(i,len(toks)-1))], 0)
    with torch.no_grad():
        lp = 0.0
        for j in range(len(toks)-1):
            x = torch.cat([m2.emb(torch.tensor([P(j)])), m2.emb(torch.tensor([P(j+1)])),
                           m2.emb(torch.tensor([P(j-1)])), m2.emb(torch.tensor([P(j+2)])),
                           m2.pos(torch.tensor([j]))],-1)
            lp += float(torch.log_softmax(m2.gap(x), -1).max())
        return lp/len(toks)

random.seed(0)
concepts = [w for w in cnames if w in c2i]
accepted = []
tries = 0
while len(accepted) < 8 and tries < 400:
    tries += 1
    k = random.randint(2, 4)
    words = random.sample(concepts, k)          # 随机组合，训练集里几乎不可能出现
    sent = compose(words)
    s = " ".join(sent)
    if s in seen: continue                       # 新颖度：训练集里没有
    r, hits = render_score(sent)
    f = fluency(s)
    pl = plausibility(hits)
    if r >= 0.5 and f > -1.2 and len(hits) >= 2 and pl > 0.7:
        accepted.append((s, hits, round(r,2), round(f,2), round(pl,2)))

print(f"尝试 {tries} 次，接受 {len(accepted)} 条（新颖 + 可渲染 + 流畅）\n")
for s, hits, r, f, pl in accepted:
    print(f"  『{s}』  概念 {hits}  渲染 {r}  流畅 {f}  共现 {pl}")
import json
(ROOT/"logs/gen_accepted.json").write_text(json.dumps(accepted, ensure_ascii=False), encoding="utf-8")
