#!/usr/bin/env python3
"""端到端 v3：真实袋头（无 hint）→ 第1段排序 → 第2段填充。"""
import re
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn

from stage1_order import Orderer, K, MAXLEN
ROOT = Path("/srv/brainai")
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]

s1 = torch.load(ROOT/"ckpt/stage1_order.pt", map_location="cpu", weights_only=False)
content = s1["content"]; c2i = {w: i for i, w in enumerate(content)}
m1 = Orderer(n=len(content), d=192); m1.load_state_dict(s1["model"]); m1.eval()

bh = torch.load(ROOT/"ckpt/bag_head.pt", map_location="cpu", weights_only=False)
D, H = 512, 768
class Bag(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H,H//2), nn.GELU(), nn.Linear(H//2, v))
    def forward(self,x): return self.net(x)
mb = Bag(len(content)); mb.load_state_dict(bh["model"]); mb.eval()

s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
patterns, leadp = s2["patterns"], s2["leadp"]
D2, H2 = 128, 512
class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(content), D2); self.pos = nn.Embedding(14, D2)
        self.gap = nn.Sequential(nn.Linear(D2*5,H2), nn.GELU(), nn.LayerNorm(H2), nn.Dropout(0.1),
                                 nn.Linear(H2,H2), nn.GELU(), nn.LayerNorm(H2), nn.Dropout(0.1),
                                 nn.Linear(H2,H2//2), nn.GELU(), nn.Linear(H2//2,len(patterns)))
        self.lead = nn.Sequential(nn.Linear(D2*3,H2), nn.GELU(), nn.LayerNorm(H2), nn.Dropout(0.1),
                                  nn.Linear(H2,H2//2), nn.GELU(), nn.Linear(H2//2,len(leadp)))
m2 = Big(); m2.load_state_dict(s2["model"]); m2.eval()

def bag_of(f, topn=12, thr=0.05):
    with torch.no_grad():
        p = torch.sigmoid(mb(f))
        idx = p[0].topk(topn).indices.tolist()
    return [content[j] for j in idx if p[0, j] > thr]

def order_of(f, ws):
    rows = [c2i[w] for w in ws][:K]; rows += [0]*(K-len(rows))
    with torch.no_grad():
        h0 = m1.enc(f); bag = m1.emb(torch.tensor([rows]))
        h = torch.zeros(1,192); prev = torch.zeros(1,192); out = []
        used = torch.zeros(1,K,dtype=torch.bool); pad = torch.tensor(rows)==0
        for t in range(MAXLEN):
            if bool((used|pad)[0].all()): break
            remain = (bag*(~used).unsqueeze(-1).float()).sum(1)/(~used).sum(1,keepdim=True).clamp(min=1)
            lg, h = m1.step(h0, bag, prev, h, t, remain)
            j = int(lg[:, :K].masked_fill(used|pad, -1e4).argmax())
            out.append(content[rows[j]]); used[0,j]=True; prev = m1.emb(torch.tensor([rows[j]]))
    return out

def fill(seq):
    if len(seq) < 2: return seq
    P = lambda i: c2i[seq[max(0,min(i,len(seq)-1))]]
    with torch.no_grad():
        lead = leadp[int(m2.lead(torch.cat([m2.emb(torch.tensor([P(0)])), m2.emb(torch.tensor([P(0)])),
                                            m2.emb(torch.tensor([P(0)]))],-1)).argmax())]
        out = lead.split() if lead != "<none>" else []
        for j in range(len(seq)-1):
            x = torch.cat([m2.emb(torch.tensor([P(j)])), m2.emb(torch.tensor([P(j+1)])),
                           m2.emb(torch.tensor([P(j-1)])), m2.emb(torch.tensor([P(j+2)])),
                           m2.pos(torch.tensor([j]))],-1)
            p = patterns[int(m2.gap(x).argmax())]
            out.append(seq[j])
            if p != "<none>": out += p.split()
        out.append(seq[-1]); return out

import sys
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
f1s, rec, exact = [], [], []
for i in range(600, 900):
    ws = bag_of(feats[i:i+1], topn=TOPN, thr=THR); seq = order_of(feats[i:i+1], ws); sent = fill(seq)
    gold = words_of(caps[i])
    tp = sum((Counter(sent) & Counter(gold)).values())
    f1s.append(2*tp/max(len(sent)+len(gold),1)); exact.append(sent == gold)
    gc = set(gold) & set(content)
    rec.append(len(set(ws) & gc)/max(len(gc),1))
print(f"[top{TOPN} thr{THR}] F1 {np.mean(f1s)*100:.0f}%  召回 {np.mean(rec)*100:.0f}%  整句 {np.mean(exact)*100:.0f}%")
for i in range(600, 602):
    ws = bag_of(feats[i:i+1], topn=TOPN, thr=THR)
    print(f"  袋: {ws}")
    print(f"  生成: {' '.join(fill(order_of(feats[i:i+1], ws)))[:56]}")
    print(f"  参考: {caps[i][:56]}")
