#!/usr/bin/env python3
"""端到端 v2：第1段排序 + 第2段加强神经填充。"""
import re
from collections import Counter
from pathlib import Path
import numpy as np, torch

from stage1_order import Orderer, words_of, K, MAXLEN
ROOT = Path("/srv/brainai")
z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
feats = torch.from_numpy(z["feats"]).float(); caps = [str(c) for c in z["captions"]]

s1 = torch.load(ROOT/"ckpt/stage1_order.pt", map_location="cpu", weights_only=False)
content = s1["content"]; c2i = {w: i for i, w in enumerate(content)}
m1 = Orderer(n=len(content), d=192); m1.load_state_dict(s1["model"]); m1.eval()

s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
patterns, leadp = s2["patterns"], s2["leadp"]
import torch.nn as nn
D, H = 128, 512
class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(content), D); self.pos = nn.Embedding(14, D)
        self.gap = nn.Sequential(nn.Linear(D*5, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H, H//2), nn.GELU(), nn.Linear(H//2, len(patterns)))
        self.lead = nn.Sequential(nn.Linear(D*3, H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                  nn.Linear(H, H//2), nn.GELU(), nn.Linear(H//2, len(leadp)))
m2 = Big(); m2.load_state_dict(s2["model"]); m2.eval()

def order(i):
    with torch.no_grad():
        h0 = m1.enc(feats[i:i+1])
        ws = [w for w in words_of(caps[i]) if w in c2i][:K]
        rows = [c2i[w] for w in ws] + [0]*(K-len(ws))
        bag = m1.emb(torch.tensor([rows])); h = torch.zeros(1,192); prev = torch.zeros(1,192); out=[]
        used = torch.zeros(1,K,dtype=torch.bool); pad = torch.tensor(rows) == 0
        for t in range(MAXLEN):
            remain = (bag * (~used).unsqueeze(-1).float()).sum(1) / (~used).sum(1, keepdim=True).clamp(min=1)
            lg, h = m1.step(h0, bag, prev, h, t, remain)
            if bool((used|pad)[0].all()): break
            j = int(lg[:, :K].masked_fill(used|pad, -1e4).argmax())
            out.append(content[rows[j]]); used[0,j] = True; prev = m1.emb(torch.tensor([rows[j]]))
        return out

def fill(seq):
    if not seq: return []
    if len(seq) == 1: return seq
    P = lambda i: c2i[seq[max(0, min(i, len(seq)-1))]]
    with torch.no_grad():
        lead = leadp[int(m2.lead(torch.cat([m2.emb(torch.tensor([P(0)])), m2.emb(torch.tensor([P(0)])),
                                            m2.emb(torch.tensor([P(0)]))], -1)).argmax())]
        out = lead.split() if lead != "<none>" else []
        for j in range(len(seq)-1):
            x = torch.cat([m2.emb(torch.tensor([P(j)])), m2.emb(torch.tensor([P(j+1)])),
                           m2.emb(torch.tensor([P(j-1)])), m2.emb(torch.tensor([P(j+2)])),
                           m2.pos(torch.tensor([j]))], -1)
            p = patterns[int(m2.gap(x).argmax())]
            out.append(seq[j])
            if p != "<none>": out += p.split()
        out.append(seq[-1])
    return out

f1s, exact = [], []
for i in range(600, 900):
    pred = fill(order(i)); gold = words_of(caps[i])
    tp = sum((Counter(pred) & Counter(gold)).values())
    f1s.append(2*tp/max(len(pred)+len(gold),1)); exact.append(pred == gold)
print(f"端到端 v2：词级 F1 {np.mean(f1s)*100:.0f}%   整句一致 {np.mean(exact)*100:.0f}%")
for i in range(600, 604):
    print(f"  生成: {' '.join(fill(order(i)))[:58]}")
    print(f"  参考: {caps[i][:58]}")
