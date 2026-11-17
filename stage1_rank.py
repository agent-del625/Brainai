#!/usr/bin/env python3
"""第 1 段（排序打分版）：给每个袋内词打分，按分数排序，不做逐步解码。

好处：并行、无停止问题、天然保证每个词只用一次。
损失：成对 logistic（A 应在 B 前）+ 位置回归辅助。
"""
import re, time
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
K = 12
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

class Ranker(nn.Module):
    def __init__(self, clip_dim=512, d=192):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(clip_dim, 384), nn.GELU(), nn.LayerNorm(384),
                                 nn.Linear(384, d), nn.GELU())
        self.emb = nn.Embedding(800, d)
        self.score = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, 1))
        self.ctx = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, clip, bag):
        h0 = self.enc(clip)                       # (B,d)
        e = self.emb(bag)                         # (B,K,d)
        x = torch.cat([e, h0.unsqueeze(1).expand_as(e)], -1)
        return self.score(x).squeeze(-1)          # (B,K)

def main() -> int:
    z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
    feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]
    cnt = Counter(w for c in caps for w in words_of(c))
    content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
    c2i = {w: i for i, w in enumerate(content)}
    keep = [i for i in range(len(caps)) if 2 <= len([w for w in words_of(caps[i]) if w in c2i]) <= K]
    perm = torch.randperm(len(keep)); tr = [keep[i] for i in perm[:int(len(keep)*0.9)]]; te = [keep[i] for i in perm[int(len(keep)*0.9):]]
    print(f"样本 {len(keep)}")

    m = Ranker(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(1, 1501):
        idx = torch.tensor([tr[i] for i in torch.randint(0, len(tr), (32,))])
        rows, pos = [], []
        for i in idx.tolist():
            ws = [w for w in words_of(caps[i]) if w in c2i][:K]
            ids = [c2i[w] for w in ws] + [0]*(K-len(ws))
            rows.append(ids)
            rank = {w: j for j, w in enumerate(ws)}
            pos.append([rank.get(content[v], -1) if v != 0 else -1 for v in ids])
        bag = torch.tensor(rows); P = torch.tensor(pos).float()
        s = m(feats[idx], bag)                    # (B,K)
        valid = (bag != 0)
        # 成对 logistic：i 应在 j 前 ⇔ pos_i < pos_j
        loss, n = 0.0, 0
        for a in range(K):
            for b in range(a+1, K):
                va, vb = valid[:, a], valid[:, b]
                mask = va & vb
                if mask.sum() == 0: continue
                target = (P[:, a] < P[:, b]).float()
                diff = s[:, a] - s[:, b]
                loss = loss + F.binary_cross_entropy_with_logits(diff[mask], target[mask], reduction="sum")
                n += int(mask.sum())
        loss = loss / max(n, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0: print(f"step {step} loss {loss.item():.3f} {(time.time()-t0)/step:.2f}s/step", flush=True)

    m.eval(); exact, lcs_sum = 0, 0
    from reward import lcs
    with torch.no_grad():
        for n0, i in enumerate(te[:300]):
            ws = [w for w in words_of(caps[i]) if w in c2i][:K]
            ids = [c2i[w] for w in ws] + [0]*(K-len(ws))
            s = m(feats[i:i+1], torch.tensor([ids])).squeeze(0)
            order = [content[ids[j]] for j in s.argsort(descending=True).tolist() if ids[j] != 0]
            exact += (order == ws); lcs_sum += lcs(order, ws)/max(len(ws),1)
            if n0 < 4:
                print(f"  生成 {order}  |  正确 {ws}")
    print(f"排序打分版：完全正确 {exact/300*100:.0f}%   LCS {lcs_sum/300*100:.0f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
