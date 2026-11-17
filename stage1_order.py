#!/usr/bin/env python3
"""第 1 段：只排内容词顺序（纯指针，不掺功能词）。"""
import argparse, re, time
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
K, MAXLEN = 12, 14
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())

def words_of(s): return re.findall(r"[a-z']+", s.lower())

class Orderer(nn.Module):
    def __init__(self, clip_dim=512, d=192, n=800):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(clip_dim, 256), nn.GELU(), nn.LayerNorm(256))
        self.emb = nn.Embedding(n, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.gru = nn.GRUCell(d + 256 + d + d, d)
        self.att = nn.Linear(d, d)
        self.ptr = nn.Linear(d, 1)
        self.stop = nn.Linear(d, 1)
    def step(self, h0, bag, prev, h, t, remain=None):
        pe = self.pos(torch.full((h.size(0),), min(t, MAXLEN-1), dtype=torch.long))
        parts = [prev, h0, pe]
        if remain is not None:
            parts.append(remain)
        h = self.gru(torch.cat(parts, -1), h)
        s = self.ptr(torch.tanh(self.att(bag)) + h.unsqueeze(1)).squeeze(-1)
        return torch.cat([s, self.stop(h)], -1), h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
    feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]
    cnt = Counter(w for c in caps for w in words_of(c))
    content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
    c2i = {w: i for i, w in enumerate(content)}
    V = len(content)
    keep = [i for i in range(len(caps)) if 2 <= len([w for w in words_of(caps[i]) if w in c2i]) <= K]
    perm = torch.randperm(len(keep)); tr = [keep[i] for i in perm[:int(len(keep)*0.9)]]; te = [keep[i] for i in perm[int(len(keep)*0.9):]]
    print(f"样本 {len(keep)}（训练 {len(tr)}/测试 {len(te)}），内容词表 {V}")

    m = Orderer(n=V); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    t0 = time.time()
    for step in range(1, a.steps+1):
        idx = torch.tensor([tr[i] for i in torch.randint(0, len(tr), (a.batch,))])
        h0 = m.enc(feats[idx])
        rows, tgts = [], []
        for i in idx.tolist():
            ws = [w for w in words_of(caps[i]) if w in c2i][:K]
            ids = [c2i[w] for w in ws] + [0]*(K-len(ws)); rows.append(ids); tgts.append(ws)
        bag = m.emb(torch.tensor(rows))
        h = torch.zeros(len(idx), 192); prev = torch.zeros(len(idx), 192)
        L = max(len(t) for t in tgts) + 1

        def used_mask(upto):
            msk = torch.zeros(len(idx), K, dtype=torch.bool)
            for r in range(len(idx)):
                for j in range(min(upto, len(tgts[r]))):
                    for jj, v in enumerate(rows[r]):
                        if v == c2i[tgts[r][j]] and not msk[r, jj]:
                            msk[r, jj] = True
                            break
            return msk
        loss = 0.0; n = 0
        for t in range(L):
            um = used_mask(t)
            remain = (bag * (~um).unsqueeze(-1).float()).sum(1) / (~um).sum(1, keepdim=True).clamp(min=1)
            logits, h = m.step(h0, bag, prev, h, t, remain)
            ptr_l, stop_l = logits[:, :K], logits[:, -1]
            pad_mask = torch.tensor(rows) == 0
            umask = um
            ptr_l = ptr_l.masked_fill(umask | pad_mask, -1e4)
            exhausted = (umask | pad_mask).all(dim=1)
            stop_l = stop_l.masked_fill(~exhausted, -1e4)
            tgt_idx = torch.tensor([rows[r][t] if t < len(tgts[r]) else 0 for r in range(len(idx))])
            # 把词 id 映射成袋内位置
            pos = torch.tensor([[j for j, v in enumerate(rows[r]) if v == tgt_idx[r].item()][0] if t < len(tgts[r]) else 0 for r in range(len(idx))])
            is_last = torch.tensor([1.0 if t >= len(tgts[r]) else 0.0 for r in range(len(idx))])
            l = F.cross_entropy(ptr_l, pos, reduction="none") * (1-is_last)
            l = l + F.cross_entropy(torch.stack([-stop_l, stop_l], -1), is_last.long(), reduction="none")
            loss = loss + l.sum(); n += len(idx)
            prev = m.emb(torch.tensor([rows[r][min(t, len(tgts[r])-1)] for r in range(len(idx))]))
        loss = loss / max(n,1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0: print(f"step {step} loss {loss.item():.3f} {(time.time()-t0)/step:.2f}s/step", flush=True)

    torch.save({"model": m.state_dict(), "content": content}, ROOT / "ckpt/stage1_order.pt")
    m.eval()
    def gen(i):
        with torch.no_grad():
            h0 = m.enc(feats[i:i+1])
            ws = [w for w in words_of(caps[i]) if w in c2i][:K]
            rows = [c2i[w] for w in ws] + [0]*(K-len(ws))
            bag = m.emb(torch.tensor([rows])); h = torch.zeros(1,192); prev = torch.zeros(1,192); out=[]
            used = torch.zeros(1, K, dtype=torch.bool)
            for t in range(MAXLEN):
                remain = (bag * (~used).unsqueeze(-1).float()).sum(1) / (~used).sum(1, keepdim=True).clamp(min=1)
                logits, h = m.step(h0, bag, prev, h, t, remain)
                pad = torch.tensor(rows) == 0
                ptr_l = logits[:, :K].masked_fill(used | pad, -1e4); stop_l = logits[:, -1]
                if bool((used | pad)[0].all()): break
                j = int(ptr_l.argmax()); w = content[rows[j]]; out.append(w)
                used[0, j] = True
                prev = m.emb(torch.tensor([rows[j]]))
            return out
    def score(i, order):
        """给一个排列打分（模型对数概率），用于局部搜索。"""
        with torch.no_grad():
            h0 = m.enc(feats[i:i+1])
            ws = [w for w in words_of(caps[i]) if w in c2i][:K]
            rows = [c2i[w] for w in ws] + [0]*(K-len(ws))
            bag = m.emb(torch.tensor([rows]))
            h = torch.zeros(1, 192); prev = torch.zeros(1, 192)
            used = torch.zeros(1, K, dtype=torch.bool); pad = torch.tensor(rows) == 0
            total = 0.0
            for t, w in enumerate(order):
                if rows.count(c2i[w]) == 0: return -1e9
                j = rows.index(c2i[w])
                remain = (bag * (~used).unsqueeze(-1).float()).sum(1) / (~used).sum(1, keepdim=True).clamp(min=1)
                lg, h = m.step(h0, bag, prev, h, t, remain)
                total += float(lg[0, j])
                used[0, j] = True
                prev = m.emb(torch.tensor([rows[j]]))
            return total

    def refine(i, order):
        best, bo = score(i, order), order
        for _ in range(2):
            improved = False
            for j in range(len(bo)-1):
                cand = bo[:]; cand[j], cand[j+1] = cand[j+1], cand[j]
                sc = score(i, cand)
                if sc > best + 1e-6:
                    best, bo, improved = sc, cand, True
            if not improved: break
        return bo

    exact, lcs_sum = 0, 0
    from reward import lcs
    for i in te[:200]:
        pred = gen(i); gold = [w for w in words_of(caps[i]) if w in c2i]
        exact += (pred == gold); lcs_sum += lcs(pred, gold)/max(len(gold),1)
    print(f"\n内容词顺序：完全正确 {exact/200*100:.0f}%   LCS 顺序分 {lcs_sum/200*100:.0f}%")
    for i in te[:4]:
        print(f"  生成顺序: {' '.join(gen(i))}")
        print(f"  正确顺序: {' '.join([w for w in words_of(caps[i]) if w in c2i])}")

if __name__ == "__main__":
    main()
