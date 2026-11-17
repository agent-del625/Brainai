#!/usr/bin/env python3
"""第 1 段（带 skip 语义）：袋子常含噪音词，模型要学会"只排对的、多余的不选"。

做法：训练时把真词 + 2~4 个随机噪音词混进袋子并打乱；目标序列只含真词，
      停止头自由触发（不再强制用完袋）。
"""
import re, random, time
from collections import Counter
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path("/srv/brainai")
K, MAXLEN = 12, 14
FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())
def words_of(s): return re.findall(r"[a-z']+", s.lower())

class SkipOrder(nn.Module):
    def __init__(self, clip_dim=512, d=192, n=800):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(clip_dim,384), nn.GELU(), nn.LayerNorm(384),
                                 nn.Linear(384,d), nn.GELU())
        self.emb = nn.Embedding(n, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.gru = nn.GRUCell(d*4, d)
        self.att = nn.Linear(d, d)
        self.ptr = nn.Linear(d, 1)
        self.stop = nn.Linear(d, 1)
    def step(self, h0, bag, prev, h, t, remain):
        pe = self.pos(torch.full((h.size(0),), min(t, MAXLEN-1), dtype=torch.long))
        h = self.gru(torch.cat([prev, h0, pe, remain], -1), h)
        s = self.ptr(torch.tanh(self.att(bag)) + h.unsqueeze(1)).squeeze(-1)
        return torch.cat([s, self.stop(h)], -1), h

def main():
    z = np.load(ROOT/"data/image/clip_feats.npz", allow_pickle=True)
    feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]
    cnt = Counter(w for c in caps for w in words_of(c))
    content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
    c2i = {w: i for i, w in enumerate(content)}
    keep = [i for i in range(len(caps)) if 2 <= len([w for w in words_of(caps[i]) if w in c2i]) <= 8]
    perm = torch.randperm(len(keep)); tr = [keep[i] for i in perm[:int(len(keep)*0.9)]]; te = [keep[i] for i in perm[int(len(keep)*0.9):]]
    print(f"样本 {len(keep)}")

    m = SkipOrder(n=len(content)); opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3)
    t0 = time.time()
    for step in range(1, 1601):
        idx = torch.tensor([tr[i] for i in torch.randint(0, len(tr), (32,))])
        rows, tgts = [], []
        for i in idx.tolist():
            gold = [w for w in words_of(caps[i]) if w in c2i][:8]
            noise = random.sample(content, random.randint(2,4))
            bagw = gold + [w for w in noise if w not in gold]
            bagw = bagw[:K]                       # 先保真词，噪音只在有空位时加入
            if len(bagw) > len(gold):
                tail = bagw[len(gold):]; random.shuffle(tail)
                head = bagw[:len(gold)]; random.shuffle(head)
                bagw = head + tail
            rows.append([c2i[w] for w in bagw] + [0]*(K-len(bagw)))
            tgts.append(gold)
        bag = torch.tensor(rows)
        h0 = m.enc(feats[idx]); bag_emb = m.emb(bag)
        h = torch.zeros(len(idx),192); prev = torch.zeros(len(idx),192)
        L = max(len(t) for t in tgts) + 1
        loss, n = 0.0, 0
        for t in range(L):
            used = torch.zeros(len(idx), K, dtype=torch.bool)
            for r in range(len(idx)):
                for j in range(min(t, len(tgts[r]))):
                    for jj, v in enumerate(rows[r]):
                        if v == c2i[tgts[r][j]] and not used[r, jj]:
                            used[r, jj] = True; break
            pad = bag == 0
            remain = (bag_emb * (~used).unsqueeze(-1).float()).sum(1) / (~used).sum(1, keepdim=True).clamp(min=1)
            logits, h = m.step(h0, bag_emb, prev, h, t, remain)
            ptr_l = logits[:, :K].masked_fill(used | pad, -1e4)
            stop_l = logits[:, -1]
            tgt_pos, is_stop = [], []
            for r in range(len(idx)):
                if t < len(tgts[r]):
                    tgt_pos.append(rows[r].index(c2i[tgts[r][t]])); is_stop.append(0)
                else:
                    tgt_pos.append(0); is_stop.append(1)
            l = F.cross_entropy(ptr_l, torch.tensor(tgt_pos), reduction="none") * (1 - torch.tensor(is_stop).float())
            l = l + F.cross_entropy(torch.stack([-stop_l, stop_l], -1), torch.tensor(is_stop), reduction="none")
            loss = loss + l.sum(); n += len(idx)
            prev = m.emb(torch.tensor([rows[r][min(t, len(tgts[r])-1)] for r in range(len(idx))]))
        loss = loss / max(n, 1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 400 == 0: print(f"step {step} loss {loss.item():.3f} {(time.time()-t0)/step:.2f}s/step", flush=True)

    torch.save({"model": m.state_dict(), "content": content}, ROOT/"ckpt/stage1_skip.pt")
    # 评测：用"真词+噪音"的袋
    m.eval(); exact = 0
    from reward import lcs; lcs_sum = 0
    with torch.no_grad():
        for n0, i in enumerate(te[:200]):
            gold = [w for w in words_of(caps[i]) if w in c2i][:8]
            noise = random.sample(content, 3)
            bagw = list(dict.fromkeys(gold + [w for w in noise if w not in gold])); random.shuffle(bagw)
            rows = [c2i[w] for w in bagw][:K]; rows += [0]*(K-len(rows))
            bag = torch.tensor([rows]); be = m.emb(bag); h0 = m.enc(feats[i:i+1])
            h = torch.zeros(1,192); prev = torch.zeros(1,192); used = torch.zeros(1,K,dtype=torch.bool)
            pad = bag == 0; out = []
            for t in range(MAXLEN):
                remain = (be*(~used).unsqueeze(-1).float()).sum(1)/(~used).sum(1,keepdim=True).clamp(min=1)
                lg, h = m.step(h0, be, prev, h, t, remain)
                if lg[:, -1].item() > 0: break
                j = int(lg[:, :K].masked_fill(used|pad, -1e4).argmax())
                out.append(content[rows[j]]); used[0,j] = True; prev = m.emb(torch.tensor([rows[j]]))
            exact += (out == gold); lcs_sum += lcs(out, gold)/max(len(gold),1)
    print(f"带噪音袋：完全正确 {exact/200*100:.0f}%   LCS {lcs_sum/200*100:.0f}%")

if __name__ == "__main__":
    main()
