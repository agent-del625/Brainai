#!/usr/bin/env python3
"""袋 + 排列器 v2（修版）。

修复三处：
1. 功能词走独立嵌入表（不再回灌成 id 0 导致输出全是 "a"）
2. 指针监督用 oracle 袋对齐（靶子不随模型预测漂移）
3. 停止符加权，避免永不收尾

评测：词级 F1 + 前 6 词完全匹配率。
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/srv/brainai")
K = 8
MAXLEN = 20
FUNC = ("a an the is are was were in on at of and or to with for by from as it its his her "
        "their this that there two one three").split()


def words_of(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


class BagArranger(nn.Module):
    def __init__(self, clip_dim: int = 512, d: int = 128, n_content: int = 800, n_func: int = 40):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(clip_dim, 256), nn.GELU(), nn.LayerNorm(256))
        self.bag_head = nn.Linear(256, n_content)
        self.emb_c = nn.Embedding(n_content, d)
        self.emb_f = nn.Embedding(n_func, d)
        self.pos = nn.Embedding(MAXLEN, d)
        self.gru = nn.GRUCell(d * 2 + 256, d)
        self.att = nn.Linear(d, d)
        self.ptr = nn.Linear(d, 1)
        self.func = nn.Linear(d, n_func)
        self.stop = nn.Linear(d, 1)

    def encode(self, clip):
        h0 = self.enc(clip)
        return h0, self.bag_head(h0)

    def step(self, h0, bag_emb, prev, h, t):
        pe = self.pos(torch.full((h.size(0),), min(t, MAXLEN - 1), dtype=torch.long))
        h = self.gru(torch.cat([prev, h0, pe], -1), h)
        s = self.ptr(torch.tanh(self.att(bag_emb)) + h.unsqueeze(1)).squeeze(-1)
        return torch.cat([s, self.func(h), self.stop(h)], -1), h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-words", type=int, default=6, help="先用短句（<=N 词）")
    args = ap.parse_args()

    z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
    feats, caps = torch.from_numpy(z["feats"]).float(), [str(c) for c in z["captions"]]
    cnt = Counter(w for c in caps for w in words_of(c))
    content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
    c2i = {w: i for i, w in enumerate(content)}
    f2i = {w: i for i, w in enumerate(FUNC)}
    Vc, Fc = len(content), len(FUNC)

    keep = [i for i, c in enumerate(caps) if len(words_of(c)) <= args.max_words]
    print(f"短句样本 {len(keep)} / {len(caps)}，内容词 {Vc}，功能词 {Fc}")

    perm = torch.randperm(len(keep))
    tr = [keep[i] for i in perm[: int(len(keep) * 0.9)]]
    te = [keep[i] for i in perm[int(len(keep) * 0.9):]]

    model = BagArranger(n_content=Vc, n_func=Fc)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
    t0 = time.time()

    def alignment(i, topk):
        """把句子转成 (类型, 索引)：0=指针，1=功能词，2=停止。用 oracle 袋对齐。"""
        types, idxs = [], []
        for w in words_of(caps[i])[: MAXLEN - 1]:
            if w in c2i:
                wid = c2i[w]
                hit = [j for j, t in enumerate(topk) if t == wid]
                if hit:
                    types.append(0); idxs.append(hit[0]); continue
            types.append(1); idxs.append(f2i.get(w, 0))
        types.append(2); idxs.append(0)
        return types, idxs

    for step in range(1, args.steps + 1):
        idx = torch.tensor([tr[i] for i in torch.randint(0, len(tr), (args.batch,))])
        clip = feats[idx]
        h0, bag_logits = model.encode(clip)
        # oracle 袋：用真实句子里的内容词
        rows = []
        for i in idx.tolist():
            ids = [c2i[w] for w in words_of(caps[i]) if w in c2i][:K]
            ids += [0] * (K - len(ids))
            rows.append(ids)
        topk = torch.tensor(rows)
        bag_emb = model.emb_c(topk)

        # 袋头监督
        y = torch.zeros(len(idx), Vc)
        for r, i in enumerate(idx.tolist()):
            for w in set(words_of(caps[i])):
                if w in c2i:
                    y[r, c2i[w]] = 1.0
        l_bag = F.binary_cross_entropy_with_logits(bag_logits, y)

        h = torch.zeros(len(idx), 128)
        prev = torch.zeros(len(idx), 128)
        seqs = [alignment(i, rows[r]) for r, i in enumerate(idx.tolist())]
        L = max(len(s[0]) for s in seqs)
        loss_seq, n_tok = 0.0, 0
        for t in range(L):
            logits, h = model.step(h0, bag_emb, prev, h, t)
            ty = torch.tensor([s[0][t] if t < len(s[0]) else 2 for s in seqs])
            ix = torch.tensor([s[1][t] if t < len(s[1]) else 0 for s in seqs])
            ptr_l, fun_l, stop_l = logits[:, :K], logits[:, K:K + Fc], logits[:, -1]
            l = F.cross_entropy(ptr_l, torch.where(ty == 0, ix, torch.zeros_like(ix)), reduction="none") * (ty == 0).float()
            l = l + F.cross_entropy(fun_l, torch.where(ty == 1, ix.clamp(max=Fc - 1), torch.zeros_like(ix)), reduction="none") * (ty == 1).float()
            l = l + 2.0 * F.cross_entropy(torch.stack([-stop_l, stop_l], -1), (ty == 2).long(), reduction="none")
            loss_seq = loss_seq + l.sum(); n_tok += len(idx)
            # 下一步输入：默认用真值，但按概率换成模型自己的预测（scheduled sampling）
            ss_p = min(0.5, step / 400 * 0.5)
            use_pred = torch.rand(1).item() < ss_p
            pred_type = logits.argmax(-1)
            if use_pred:
                is_ptr = (pred_type < K).float().unsqueeze(-1)
                pi = pred_type.clamp(max=K - 1)
                fi = (pred_type - K).clamp(0, Fc - 1)
                prev = model.emb_c(topk.gather(1, pi.unsqueeze(1)).squeeze(1)) * is_ptr + \
                       model.emb_f(fi) * (1 - is_ptr)
            else:
                prev_c = model.emb_c(torch.tensor([c2i.get(words_of(caps[i])[t] if t < len(words_of(caps[i])) else '', 0) for i in idx.tolist()]))
                prev_f = model.emb_f(torch.tensor([f2i.get(words_of(caps[i])[t] if t < len(words_of(caps[i])) else '', 0) for i in idx.tolist()]))
                is_ptr = (ty == 0).float().unsqueeze(-1)
                prev = prev_c * is_ptr + prev_f * (1 - is_ptr)

        loss = l_bag + loss_seq / max(n_tok, 1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"step {step} bag {l_bag.item():.3f} seq {loss_seq.item()/max(n_tok,1):.3f} "
                  f"{(time.time()-t0)/step:.2f}s/step", flush=True)

    # 评测：贪心排列（不再回灌 id 0）
    model.eval()
    def gen(i):
        with torch.no_grad():
            h0, _ = model.encode(feats[i:i + 1])
            ids = [c2i[w] for w in words_of(caps[i]) if w in c2i][:K]
            ids += [0] * (K - len(ids))
            topk = torch.tensor([ids]); bag_emb = model.emb_c(topk)
            h = torch.zeros(1, 128); prev = torch.zeros(1, 128); out = []
            for t in range(MAXLEN):
                logits, h = model.step(h0, bag_emb, prev, h, t)
                ptr_l, fun_l, stop_l = logits[:, :K], logits[:, K:K + Fc], logits[:, -1]
                scores = torch.cat([ptr_l, fun_l, stop_l.unsqueeze(-1)], -1).squeeze(0)
                j = int(scores.argmax())
                if j >= K + Fc:
                    break
                if j < K:
                    w = content[int(topk[0, j])]; prev = model.emb_c(torch.tensor([c2i.get(w, 0)]))
                else:
                    w = FUNC[j - K]; prev = model.emb_f(torch.tensor([j - K]))
                out.append(w)
            return out

    f1s, exact = [], []
    for i in te[:100]:
        pred = gen(i); gold = words_of(caps[i])
        inter = Counter(pred) & Counter(gold)
        tp = sum(inter.values())
        f1 = 2 * tp / max(len(pred) + len(gold), 1)
        f1s.append(f1); exact.append(pred[:6] == gold[:6])
    print(f"\n词级 F1 {np.mean(f1s)*100:.1f}%   前6词完全匹配 {np.mean(exact)*100:.0f}%")
    for i in te[:5]:
        print(f"  生成: {' '.join(gen(i))[:56]}")
        print(f"  参考: {caps[i][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
