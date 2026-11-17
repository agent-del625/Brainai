#!/usr/bin/env python3
"""袋 + 排列器（v4 的文本输出路线）。

流程：
  CLIP 特征 → 袋头（多标签，选出 k 个关键词） → 排列器（指针 + 功能词表） → 句子

排列器每一步二选一：
  · 从袋里"指"一个词（pointer）
  · 从功能词表里插入一个词（the/a/is/on…）

支持课程：先用短句（小山）训，再用全量（大山）微调，对比从零训全量。
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/srv/brainai")
K = 8                      # 袋大小
MAXLEN = 24                # 句子最大词数
FUNC = "a an the is are was were in on at of and or to with for by from as it its this that there two one three his her their".split()


def words_of(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s.lower())


class BagArranger(nn.Module):
    def __init__(self, clip_dim: int = 512, d: int = 128, content_vocab: int = 800, func_vocab: int = 40):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(clip_dim, 256), nn.GELU(), nn.LayerNorm(256))
        self.bag_head = nn.Linear(256, content_vocab)
        self.wemb = nn.Embedding(content_vocab, d)         # 袋中词的嵌入
        self.pos = nn.Embedding(MAXLEN, d)
        self.gru = nn.GRUCell(d + 256 + d, d)
        self.att = nn.Linear(d, d)
        self.ptr = nn.Linear(d, 1)                         # 对袋中每个词的打分
        self.func = nn.Linear(d, func_vocab)
        self.stop = nn.Linear(d, 1)

    def forward(self, clip: torch.Tensor):
        h0 = self.enc(clip)                                # (B,256)
        bag_logits = self.bag_head(h0)                     # (B,Vc)
        topk = bag_logits.topk(K, dim=1).indices           # (B,K)
        bag_emb = self.wemb(topk)                          # (B,K,d)
        return h0, bag_logits, topk, bag_emb

    def decode(self, h0, bag_emb, prev_emb, h, t: int = 0):
        if t >= MAXLEN:
            t = MAXLEN - 1
        pe = self.pos(torch.full((h.size(0),), t, dtype=torch.long))
        h = self.gru(torch.cat([prev_emb, h0, pe], -1), h)
        scores = self.ptr(torch.tanh(self.att(bag_emb)) + h.unsqueeze(1)).squeeze(-1)  # (B,K)
        logits = torch.cat([scores, self.func(h), self.stop(h)], -1)                   # (B,K+Fc+1)
        return logits, h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-small", type=int, default=600)
    ap.add_argument("--steps-big", type=int, default=600)
    ap.add_argument("--max-words-small", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--oracle", action="store_true", help="用真实关键词袋（只练排列）")
    args = ap.parse_args()

    z = np.load(ROOT / "data/image/clip_feats.npz", allow_pickle=True)
    feats = torch.from_numpy(z["feats"]).float()
    caps = [str(c) for c in z["captions"]]

    cnt = Counter(w for c in caps for w in words_of(c))
    content = [w for w, n in cnt.most_common(800) if n >= 5 and w not in FUNC]
    c2i = {w: i for i, w in enumerate(content)}
    f2i = {w: i for i, w in enumerate(FUNC)}
    Vc, Fc = len(content), len(FUNC)
    print(f"内容词 {Vc} 个，功能词 {Fc} 个，样本 {len(caps)}")

    n = len(caps)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[: int(n * 0.9)], perm[int(n * 0.9):]

    def target_seq(cap: str, topk_row: torch.Tensor):
        """把句子编码成 (类型, 索引)：0=指针，1=功能词，2=停止。"""
        types, idxs = [], []
        bag_set = set(topk_row.tolist())
        for w in words_of(cap)[: MAXLEN - 1]:
            if w in c2i:
                pos = content.index(w) if False else None
        # 指针位置需要按"这个词在袋中的位置"来定，袋是模型预测的，训练时用预测袋
        for w in words_of(cap)[: MAXLEN - 1]:
            if w in c2i:
                wid = c2i[w]
                hit = [j for j, t in enumerate(topk_row.tolist()) if t == wid]
                if hit:
                    types.append(0); idxs.append(hit[0]); continue
            if w in f2i:
                types.append(1); idxs.append(f2i[w]); continue
            types.append(1); idxs.append(0)      # 兜底当 a
        types.append(2); idxs.append(0)
        return torch.tensor(types), torch.tensor(idxs)

    def build_batch(idx):
        clip = feats[idx]
        model_ = None
        return clip

    model = BagArranger(clip_dim=feats.size(1), content_vocab=Vc, func_vocab=Fc)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def run(data_idx, steps, tag: str):
        t0 = time.time()
        for step in range(1, steps + 1):
            idx = data_idx[torch.randint(0, len(data_idx), (args.batch,))]
            clip = feats[idx]
            h0, bag_logits, topk, bag_emb = model(clip)
            if args.oracle:                     # 小山：先用真袋，只练排列
                rows = []
                for i in idx.tolist():
                    ids = [c2i[w] for w in words_of(caps[i]) if w in c2i][:K]
                    ids += [0] * (K - len(ids))
                    rows.append(ids)
                topk = torch.tensor(rows)
                bag_emb = model.wemb(topk)
            # 袋头监督
            y = torch.zeros(len(idx), Vc)
            for r, i in enumerate(idx.tolist()):
                for w in set(words_of(caps[i])):
                    if w in c2i:
                        y[r, c2i[w]] = 1.0
            l_bag = F.binary_cross_entropy_with_logits(bag_logits, y)
            # 排列器监督（teacher forcing）
            h = torch.zeros(len(idx), 128)
            prev = torch.zeros(len(idx), model.wemb.embedding_dim)
            loss_seq, cnt_tok = 0.0, 0
            seqs = [target_seq(caps[i], topk[r]) for r, i in enumerate(idx.tolist())]
            maxlen = max(s[0].size(0) for s in seqs)
            for t in range(maxlen):
                logits, h = model.decode(h0, bag_emb, prev, h, t)
                tgt_type = torch.tensor([s[0][t].item() if t < s[0].size(0) else 2 for s in seqs])
                tgt_idx = torch.tensor([s[1][t].item() if t < s[1].size(0) else 0 for s in seqs])
                ptr_l = logits[:, :K]
                fun_l = logits[:, K:K + Fc]
                stop_l = logits[:, -1]
                ptr_tgt = torch.where(tgt_type == 0, tgt_idx, torch.zeros_like(tgt_idx))
                fun_tgt = torch.where(tgt_type == 1, tgt_idx.clamp(max=Fc - 1), torch.zeros_like(tgt_idx))
                l = F.cross_entropy(ptr_l, ptr_tgt, reduction="none") * (tgt_type == 0).float()
                l = l + F.cross_entropy(fun_l, fun_tgt, reduction="none") * (tgt_type == 1).float()
                stop_logits = torch.stack([-stop_l, stop_l], -1)
                l = l + F.cross_entropy(stop_logits, (tgt_type == 2).long(), reduction="none")
                loss_seq = loss_seq + l.sum(); cnt_tok += len(idx)
                # 下一步输入：用真实词（teacher forcing）
                pure = torch.tensor([c2i.get(words_of(caps[i])[t], 0) if t < len(words_of(caps[i])) else 0
                                     for i in idx.tolist()])
                prev = model.wemb(pure)
                if torch.rand(1).item() < 0.5:
                    prev = torch.zeros_like(prev)
            loss = l_bag + loss_seq / max(cnt_tok, 1)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 100 == 0:
                print(f"[{tag}] step {step} bag {l_bag.item():.3f} seq {loss_seq.item()/max(cnt_tok,1):.3f} "
                      f"{(time.time()-t0)/step:.2f}s/step", flush=True)
        return model

    # 小山：短句
    short = torch.tensor([i for i in train_idx.tolist() if len(words_of(caps[i])) <= args.max_words_small])
    print(f"小山（≤{args.max_words_small} 词）：{len(short)} 条")
    run(short, args.steps_small, "小山")

    # 大山：全量微调
    print("转移到大山：全量微调")
    run(train_idx, args.steps_big, "大山")

    # 评测：贪心排列
    def generate(clip):
        h0, bag_logits, topk, bag_emb = model(clip)
        h = torch.zeros(1, 128); prev = torch.zeros(1, model.wemb.embedding_dim)
        out = []
        for _ in range(MAXLEN):
            logits, h = model.decode(h0, bag_emb, prev, h, len(out))
            ptr_l, fun_l, stop_l = logits[:, :K], logits[:, K:K + Fc], logits[:, -1]
            scores = torch.cat([ptr_l, fun_l, stop_l.unsqueeze(-1)], -1).squeeze(0)
            j = int(scores.argmax())
            if j >= K + Fc:
                break
            if j < K:
                w = content[int(topk[0, j])]
            else:
                w = FUNC[j - K]
            out.append(w)
            prev = model.wemb(torch.tensor([c2i.get(w, 0)]))
        return " ".join(out)

    print("\n生成样例：")
    with torch.no_grad():
        for i in test_idx[:6].tolist():
            print(f"  生成: {generate(feats[i:i+1])[:60]}")
            print(f"  参考: {caps[i][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
