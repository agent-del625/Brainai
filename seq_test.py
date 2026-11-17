#!/usr/bin/env python3
"""验证第二步：给"正确的关键词袋"，模型能不能排出通顺句子。"""
import random, re
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0)
CAPS = ["a boy laying face down on a skateboard", "a boy in a blue top is jumping off a swing",
        "a lady walking her dog through an obstacle course", "a big black dog jumps in the air to catch a ball",
        "a girl in a red shirt is sitting on a bench", "two women climbing rocks around a river",
        "a man riding a bicycle down a dirt road", "a white dog running on the grass"]
# 扩充：用 6060 条真实 caption
import numpy as np
z = np.load("data/image/clip_feats.npz", allow_pickle=True)
CAPS = [str(c) for c in z["captions"]]
chars = sorted({ch for c in CAPS for ch in c})
vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
for ch in chars: vocab[ch] = len(vocab)
L = 32

def enc(s):
    ids = [2] + [vocab.get(ch, 1) for ch in s][:L-1]
    return ids + [0]*(L-len(ids))

class Seq(nn.Module):
    """条件：关键词袋（词嵌入均值）+ 前缀字符 → 下一字。"""
    def __init__(self, n_words, d=96, ch=192):
        super().__init__()
        self.wemb = nn.Embedding(n_words, d, padding_idx=0)
        self.emb = nn.Embedding(len(vocab), d)
        self.inp = nn.Conv1d(d*2, ch, 1)
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
                                 nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
                                 nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(ch, len(vocab), 1))
    def forward(self, toks, bag):
        lg = self.logits(toks, bag)
        return lg[:, :, :-1], toks[:, 1:]

    def logits(self, toks, bag):
        we = self.wemb(bag)
        m_ = (bag != 0).float().unsqueeze(-1)
        b = (we * m_).sum(1) / m_.sum(1).clamp(min=1)
        e = self.emb(toks).transpose(1, 2)
        bb = b.unsqueeze(-1).expand(-1, -1, e.size(-1))
        return self.net(self.inp(torch.cat([e, bb], 1)))

tok = lambda s: re.findall(r"[a-z']+", s.lower())
words = sorted({w for c in CAPS for w in tok(c)})
w2i = {w: i for i, w in enumerate(words)}
MAXW = 12
def bag_ids(c):
    ids = [w2i[w] for w in tok(c)][:MAXW] or [0]
    return ids + [0] * (MAXW - len(ids))
bags = torch.tensor([bag_ids(c) for c in CAPS])
toks = torch.tensor([enc(c) for c in CAPS])
print("词表", len(words), "字符表", len(vocab), "样本", len(CAPS))

m = Seq(len(words)); opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
for step in range(1, 1501):
    idx = torch.randperm(len(CAPS))[:48]
    lg, tg = m(toks[idx], bags[idx])
    loss = F.cross_entropy(lg.reshape(-1, len(vocab)), tg.reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 300 == 0: print(f"step {step} ce {loss.item():.3f}")

inv = {v: k for k, v in vocab.items()}
m.eval()
with torch.no_grad():
    for i in range(4):
        out = torch.full((1, 1), 2, dtype=torch.long)
        for _ in range(30):
            lg = m.logits(out, bags[i:i+1])
            nxt = lg[:, :, -1].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
            if nxt.item() == 2: break
        txt = "".join(inv.get(int(c), "") for c in out[0]).replace("<eos>", "")
        print(f"  给出关键词 {[words[j] for j in bags[i][:5].tolist()]}")
        print(f"  排列成句 → {txt[:52]}")
        print(f"  参考答案   {CAPS[i][:52]}")
