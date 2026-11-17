#!/usr/bin/env python3
"""验证"逻辑袋 + 多次循环 + 复核"：多位加法需要进位，单次前向不够，多轮循环应该变好。

结构：编码 → 逻辑袋(4 槽) → [循环：解码草图 → 复核重编码 → 更新袋] → 输出
对比：循环 1 / 4 / 8 次，看准确率。
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0); random.seed(0)
D, SLOTS, DIG = 96, 4, 3
V = 12          # 0-9 + '+' + '='
LEN = 2 * DIG + 1

def make_batch(n):
    a = torch.randint(0, 1000, (n,))
    b = torch.randint(0, 1000, (n,))
    xs, ys = [], []
    for i in range(n):
        sa, sb = f"{a[i]:03d}", f"{b[i]:03d}"
        seq = [int(c) for c in sa] + [10] + [int(c) for c in sb]
        xs.append(seq)
        s = a[i].item() + b[i].item()
        ys.append([int(c) for c in f"{s:04d}"[:DIG]][-DIG:])
    return torch.tensor(xs), torch.tensor(ys)

class LoopNet(nn.Module):
    def __init__(self, dim=D, slots=SLOTS, loops=4):
        super().__init__()
        self.loops = loops
        self.emb = nn.Embedding(V, dim)
        self.enc = nn.Sequential(nn.Conv1d(dim, 128, 3, padding=1), nn.GELU(),
                                 nn.Conv1d(128, dim, 3, padding=1))
        self.slots0 = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.cell = nn.GRUCell(dim * 2, dim)
        self.dec = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, DIG * 10))
        self.verify = nn.Sequential(nn.Linear(dim * 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.halt = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        cur = x                                       # 当前输入（每轮把上一轮输出拼进来）
        s = self.slots0.expand(x.size(0), -1, -1)
        outs, halts, verifies = [], [], []
        for _ in range(self.loops):
            e = self.emb(cur).transpose(1, 2)
            ctx = self.enc(e).mean(-1)
            flat = s.reshape(-1, s.size(-1))          # (B*S,D)
            h = ctx.repeat_interleave(s.size(1), 0)   # (B*S,D)
            new = self.cell(torch.cat([flat, h], 1), flat)
            s = new.view(s.size(0), -1, s.size(-1))
            bag = s.mean(1)                           # (B,D) 当前逻辑袋
            logits = self.dec(bag).view(-1, DIG, 10)
            outs.append(logits)
            # 关键：把这一轮的答案当成新输入拼回去（相当于"写下中间结果再继续想"）
            if self.training or True:
                guess = logits.argmax(-1) + 1              # 1..10，避免和 pad 冲突
                cur = torch.cat([x, guess], 1)
            halts.append(self.halt(bag).squeeze(-1))
            # 复核：把当前输出重新编码，和原始上下文比对
            verifies.append(self.verify(torch.cat([bag, ctx], 1)).squeeze(-1))
        return outs, halts, verifies

def run(loops, steps=600):
    torch.manual_seed(1)
    m = LoopNet(loops=loops)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    for step in range(steps):
        x, y = make_batch(64)
        outs, halts, verifies = m(x)
        # 每一轮的输出都算损失，越靠后权重越高（最终答案以最后一轮为准）
        loss = sum(F.cross_entropy(o.reshape(-1, 10), y.reshape(-1)) * (0.3 + 0.7 * (i + 1) / loops)
                   for i, o in enumerate(outs))
        opt.zero_grad(); loss.backward(); opt.step()
    x, y = make_batch(512)
    with torch.no_grad():
        outs, _, _ = m(x)
        acc = [(o.argmax(-1) == y).all(1).float().mean().item() * 100 for o in outs]
    print(f"循环 {loops} 轮：各轮准确率 " + " → ".join(f"{a:.1f}%" for a in acc))
    return acc

for L in (1, 2, 4, 8):
    run(L)
