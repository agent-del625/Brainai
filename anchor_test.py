#!/usr/bin/env python3
"""锚点实验：只有两端监督 vs 中间有锚点，哪条路更容易走到正确解。

任务：三位数加法（按列计算，天然有中间结果）。
k = 中间锚点数（0~3），k 越大相当于"山中间有越多队伍同时开凿"。
"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

def batch(n):
    a = torch.randint(0, 1000, (n,)); b = torch.randint(0, 1000, (n,))
    da = torch.stack([(a // 10**i) % 10 for i in range(3)], 1)
    db = torch.stack([(b // 10**i) % 10 for i in range(3)], 1)
    s = a + b
    ds = torch.stack([(s // 10**i) % 10 for i in range(4)], 1)
    return da, db, ds

class ColNet(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(10, d); self.cell = nn.GRUCell(d*2+1, d)
        self.out = nn.Linear(d, 10); self.carry = nn.Linear(d, 1); self.ans = nn.Linear(d, 100)
    def forward(self, da, db):
        b = da.size(0); h = torch.zeros(b, 64); carry = torch.zeros(b, 1); digits = []
        for i in range(3):
            x = torch.cat([self.emb(da[:, i]), self.emb(db[:, i]), carry], 1)
            h = self.cell(x, h); digits.append(self.out(h)); carry = torch.sigmoid(self.carry(h))
        final = self.ans(h)          # 只从最终隐状态出答案（端到端监督用）
        return torch.stack(digits, 1), carry, final

def run(k_anchors, steps=1200):
    torch.manual_seed(1)
    m = ColNet(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for step in range(steps):
        da, db, ds = batch(64)
        digits, carry, final = m(da, db)
        # 两端监督：最终答案
        loss = F.cross_entropy(final, ds[:, 3] * 10 + ds[:, 2]) * 0.5
        # 锚点监督：前 k 列（从左到右，代表"中途的已知节点"）
        if k_anchors > 0:
            k = min(k_anchors, 3)
            loss = loss + F.cross_entropy(digits[:, :k].reshape(-1, 10), ds[:, :k].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    da, db, ds = batch(512)
    with torch.no_grad():
        digits, carry, final = m(da, db)
        # 端到端答案正确率
        s_pred = digits.argmax(-1)
        acc_all = (s_pred == ds[:, :3]).all(1).float().mean().item() * 100
        # "过程正确"：每一列都和真实一致
        proc = (s_pred == ds[:, :3]).float().mean().item() * 100
    print(f"锚点数 {k_anchors}：整题正确率 {acc_all:.1f}%   过程逐步正确率 {proc:.1f}%")

for k in (0, 1, 2, 3):
    run(k)
