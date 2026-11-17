#!/usr/bin/env python3
"""对照组：把加法拆成"逐列 + 进位"的可验证子步骤，看循环是否有用。"""
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

def batch(n):
    a = torch.randint(0, 1000, (n,)); b = torch.randint(0, 1000, (n,))
    da = torch.stack([(a // 10**i) % 10 for i in range(3)], 1)   # 个位在前
    db = torch.stack([(b // 10**i) % 10 for i in range(3)], 1)
    s = a + b
    ds = torch.stack([(s // 10**i) % 10 for i in range(4)], 1)
    return da, db, ds

class ColNet(nn.Module):
    """每轮处理一列：输入两位数字 + 当前进位，输出这一位 + 新进位。"""
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(10, d)
        self.cell = nn.GRUCell(d * 2 + 1, d)
        self.out = nn.Linear(d, 10)
        self.carry = nn.Linear(d, 1)

    def forward(self, da, db):
        b = da.size(0)
        h = torch.zeros(b, 64)
        carry = torch.zeros(b, 1)
        digits = []
        for i in range(3):
            x = torch.cat([self.emb(da[:, i]), self.emb(db[:, i]), carry], 1)
            h = self.cell(x, h)
            digits.append(self.out(h))
            carry = torch.sigmoid(self.carry(h))          # 0/1 进位
        digits.append(torch.zeros(b, 10))                 # 最高位占位
        return torch.stack(digits, 1), carry

m = ColNet(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
for step in range(1500):
    da, db, ds = batch(64)
    logits, carry = m(da, db)
    loss = F.cross_entropy(logits[:, :3].reshape(-1, 10), ds[:, :3].reshape(-1))
    loss += 0.5 * F.binary_cross_entropy(carry.squeeze(-1).clamp(1e-4, 1-1e-4), (ds[:, 3] > 0).float())
    opt.zero_grad(); loss.backward(); opt.step()

da, db, ds = batch(512)
with torch.no_grad():
    logits, carry = m(da, db)
    pred = logits[:, :3].argmax(-1)
    acc = (pred == ds[:, :3]).all(1).float().mean().item() * 100
    digit_acc = (pred == ds[:, :3]).float().mean().item() * 100
print(f"逐列+进位：整题准确率 {acc:.1f}%   单列准确率 {digit_acc:.1f}%")
