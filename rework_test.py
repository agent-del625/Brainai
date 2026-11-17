#!/usr/bin/env python3
"""返工策略对比：错误发生在不同位置时，"局部返工"和"从头重做"谁能救回来。"""
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
        self.out = nn.Linear(d, 10); self.carry = nn.Linear(d, 1)
    def forward(self, da, db):
        b = da.size(0); h = torch.zeros(b, 64); carry = torch.zeros(b, 1); digits = []
        for i in range(3):
            x = torch.cat([self.emb(da[:, i]), self.emb(db[:, i]), carry], 1)
            h = self.cell(x, h); digits.append(self.out(h)); carry = torch.sigmoid(self.carry(h))
        digits.append(torch.zeros(b, 10))
        return torch.stack(digits, 1), carry, h

m = ColNet(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
for step in range(1500):
    da, db, ds = batch(64)
    logits, carry, _ = m(da, db)
    loss = F.cross_entropy(logits[:, :3].reshape(-1, 10), ds[:, :3].reshape(-1))
    loss += 0.5 * F.binary_cross_entropy(carry.squeeze(-1).clamp(1e-4, 1-1e-4), (ds[:, 3] > 0).float())
    opt.zero_grad(); loss.backward(); opt.step()

da, db, ds = batch(256)
with torch.no_grad():
    logits, carry, _ = m(da, db)
print("基线整题准确率:", round((logits[:, :3].argmax(-1) == ds[:, :3]).all(1).float().mean().item()*100, 1), "%")

# 模拟"某一步把进位搞错"，再分别用不同策略修复
def run_with_corruption(corrupt_step, policy):
    """在 corrupt_step 注入错误进位；policy: none / local / backtrack / restart"""
    b = da.size(0)
    with torch.no_grad():
        h = torch.zeros(b, 64); carry = torch.zeros(b, 1); preds = []
        for i in range(3):
            if i == corrupt_step:
                carry = 1 - carry            # 注入错误
            if policy == "local" and i == corrupt_step:
                carry = torch.zeros(b, 1)    # 局部返工：这一步重新拿一个干净状态
            if policy == "backtrack" and i == corrupt_step + 1:
                h = torch.zeros(b, 64); carry = torch.zeros(b, 1); preds = []   # 回到开头重算
            x = torch.cat([m.emb(da[:, i]), m.emb(db[:, i]), carry], 1)
            h = m.cell(x, h); preds.append(m.out(h).argmax(-1)); carry = torch.sigmoid(m.carry(h))
        if policy == "restart":
            h = torch.zeros(b, 64); carry = torch.zeros(b, 1); preds = []
            for i in range(3):
                x = torch.cat([m.emb(da[:, i]), m.emb(db[:, i]), carry], 1)
                h = m.cell(x, h); preds.append(m.out(h).argmax(-1)); carry = torch.sigmoid(m.carry(h))
        return torch.stack(preds, 1)

print("\n注入错误位置 → 各策略的整题准确率")
print(f"{'错误位置':<10}{'不修':>8}{'局部返工':>10}{'从头重做':>10}")
for k in (0, 1, 2):
    row = [run_with_corruption(k, p) for p in ("none", "local", "restart")]
    accs = [round((p == ds[:, :3]).all(1).float().mean().item()*100, 1) for p in row]
    print(f"{'第'+str(k+1)+'列':<10}{accs[0]:>8}{accs[1]:>10}{accs[2]:>10}")
