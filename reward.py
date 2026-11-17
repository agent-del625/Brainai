"""奖励函数：不靠规则约束，用"和标准答案的差距"来打分。

R = 0.45·集合F1 + 0.30·顺序分(LCS) + 0.15·长度贴合 - 0.25·重复惩罚 - 0.15·偏离惩罚
"""
from collections import Counter


def lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


FUNC = set("a an the is are was were in on at of and or to with for by from as it its his her their this that there two one three".split())


def reward(pred: list[str], gold: list[str]) -> dict:
    if not pred:
        return {"r": -0.5, "f1": 0.0, "order": 0.0, "len": 0.0, "rep": 1.0}
    inter = Counter(pred) & Counter(gold)
    tp = sum(inter.values())
    prec = tp / len(pred)
    rec = tp / max(len(gold), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)

    order = lcs(pred, gold) / max(len(gold), 1)
    len_fit = 1.0 - min(abs(len(pred) - len(gold)) / max(len(gold), 1), 1.0)

    # 重复惩罚：同一个词出现次数超过 3 次就罚
    cnt = Counter(pred)
    rep = max(cnt.values()) / len(pred) if pred else 1.0
    rep_pen = max(0.0, rep - 0.4) * 2.0

    # 内容词覆盖率：功能词不算分（否则 "a a a a" 也能拿分）
    pc = [w for w in pred if w not in FUNC]
    gc = [w for w in gold if w not in FUNC]
    c_inter = Counter(pc) & Counter(gc)
    cover = sum(c_inter.values()) / max(len(gc), 1)
    if not pc:
        cover = 0.0

    r = (0.25 * f1 + 0.35 * cover + 0.25 * order + 0.15 * len_fit
         - 0.30 * rep_pen - 0.25 * (1.0 if cover == 0 else 0.0))
    return {"r": r, "f1": f1, "cover": cover, "order": order, "len": len_fit, "rep": rep}
