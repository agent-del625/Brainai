#!/bin/bash
# 训练后自动评测钩子：跑统一评测，产出 logs/baseline.json（最新）+ 归档，
# 并把「统一模型」关键指标追加到 logs/eval_history.csv 供看趋势。
set -u
ROOT="${BRAINAI_ROOT:-/srv/brainai}"
cd "$ROOT" || exit 1
[ -f ckpt/model.pt ] || exit 0

# 跑统一评测（缺失的资产各自记 skip，不中断整次评测）
nice -n 15 python3 eval_all.py --root "$ROOT" --n 32 2>/dev/null

# 从最新基线里抽「统一模型」关键指标，追加进历史表（表头一次，之后每次一行）
python3 - "$ROOT/logs/baseline.json" "$ROOT/logs/eval_history.csv" <<'PY'
import json, sys, time, os
baseline = sys.argv[1]
history = sys.argv[2]
with open(baseline, encoding="utf-8") as f:
    data = json.load(f)
u = data.get("probes", {}).get("unified", {})
m = u.get("metrics", {})
row = [
    time.strftime("%Y-%m-%d %H:%M"),
    str(m.get("最优分支", "")),
    str(m.get("重建PSNR", "")),
    str(m.get("视频→文本_R@1", "")),
    str(m.get("音频→文本_R@1", "")),
    str(m.get("生成字符准确率", "")),
]
new = not os.path.exists(history)
with open(history, "a", encoding="utf-8") as f:
    if new:
        f.write("time,best_branch,recon_psnr,v2t_r1,a2t_r1,char_acc\n")
    f.write(",".join(row) + "\n")
PY
