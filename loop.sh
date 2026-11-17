#!/bin/bash
# 连续训练：按课程轮换 img → audio → video，每阶段 1500 步，结束立刻下一阶段
cd /srv/brainai || exit 1
STAGES=(img audio video)
ROUND=0
while true; do
  ST=${STAGES[$((ROUND % 3))]}
  ROUND=$((ROUND+1))
  echo "=== $(date '+%F %T') 第 $ROUND 轮（stage=$ST）开始 ===" >> logs/loop.log
  python3 -u train.py --data real --steps 1500 --batch 8 --mix 0.7 --stage "$ST" --prefix-dropout 0.5 \
      >> "logs/round-$(date +%m%d).log" 2>&1
  echo "=== $(date '+%F %T') 第 $ROUND 轮（stage=$ST）结束，立即下一轮 ===" >> logs/loop.log
  if [ $((ROUND % 3)) -eq 0 ]; then ./eval_job.sh >/dev/null 2>&1; fi
  sleep 3
done
