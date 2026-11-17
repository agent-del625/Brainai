#!/bin/bash
# 周期性评测，把指标追加到 logs/eval_history.csv，方便看趋势
cd /srv/brainai || exit 1
[ -f ckpt/model.pt ] || exit 0
nice -n 15 python3 eval.py --n 32 2>/dev/null | python3 -c "
import json,sys
txt=sys.stdin.read()
try:
    start=txt.index('{'); end=txt.rindex('}')+1
    m=json.loads(txt[start:end])
except Exception:
    sys.exit(0)
import time
row=[time.strftime('%Y-%m-%d %H:%M'), str(m.get('最优分支')), str(m.get('重建PSNR')),
     str(m.get('视频→文本_R@1')), str(m.get('音频→文本_R@1')), str(m.get('生成字符准确率'))]
with open('/srv/brainai/logs/eval_history.csv','a') as f:
    f.write(','.join(row)+'\n')
"
