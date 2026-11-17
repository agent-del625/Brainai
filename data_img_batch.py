#!/usr/bin/env python3
"""批量把 Flickr8k 训练 parquet 转成图片集（一次跑完两个分片）。"""
import glob
import subprocess
import sys

for p in sorted(glob.glob("/srv/brainai/data/downloads/f8k_train*.parquet")):
    print("处理", p, flush=True)
    subprocess.run([sys.executable, "/srv/brainai/data_img.py", "--parquet", p, "--limit", "5000"])
