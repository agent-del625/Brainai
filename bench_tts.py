#!/usr/bin/env python3
"""对比两种 TTS 的首字节延迟：本地 espeak-ng vs 云端 MiniMax。"""
import time, subprocess, tempfile, os
from pathlib import Path
texts=["The capital of France is Paris.","Two plus two is four.","A banana is yellow."]
def espeak_latency(t):
    f=tempfile.mktemp(suffix=".wav")
    t0=time.time()
    subprocess.run(["espeak-ng","-w",f,"-s","170",t],capture_output=True)
    dt=time.time()-t0
    sz=os.path.getsize(f) if os.path.exists(f) else 0
    os.path.exists(f) and os.remove(f)
    return dt, sz
print("== 本地 espeak-ng（离线）")
for t in texts:
    dt,sz=espeak_latency(t)
    print(f"  {dt*1000:7.0f} ms  音频 {sz//1024} KB   ← {t[:40]}")
