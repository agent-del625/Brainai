#!/usr/bin/env python3
"""对比三种 TTS：espeak（机械）/ Piper（本地神经）/ MiniMax（云端）。"""
import json, os, time, wave, subprocess, urllib.request, numpy as np
from pathlib import Path
ROOT=Path("/srv/brainai"); OUT=ROOT/"logs/tts_cmp"; OUT.mkdir(parents=True,exist_ok=True)
TEXT="The capital of France is Paris. It is the largest city in the country."
def espeak(path):
    t0=time.time(); subprocess.run(["espeak-ng","-w",str(path),"-s","165",TEXT],capture_output=True); return time.time()-t0
def piper(path):
    t0=time.time()
    subprocess.run(["python3","-m","piper","-m",str(ROOT/"data/piper/en_US-lessac-medium.onnx"),
                    "-f",str(path)],input=TEXT.encode(),capture_output=True)
    return time.time()-t0
def minimax(path):
    key=open(ROOT/".mmkey").read().strip()
    body=json.dumps({"model":"speech-2.8-hd","text":TEXT,"stream":False,"output_format":"hex",
                     "voice_setting":{"voice_id":"male-qn-qingse","speed":1.0}}).encode()
    req=urllib.request.Request("https://api.minimaxi.com/v1/t2a_v2",data=body,
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=60) as r: d=json.loads(r.read())
    open(path,"wb").write(bytes.fromhex(d["data"]["audio"]))
    return time.time()-t0
for name,fn in (("espeak",espeak),("piper",piper),("minimax",minimax)):
    p=OUT/f"{name}.wav"
    try:
        dt=fn(p)
        sz=p.stat().st_size/1024 if p.exists() else 0
        print(f"{name:8s} 延迟 {dt*1000:6.0f} ms  文件 {sz:.0f} KB")
    except Exception as e:
        print(f"{name:8s} 失败: {e}")
