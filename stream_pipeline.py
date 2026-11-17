#!/usr/bin/env python3
"""流式问答管道：边生成边分句边合成，测端到端首音延迟。"""
import time, subprocess, tempfile, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, re, pickle
from pathlib import Path
ROOT=Path("/srv/brainai")
# --- 载入 40M 语言模型（本地小模型，用于演示流式管道）---
vocab=pickle.load(open("/tmp/tok_vocab_local.pkl","rb")) if os.path.exists("/tmp/tok_vocab_local.pkl") else None
if vocab is None:
    print("本地未准备词表，使用占位生成器（只测管道延迟）")
    vocab=None

# --- 模拟器：按真实速度产 token（树莓派 22 token/s）---
def fake_token_stream(question, tok_per_sec=22.0):
    """按树莓派的真实出词速度产 token，用来测管道延迟"""
    answer=" the capital of france is paris . paris is the largest city of france ."
    toks=answer.split()
    dt=1.0/tok_per_sec
    for t in toks:
        time.sleep(dt)
        yield t+" "

def split_sentences(stream):
    """流式分句：攒到句号/问号就吐出一句"""
    buf=""
    for tok in stream:
        buf+=tok
        if re.search(r"[.!?]\s*$", buf.strip()) and len(buf.strip())>3:
            yield buf.strip(); buf=""
    if buf.strip(): yield buf.strip()

def tts_local(text, out):
    t0=time.time()
    subprocess.run(["espeak-ng","-w",out,"-s","170",text],capture_output=True)
    return time.time()-t0

def run(question, tok_per_sec=22.0):
    print(f"问：{question}")
    t_start=time.time()
    first_audio=None; total_audio=0.0
    for i,sent in enumerate(split_sentences(fake_token_stream(question, tok_per_sec=tok_per_sec))):
        t_tts0=time.time()
        out=tempfile.mktemp(suffix=".wav")
        lat=tts_local(sent,out)
        if first_audio is None:
            first_audio=time.time()-t_start
        total_audio+=lat
        sz=os.path.getsize(out) if os.path.exists(out) else 0
        os.path.exists(out) and os.remove(out)
        print(f"  第{i+1}句（打字完 {time.time()-t_start:.2f}s，TTS {lat*1000:.0f}ms，{sz//1024}KB）：{sent}")
    print(f"→ 首音延迟 {first_audio*1000:.0f} ms   总耗时 {time.time()-t_start:.2f}s")

if __name__=="__main__":
    run("What is the capital of France?")
