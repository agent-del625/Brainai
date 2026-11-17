#!/usr/bin/env python3
"""本地流式 ASR（sherpa-onnx zipformer 中英双语）测速与准确率。"""
import time, subprocess, tempfile, os, numpy as np, soundfile as sf, sherpa_onnx
from pathlib import Path
ROOT=Path("/srv/brainai/data/asr")
rec=sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(ROOT/"tokens.txt"),
    encoder=str(ROOT/"encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(ROOT/"decoder-epoch-99-avg-1.int8.onnx"),
    joiner=str(ROOT/"joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
print("ASR 模型已载入")
def make_wav(text, path):
    subprocess.run(["espeak-ng","-w",path,"-s","150","-v","en-us",text],capture_output=True)
    y,sr=sf.read(path)
    if sr!=16000:
        import scipy.signal as ss
        y=ss.resample(y,int(len(y)*16000/sr)); sr=16000
    sf.write(path,y,sr)
    return y,sr
tests=["the capital of france is paris",
       "two plus two is four",
       "a banana is yellow",
       "who wrote romeo and juliet"]
tot_audio=0.0; tot_proc=0.0
for t in tests:
    p=tempfile.mktemp(suffix=".wav")
    y,sr=make_wav(t,p)
    dur=len(y)/sr
    t0=time.time()
    s=rec.create_stream(); s.accept_waveform(sr,y)
    s.accept_waveform(sr,np.zeros(int(sr*0.3),dtype=np.float32))   # 尾部静音
    s.input_finished()
    while rec.is_ready(s): rec.decode_stream(s)
    out=rec.get_result(s)
    dt=time.time()-t0
    tot_audio+=dur; tot_proc+=dt
    print(f"  音频 {dur:.2f}s  处理 {dt:.2f}s  RTF {dt/dur:.2f}  识别：{out}")
    os.remove(p)
print(f"平均 RTF {tot_proc/tot_audio:.2f}（<1 表示快于实时）")
