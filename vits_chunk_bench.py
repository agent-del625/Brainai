#!/usr/bin/env python3
"""VITS 分块流式：测"首块合成延迟"随文本长度的变化。"""
import time, numpy as np, soundfile as sf, sherpa_onnx
from pathlib import Path
V=Path("/srv/brainai/data/vits")
cfg=sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(V/"model.onnx"), tokens=str(V/"tokens.txt"), lexicon=str(V/"lexicon.txt"),
            dict_dir=str(V/"dict")), num_threads=4), max_num_sentences=1)
tts=sherpa_onnx.OfflineTts(cfg)
cases=["巴黎。","法国的首都是巴黎。","法国的首都是巴黎。它也是法国最大的城市。",
       "法国的首都是巴黎，它位于欧洲西部，是法国最大的城市和经济文化中心。"]
for t in cases:
    t0=time.time(); a=tts.generate(t,sid=0,speed=1.0); dt=time.time()-t0
    dur=len(a.samples)/a.sample_rate
    print(f"  {len(t):2d}字  延迟 {dt*1000:5.0f} ms  音频 {dur:.2f}s  RTF {dt/max(dur,1e-6):.2f}   «{t[:16]}»")
