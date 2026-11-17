#!/usr/bin/env python3
"""sherpa-onnx VITS 中文 TTS：延迟与音质测试。"""
import time, numpy as np, soundfile as sf, sherpa_onnx
from pathlib import Path
ROOT=Path("/srv/brainai/data/vits")
cfg=sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(ROOT/"model.onnx"), tokens=str(ROOT/"tokens.txt"),
            lexicon=str(ROOT/"lexicon.txt"),
            dict_dir=str(ROOT/"dict") if (ROOT/"dict").exists() else ""),
        num_threads=4),
    max_num_sentences=1)
tts=sherpa_onnx.OfflineTts(cfg)
print("VITS 已载入, 采样率", tts.sample_rate)
texts=["法国的首都是巴黎。","二加二等于四。","香蕉是黄色的。","今天天气很好，我们一起去公园散步吧。"]
out=Path("/srv/brainai/logs/vits"); out.mkdir(parents=True,exist_ok=True)
for i,t in enumerate(texts):
    t0=time.time()
    audio=tts.generate(t, sid=0, speed=1.0)
    dt=time.time()-t0
    y=np.array(audio.samples,dtype=np.float32)
    dur=len(y)/audio.sample_rate
    sf.write(str(out/f"vits_{i}.wav"), y, audio.sample_rate)
    print(f"  {t[:20]:22s} 延迟 {dt*1000:6.0f} ms  音频 {dur:.2f}s  RTF {dt/max(dur,1e-6):.2f}")
