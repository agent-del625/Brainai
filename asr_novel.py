#!/usr/bin/env python3
"""用完全没见过的句子测 ASR 泛化能力（含中英文）。"""
import json, os, time, tempfile, urllib.request, numpy as np, soundfile as sf, scipy.signal as ss, sherpa_onnx, difflib
from pathlib import Path
ROOT=Path("/srv/brainai/data/asr")
URL="https://api.minimaxi.com/v1/t2a_v2"; KEY=open("/srv/brainai/.mmkey").read().strip()
def tts(text,path,voice="male-qn-qingse"):
    body=json.dumps({"model":"speech-2.8-hd","text":text,"stream":False,"output_format":"hex",
                     "voice_setting":{"voice_id":voice,"speed":1.0}}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=60) as r: d=json.loads(r.read())
    open(path,"wb").write(bytes.fromhex(d["data"]["audio"]))
rec=sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(ROOT/"tokens.txt"), encoder=str(ROOT/"encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(ROOT/"decoder-epoch-99-avg-1.int8.onnx"), joiner=str(ROOT/"joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
# 全部是模型从未见过的句子（自造）
tests=["my grandmother keeps bees in the garden behind her house",
       "the train to edinburgh leaves at half past seven",
       "i forgot my umbrella on the bus again",
       "she painted the kitchen a pale green last summer",
       "three hundred students signed up for the robotics club",
       "今天天气很好，我们一起去公园散步吧",
       "这个周末我要去看一场电影"]
def wer(ref,hyp):
    r=ref.lower().replace("，"," ").replace("。"," ").split()
    h=hyp.lower().split()
    same=sum(b.size for b in difflib.SequenceMatcher(None,r,h).get_matching_blocks())
    return 1-same/max(len(r),1)
errs=[]; tot_d=tot_p=0.0
for t in tests:
    p=tempfile.mktemp(suffix=".mp3")
    try: tts(t,p)
    except Exception as e: print("TTS失败",e); continue
    y,sr=sf.read(p,dtype="float32")
    if y.ndim>1: y=y.mean(1)
    if sr!=16000: y=ss.resample(y,int(len(y)*16000/sr)).astype(np.float32)
    dur=len(y)/16000; t0=time.time()
    s=rec.create_stream(); s.accept_waveform(16000,y.astype(np.float32))
    s.accept_waveform(16000,np.zeros(8000,dtype=np.float32)); s.input_finished()
    while rec.is_ready(s): rec.decode_stream(s)
    out=rec.get_result(s); dt=time.time()-t0
    e=wer(t,out); errs.append(e); tot_d+=dur; tot_p+=dt
    print(f"  WER {e*100:3.0f}%  RTF {dt/dur:.2f}")
    print(f"     原文: {t}")
    print(f"     识别: {out}")
    os.remove(p)
print(f"平均 WER {np.mean(errs)*100:.0f}%  平均 RTF {tot_p/tot_d:.2f}")
