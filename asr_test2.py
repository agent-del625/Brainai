#!/usr/bin/env python3
"""用自然的 TTS 语音（MiniMax）测本地 ASR 速度和准确率。"""
import json, os, time, tempfile, urllib.request, numpy as np, soundfile as sf, scipy.signal as ss, sherpa_onnx
from pathlib import Path
ROOT=Path("/srv/brainai/data/asr")
URL="https://api.minimaxi.com/v1/t2a_v2"
KEY=os.environ.get("MINIMAX_CN_API_KEY") or open("/srv/brainai/.mmkey").read().strip()
def tts(text,path):
    body=json.dumps({"model":"speech-2.8-hd","text":text,"stream":False,"output_format":"hex",
                     "voice_setting":{"voice_id":"male-qn-qingse","speed":1.0}}).encode()
    req=urllib.request.Request(URL,data=body,headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=60) as r: d=json.loads(r.read())
    audio=bytes.fromhex(d["data"]["audio"])
    open(path,"wb").write(audio)
    return path
rec=sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=str(ROOT/"tokens.txt"), encoder=str(ROOT/"encoder-epoch-99-avg-1.int8.onnx"),
    decoder=str(ROOT/"decoder-epoch-99-avg-1.int8.onnx"), joiner=str(ROOT/"joiner-epoch-99-avg-1.int8.onnx"),
    num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
tests=["the capital of france is paris","two plus two is four","a banana is yellow",
       "who wrote romeo and juliet","please turn on the light"]
def wer(ref,hyp):
    r=ref.lower().split(); h=hyp.lower().split()
    import difflib
    sm=difflib.SequenceMatcher(None,r,h)
    same=sum(b.size for b in sm.get_matching_blocks())
    return 1-same/max(len(r),1)
tot_d=tot_p=0.0; errs=[]
for t in tests:
    p=tempfile.mktemp(suffix=".mp3")
    try: tts(t,p)
    except Exception as e:
        print("TTS 失败:",e); continue
    y,sr=sf.read(p,dtype="float32")
    if y.ndim>1: y=y.mean(1)
    if sr!=16000: y=ss.resample(y,int(len(y)*16000/sr)).astype(np.float32)
    dur=len(y)/16000
    t0=time.time()
    s=rec.create_stream(); s.accept_waveform(16000,y.astype(np.float32))
    s.accept_waveform(16000,np.zeros(8000,dtype=np.float32)); s.input_finished()
    while rec.is_ready(s): rec.decode_stream(s)
    out=rec.get_result(s); dt=time.time()-t0
    tot_d+=dur; tot_p+=dt; e=wer(t,out); errs.append(e)
    print(f"  {dur:.2f}s音频 {dt:.2f}s RTF {dt/dur:.2f} WER {e*100:.0f}%  → {out}")
    os.remove(p)
print(f"平均 RTF {tot_p/max(tot_d,1e-6):.2f}   平均 WER {np.mean(errs)*100:.0f}%")
