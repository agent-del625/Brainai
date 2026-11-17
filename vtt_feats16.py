#!/usr/bin/env python3
"""MSR-VTT 全部 2990 个视频，每个抽 16 帧（帧即时间轴）。"""
import json, subprocess, numpy as np, cv2, onnxruntime as ort, zipfile, tempfile, os, time
from pathlib import Path
ROOT=Path("/srv/brainai"); FR=16
d=json.load(open("/tmp/vtt/test_videodatainfo.json"))
caps={}
for s in d["sentences"]: caps.setdefault(s["video_id"],[]).append(s["caption"])
ids=[v["video_id"] for v in d["videos"]]
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
zf=zipfile.ZipFile(ROOT/"data/downloads/msrvtt_test.zip"); names=set(zf.namelist())
feats=[]; keep=[]
tmp=tempfile.mkdtemp(); t0=time.time()
for k,vid in enumerate(ids):
    arc=f"TestVideo/{vid}.mp4"
    if arc not in names: continue
    p=os.path.join(tmp,vid+".mp4")
    with open(p,"wb") as f: f.write(zf.read(arc))
    # 一次性抽 16 帧，避免逐帧起进程
    r=subprocess.run(["ffmpeg","-v","quiet","-i",p,"-vf",f"fps=16/10","-frames:v",str(FR),"-f","image2pipe","-vcodec","png","-"],capture_output=True)
    os.remove(p)
    if r.returncode!=0 or not r.stdout: continue
    frames=[]
    buf=bytes(r.stdout)
    off=0
    while off < len(buf) and len(frames)<FR:
        nxt=buf.find(b'\x89PNG\r\n\x1a\n', off+8)
        chunk=buf[off:] if nxt<0 else buf[off:nxt]
        im=cv2.imdecode(np.frombuffer(chunk,np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            frames.append(cv2.cvtColor(cv2.resize(im,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.)
        if nxt<0: break
        off=nxt
    if not frames: continue
    while len(frames)<FR: frames.append(frames[-1])
    x=np.stack([((f-mean)/std).transpose(2,0,1) for f in frames[:FR]]).astype(np.float32)
    f=sess.run([oname],{iname:x})[0]
    feats.append(f); keep.append(vid)
    if (k+1)%200==0: print(k+1, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed(ROOT/"data/vtt16.npz", feats=np.stack(feats), ids=np.array(keep),
                    caps=np.array([caps.get(v,[]) for v in keep],dtype=object))
print("视频",len(feats),"帧",feats[0].shape)
