#!/usr/bin/env python3
"""MSR-VTT：抽帧 + 每帧 CLIP 特征（帧即时间轴）。"""
import json, subprocess, numpy as np, cv2, onnxruntime as ort, zipfile, tempfile, os, sys
from pathlib import Path
ROOT=Path("/srv/brainai")
N=300; FR=8
d=json.load(open("/tmp/vtt/test_videodatainfo.json"))
caps={}
for s in d["sentences"]: caps.setdefault(s["video_id"],[]).append(s["caption"])
ids=[v["video_id"] for v in d["videos"]][:N]
sess=ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname=sess.get_inputs()[0].name; oname=sess.get_outputs()[0].name
mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
zf=zipfile.ZipFile(ROOT/"data/downloads/msrvtt_test.zip")
names=set(zf.namelist())
feats=[]; keep=[]; texts=[]
tmp=tempfile.mkdtemp()
for k,vid in enumerate(ids):
    arc=f"TestVideo/{vid}.mp4"
    if arc not in names: continue
    p=os.path.join(tmp,vid+".mp4")
    with open(p,"wb") as f: f.write(zf.read(arc))
    frames=[]
    for i in range(FR):
        t=i*1.0
        r=subprocess.run(["ffmpeg","-v","quiet","-ss",str(t),"-i",p,"-frames:v","1","-f","image2pipe","-vcodec","png","-"],capture_output=True)
        if r.returncode!=0 or not r.stdout: continue
        im=cv2.imdecode(np.frombuffer(r.stdout,np.uint8),cv2.IMREAD_COLOR)
        if im is None: continue
        frames.append(cv2.cvtColor(cv2.resize(im,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.)
    os.remove(p)
    if not frames: continue
    x=np.stack([((f-mean)/std).transpose(2,0,1) for f in frames]).astype(np.float32)
    f=sess.run([oname],{iname:x})[0]
    feats.append(f); keep.append(vid); texts.append(" | ".join(caps.get(vid,[])[:3]))
    if (k+1)%50==0: print(k+1,flush=True)
np.savez_compressed(ROOT/"data/vtt_feats.npz", feats=np.stack(feats), ids=np.array(keep),
                    caps=np.array([caps.get(v,[]) for v in keep],dtype=object))
print("视频",len(feats),"帧/视频",feats[0].shape)
