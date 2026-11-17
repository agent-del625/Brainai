#!/usr/bin/env python3
"""自动扩充概念：ESC-50 的 50 个声音类别 → 在已有 ImageNet/Places 分片里找对应图像。"""
import io, json, re
from pathlib import Path
import numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image

ROOT = Path("/srv/brainai")
MEAN = np.array([0.48145466,0.4578275,0.40821073], np.float32)
STD = np.array([0.26862954,0.26130258,0.27577711], np.float32)
id2 = json.loads((ROOT/"data/downloads/imagenet_id2label.json").read_text())
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
cats = [str(c) for c in ep["cats"]]

# ESC-50 类别 → ImageNet 名字关键词
KW = {"dog":["Siberian husky","golden retriever","Labrador retriever"],
      "cat":["tabby","tiger cat","Persian cat"],"chirping_birds":["robin","jay","magpie"],
      "rooster":["cock"],"hen":["hen"],"pig":["hog","pig"],"cow":["ox","cow"],
      "sheep":["sheep"],"frog":["bullfrog","tree frog"],"crow":["crow"],
      "insects":["grasshopper","cricket","bee"],"train":["bullet train","locomotive"],
      "airplane":["airliner","warplane"],"helicopter":["helicopter"],
      "car_horn":["sports car","limousine"],"engine":["sports car","jeep"],
      "clock_alarm":["analog clock","digital clock"],"clock_tick":["wall clock"],
      "keyboard_typing":["computer keyboard","typewriter"],"mouse_click":["computer mouse","mouse"],
      "door_wood_knock":["sliding door","door"],"church_bells":["bell cote","church"],
      "sea_waves":["seashore","lakeside"],"crickets":["grasshopper"],
      "chainsaw":["chain saw"],"siren":["police van"],"fireworks":["firework"]}

name2id = {v:k for k,v in id2.items()}
want = {}
for c in cats:
    ids = []
    for kw in KW.get(c, [c]):
        for name, i in name2id.items():
            if kw.lower() in name.lower():
                ids.append(int(i))
    if ids: want[c] = set(ids)
print("可匹配类别", len(want), sorted(want)[:15])

sess = ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
def prep(b):
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((224,224), Image.BILINEAR)
    x = np.asarray(im, np.float32)/255.0
    return ((x-MEAN)/STD).transpose(2,0,1)[None]

found = {c: [] for c in want}
shards = [f for f in sorted(ROOT.glob("data/downloads/imagenet_val*.parquet")) if f.stat().st_size > 100*2**20]
for f in shards:
    t = pq.read_table(f)
    labs = t["label"].to_pylist()
    for c, ids in want.items():
        if len(found[c]) >= 4: continue
        for i, l in enumerate(labs):
            if l in ids and len(found[c]) < 4:
                raw = t["image"][i].as_py(); b = raw["bytes"] if isinstance(raw, dict) else raw
                try: found[c].append(sess.run([oname], {iname: prep(b)})[0][0])
                except Exception: pass

names, protos = [], []
for c, v in found.items():
    if len(v) >= 2:
        names.append(c); protos.append(np.mean(v, 0))
print(f"得到 {len(names)} 个概念: {names}")
np.savez_compressed(ROOT/"data/concept_protos.npz", cats=np.array(names),
                    feats=np.stack(protos).astype(np.float32), sound=np.array(names))
