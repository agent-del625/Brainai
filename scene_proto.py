#!/usr/bin/env python3
"""Places365 场景原型：beach(48) / forest(150)。"""
import io
from pathlib import Path
import numpy as np, onnxruntime as ort, pyarrow.parquet as pq
from PIL import Image

ROOT = Path("/srv/brainai")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
TARGETS = {"beach": 48, "forest": 150, "ocean": 243, "river": 288, "waterfall": 355}

def prep(b):
    im = Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)

sess = ort.InferenceSession(str(ROOT / "data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

out = {}
for f in sorted(ROOT.glob("data/downloads/places_*.parquet")):
    if f.stat().st_size < 100 * 2**20:
        continue
    t = pq.read_table(f)
    labs = t["label"].to_pylist()
    for name, lab in TARGETS.items():
        if lab not in labs or name in out:
            continue
        vecs = []
        for i, l in enumerate(labs):
            if l == lab and len(vecs) < 8:
                raw = t["image"][i].as_py()
                b = raw["bytes"] if isinstance(raw, dict) else raw
                try:
                    vecs.append(sess.run([oname], {iname: prep(b)[None]})[0][0])
                except Exception:
                    continue
        if len(vecs) >= 3:
            out[name] = np.mean(vecs, 0)
            print(f"{name}: {len(vecs)} 张")

np.savez_compressed(ROOT / "data/scene_proto.npz",
                    cats=np.array(list(out)), feats=np.stack(list(out.values())).astype(np.float32))
print("保存", list(out))
