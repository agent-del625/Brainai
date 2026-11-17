#!/usr/bin/env python3
"""从 ImageNet / Places parquet 里给每个概念存 3 张示例图（供第 3 步"概念渲染"用）。"""
import io
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("/srv/brainai"); OUT = ROOT/"data/gallery"
IMG_LABELS = {"dog": ["Siberian husky","golden retriever","Labrador retriever"],
              "cat": ["tabby, tabby cat","tiger cat","Persian cat"],
              "bird": ["robin, American robin","jay","magpie"],
              "train": ["bullet train, bullet","steam locomotive","electric locomotive"],
              "clock": ["analog clock","digital clock","wall clock"],
              "keyboard": ["computer keyboard, keypad"],
              "plane": ["airliner","warplane, military plane"],
              "bike": ["mountain bike, all-terrain bike"]}
SCENE_LABELS = {"beach": 48, "forest": 150, "ocean": 243, "river": 288, "waterfall": 355}
import json
id2 = json.loads((ROOT/"data/downloads/imagenet_id2label.json").read_text())
name2id = {v:int(k) for k,v in id2.items()}

def save(img_bytes, path):
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((160,160))
        OUT.joinpath(path).parent.mkdir(parents=True, exist_ok=True)
        im.save(OUT/path, quality=88); return True
    except Exception:
        return False

got = {}
for f in sorted(ROOT.glob("data/downloads/imagenet_val*.parquet")):
    if f.stat().st_size < 100*2**20: continue
    t = pq.read_table(f); labs = t["label"].to_pylist()
    for cat, names in IMG_LABELS.items():
        ids = [name2id[n] for n in names if n in name2id]
        for i, l in enumerate(labs):
            if l in ids and got.get(cat, 0) < 3:
                raw = t["image"][i].as_py(); b = raw["bytes"] if isinstance(raw, dict) else raw
                if save(b, f"imagenet/{cat}_{got.get(cat,0)}.jpg"): got[cat] = got.get(cat,0)+1
for f in sorted(ROOT.glob("data/downloads/places_*.parquet")):
    if f.stat().st_size < 100*2**20: continue
    t = pq.read_table(f); labs = t["label"].to_pylist()
    for cat, lab in SCENE_LABELS.items():
        for i, l in enumerate(labs):
            if l == lab and got.get(cat, 0) < 3:
                raw = t["image"][i].as_py(); b = raw["bytes"] if isinstance(raw, dict) else raw
                if save(b, f"scene/{cat}_{got.get(cat,0)}.jpg"): got[cat] = got.get(cat,0)+1
print("已保存:", {k:v for k,v in sorted(got.items())})
