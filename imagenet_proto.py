#!/usr/bin/env python3
"""从 ImageNet 验证集抽取几个概念类别的图片，算 CLIP 原型。"""
import io, json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("/srv/brainai")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# ESC-50 类别 → ImageNet 标签名（精确匹配）
MAP = {
    "dog": ["Siberian husky", "golden retriever", "Labrador retriever", "German shepherd"],
    "cat": ["tabby, tabby cat", "tiger cat", "Persian cat", "Siamese cat"],
    "bird": ["robin, American robin", "jay", "magpie"],
    "car": ["sports car, sport car", "limousine, limo", "minivan"],
    "train": ["bullet train, bullet", "steam locomotive", "electric locomotive"],
    "clock": ["analog clock", "digital clock", "wall clock"],
    "keyboard": ["computer keyboard, keypad"],
    "plane": ["airliner", "warplane, military plane"],
    "bike": ["mountain bike, all-terrain bike", "bicycle-built-for-two, tandem bicycle, tandem"],
    "truck": ["fire engine, fire truck", "garbage truck"],
}

def prep(img_bytes):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)

def main() -> int:
    id2label = json.loads((ROOT / "data/downloads/imagenet_id2label.json").read_text())
    name2id = {v: int(k) for k, v in id2label.items()}
    want = {}
    for concept, names in MAP.items():
        ids = [name2id[n] for n in names if n in name2id]
        if ids:
            want[concept] = ids
    print({k: len(v) for k, v in want.items()})

    sess = ort.InferenceSession(str(ROOT / "data/downloads/clip_vision.onnx"),
                                providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    feats = {c: [] for c in want}
    shards = sorted(ROOT.glob("data/downloads/imagenet_val*.parquet"))
    for shard in shards:
        if shard.stat().st_size < 100 * 2**20:
            continue
        t = pq.read_table(shard)
        labels = t["label"].to_pylist()
        for i, lab in enumerate(labels):
            for c, ids in want.items():
                if lab in ids and len(feats[c]) < 8:
                    raw = t["image"][i].as_py()
                    b = raw["bytes"] if isinstance(raw, dict) else raw
                    try:
                        x = prep(b)[None]
                    except Exception:
                        continue
                    feats[c].append(sess.run([oname], {iname: x})[0][0])


    print({k: len(v) for k, v in feats.items()})
    cats, protos = [], []
    for c, v in feats.items():
        if len(v) >= 2:
            cats.append(c); protos.append(np.mean(v, 0))
    np.savez_compressed(ROOT / "data/imagenet_proto.npz", cats=np.array(cats),
                        feats=np.stack(protos).astype(np.float32))
    print(f"保存 {len(cats)} 个概念: {cats}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
