#!/usr/bin/env python3
"""把 Flickr8k 的 parquet 转成图片-描述训练集。

parquet 里每行是一张图 + 5 句描述；这里取第一句，图缩到 64×64 RGB。
用法: data_img.py --parquet data/downloads/flickr8k_val.parquet --limit 2000
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("/srv/brainai")
OUT = ROOT / "data" / "image"
H = 64
CAPTION_LEN = 36


def pick_columns(schema) -> tuple[str, str]:
    names = [f.name for f in schema]
    img = next((n for n in names if n in ("image", "img", "jpg")), names[0])
    cap = next((n for n in names if "caption" in n.lower() or n in ("text", "sentences")), names[1])
    return img, cap


def first_caption(value) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (list, tuple)) and v:
                return str(v[0])
    return str(value)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "data/downloads/flickr8k_val.parquet"))
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    table = pq.read_table(args.parquet)
    img_col, cap_col = pick_columns(table.schema)
    print(f"列：{img_col} / {cap_col}，行数 {table.num_rows}")

    images, captions = [], []
    for i in range(min(args.limit, table.num_rows)):
        row = {k: table[k][i].as_py() for k in (img_col, cap_col)}
        raw = row[img_col]
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB").resize((H, H), Image.BILINEAR)
        except Exception:
            continue
        images.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
        captions.append(first_caption(row[cap_col]).strip()[:CAPTION_LEN])
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{args.limit} 已处理", flush=True)

    if not images:
        print("没有解析出图片")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"img_{int(__import__('time').time())}.npz"
    np.savez_compressed(path, images=np.stack(images), captions=np.array(captions))
    print(f"保存 {len(images)} 张 → {path}")
    print("样例:", captions[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
