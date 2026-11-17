#!/usr/bin/env python3
"""用 CLIP ViT-B/32 ONNX 预计算图片特征（离线一次，训练时直接用）。"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path("/srv/brainai")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess(imgs: np.ndarray) -> np.ndarray:
    """(N,3,64,64) uint8 → (N,3,224,224) float32 归一化。"""
    out = []
    for im in imgs:
        x = im.transpose(1, 2, 0).astype(np.float32) / 255.0
        # 最近邻放大到 224（原图只有 64×64，细节无法恢复，但足够 CLIP 提取语义）
        idx = (np.arange(224) * x.shape[0] / 224).astype(int)
        x = x[idx][:, idx]
        x = (x - MEAN) / STD
        out.append(x.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=str(ROOT / "data/downloads/clip_vision.onnx"))
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    print("输入", iname, sess.get_inputs()[0].shape, "→ 输出", oname, sess.get_outputs()[0].shape)

    imgs, caps = [], []
    for f in sorted((ROOT / "data" / "image").glob("img_*.npz")):
        z = np.load(f, allow_pickle=True)
        imgs.append(z["images"])
        caps.extend([str(c) for c in z["captions"]])
    images = np.concatenate(imgs)
    if args.limit:
        images, caps = images[: args.limit], caps[: args.limit]
    print("图片", images.shape, "描述", len(caps))

    feats = []
    for i in range(0, len(images), 16):
        batch = preprocess(images[i:i + 16])
        out = sess.run([oname], {iname: batch})[0]
        feats.append(out)
        if (i // 16) % 40 == 0:
            print(f"{i}/{len(images)}", flush=True)
    feats = np.concatenate(feats).astype(np.float32)
    print("特征", feats.shape)
    out_path = ROOT / "data" / "image" / "clip_feats.npz"
    np.savez_compressed(out_path, feats=feats, captions=np.array(caps))
    print("保存", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
