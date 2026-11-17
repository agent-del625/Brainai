#!/usr/bin/env python3
"""把 MSR-VTT 的 mp4 + json 字幕转成我们的 npz 片段格式。

用法: data_msrvtt.py --limit 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from audio_feat import mel_spec

ROOT = Path("/srv/brainai")
VIDEOS = Path("/home/del/media/Documentaries/msrvtt/TestVideo")
CAPS = Path("/srv/brainai/data/downloads/test_videodatainfo.json")
T, H, CLIP_SEC = 16, 64, 8.0
CAPTION_LEN = 36


def run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, capture_output=True, check=False).stdout


def grab(video: Path, start: float):
    frames = run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{CLIP_SEC}", "-i", str(video),
                  "-vf", f"fps=2,scale={H}:{H},format=rgb24", "-frames:v", str(T),
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    audio = run(["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{CLIP_SEC}", "-i", str(video),
                 "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
    if len(frames) < T * H * H * 3:
        return None, None
    v = np.frombuffer(frames[: T * H * H * 3], np.uint8).reshape(T, H, H, 3).transpose(0, 3, 1, 2)
    a = mel_spec(np.frombuffer(audio, np.int16)) if len(audio) > 16000 else np.zeros((1, 40, 64), np.float32)
    return v, a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    data = json.loads(CAPS.read_text(encoding="utf-8"))
    caps: dict[str, list[str]] = defaultdict(list)
    for s in data["sentences"]:
        caps[s["video_id"]].append(s["caption"])

    files = sorted(VIDEOS.glob("*.mp4"))[: args.limit]
    frames, mels, texts = [], [], []
    for i, f in enumerate(files, 1):
        vid = f.stem
        if vid not in caps:
            continue
        caption = caps[vid][0].strip()[:CAPTION_LEN]
        v, a = grab(f, 0.0)
        if v is None:
            continue
        frames.append(v)
        mels.append(a)
        texts.append(caption)
        if i % 50 == 0:
            print(f"{i}/{len(files)} 完成，样本 {len(frames)}", flush=True)

    if not frames:
        print("没有成功样本")
        return 1
    out = ROOT / "data" / "real" / f"clips_msrvtt_{int(__import__('time').time())}.npz"
    np.savez_compressed(out, frames=np.stack(frames), mels=np.stack(mels), captions=np.array(texts))
    print(f"保存 {len(frames)} 片段 → {out}")
    print("样例:", texts[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
