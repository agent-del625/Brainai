#!/usr/bin/env python3
"""从媒体库抽取"视频片段 + 字幕文本"训练样本。

做法：抽字幕到 SRT → 用字幕时间点划 8 秒窗口 → ffmpeg 稀疏取 16 帧（2fps，64×64 灰度）
      → 窗口内字幕拼成描述 → 存 npz 分片。
用法: data_video.py --files 3 --clips 15
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from audio_feat import mel_spec

ROOT = Path("/srv/brainai")
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "real"
MEDIA = Path("/home/del/media")
T, H = 16, 64
CLIP_SEC = 8.0


def run(cmd: list[str], binary: bool = False):
    return subprocess.run(cmd, capture_output=True, check=False).stdout if binary else \
        subprocess.run(cmd, capture_output=True, text=True, check=False)


def extract_srt(video: Path) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    srt = CACHE / (video.stem + ".srt")
    if srt.exists() and srt.stat().st_size > 0:
        return srt
    run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:s:0", str(srt)])
    return srt


def parse_srt(srt: Path) -> list[tuple[float, float, str]]:
    if not srt.exists():
        return []
    text = srt.read_text(encoding="utf-8", errors="replace")
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[1] if lines[0].isdigit() else lines[0])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = " ".join(lines[2:] if lines[0].isdigit() else lines[1:])
        body = re.sub(r"<[^>]+>|\{[^}]+\}", "", body).strip()
        if body:
            out.append((start, end, body))
    return out


def grab_clip(video: Path, start: float) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{CLIP_SEC}", "-i", str(video),
           "-vf", f"fps=2,scale={H}:{H},format=rgb24", "-frames:v", str(T),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = run(cmd, binary=True)
    if not raw or len(raw) < T * H * H * 3:
        return None
    arr = np.frombuffer(raw[: T * H * H * 3], dtype=np.uint8).reshape(T, H, H, 3)
    return arr


def grab_audio(video: Path, start: float) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{CLIP_SEC}", "-i", str(video),
           "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = run(cmd, binary=True)
    if not raw or len(raw) < 16000:
        return None
    return mel_spec(np.frombuffer(raw, dtype=np.int16))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=3)
    ap.add_argument("--clips", type=int, default=15, help="每个文件抽多少片段")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cands = [p for p in MEDIA.rglob("*.mkv") if p.stat().st_size > 100 * 2**20]
    # 1080p 优先（解码快），4K 往后排
    videos = sorted(cands, key=lambda p: (0 if "1080p" in p.name else 1, p.stat().st_size))
    videos = videos[: args.files]
    print(f"待处理 {len(videos)} 个文件")

    all_frames, all_caps, all_mels = [], [], []
    for vi, video in enumerate(videos, 1):
        srt = extract_srt(video)
        subs = parse_srt(srt)
        if not subs:
            print(f"[{vi}] {video.name}: 无字幕，跳过")
            continue
        picks = subs[:: max(1, len(subs) // args.clips)][: args.clips]
        got = 0
        for start, _end, _txt in picks:
            clip = grab_clip(video, start)
            if clip is None:
                continue
            mel = grab_audio(video, start)
            if mel is None:
                mel = np.zeros((1, 40, 64), dtype=np.float32)
            cap = "".join(
                t for s, e, t in subs if s < start + CLIP_SEC and e > start
            )[:16]
            if len(cap) < 2:
                continue
            all_frames.append(clip)
            all_mels.append(mel)
            all_caps.append(cap)
            got += 1
        print(f"[{vi}/{len(videos)}] {video.name[:44]}: {got} 个片段", flush=True)

    if not all_frames:
        print("没抓到样本")
        return 1
    frames = np.stack(all_frames)
    mels = np.stack(all_mels)
    np.savez_compressed(OUT / f"clips_{int(__import__('time').time())}.npz",
                        frames=frames, mels=mels, captions=np.array(all_caps))
    print(f"保存 {frames.shape[0]} 个片段 → {OUT}")
    print("样例描述:", all_caps[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
