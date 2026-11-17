"""合成数据：移动的几何体 + 自动生成的描述，用于验证架构能否学动。

每个样本：16 帧 64×64 灰度视频 + 一句中文描述（颜色/形状/方向）。
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from audio_feat import mel_spec, synthetic_audio

COLORS = {"红": (1.0, 0.15, 0.15), "绿": (0.2, 0.9, 0.3), "蓝": (0.2, 0.4, 1.0), "白": (0.95, 0.95, 0.95)}
SHAPES = ["方块", "圆", "三角"]
DIRS = {"向右": (1, 0), "向左": (-1, 0), "向上": (0, -1), "向下": (0, 1)}
T, H = 16, 64


def draw_frame(cx: float, cy: float, shape: str, value, size: int = 9) -> torch.Tensor:
    frame = torch.zeros(3, H, H)
    color = torch.tensor(value).view(3, 1, 1) if not torch.is_tensor(value) else value
    x0, y0 = int(cx), int(cy)
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            x, y = x0 + dx, y0 + dy
            if not (0 <= x < H and 0 <= y < H):
                continue
            inside = {
                "方块": abs(dx) <= size and abs(dy) <= size,
                "圆": dx * dx + dy * dy <= size * size,
                "三角": dy >= -size and abs(dx) <= (dy + size) // 2,
            }[shape]
            if inside:
                frame[:, y, x] = color.flatten()
    return frame


class SynthDataset(Dataset):
    def __init__(self, n: int = 4096, seed: int = 0):
        self.n = n
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        color = rng.choice(list(COLORS))
        shape = rng.choice(SHAPES)
        direction = rng.choice(list(DIRS))
        speed = rng.choice([0.6, 1.0, 1.6])
        value = COLORS[color]

        dx, dy = DIRS[direction]
        start = rng.uniform(14, 50)
        cx, cy = (start, 32.0) if dx else (32.0, start)
        frames = []
        for step in range(T):
            cx += dx * speed
            cy += dy * speed
            cx = min(max(cx, 8), H - 8)
            cy = min(max(cy, 8), H - 8)
            frames.append(draw_frame(cx, cy, shape, value))
        video = torch.stack(frames)                    # (T,1,64,64)
        caption = f"{color}{shape}{direction}移动"
        pcm = synthetic_audio(list(DIRS).index(direction), speed, sum(value))
        mel = torch.from_numpy(mel_spec(pcm))
        return video, mel, caption


def encode_caption(text: str, vocab: dict[str, int], length: int = 16) -> torch.Tensor:
    ids = [vocab.get(ch, vocab.get("<unk>", 1)) for ch in text][: length - 1]
    ids.append(vocab["<eos>"])
    ids += [0] * (length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def build_synth_vocab() -> dict[str, int]:
    chars = set("".join(COLORS) + "".join(SHAPES) + "".join(DIRS) + "移动")
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for i, ch in enumerate(sorted(chars), start=3):
        vocab[ch] = i
    return vocab
