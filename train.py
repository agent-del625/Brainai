#!/usr/bin/env python3
"""训练脚本：合成数据上验证架构，支持断点续训、CSV 日志。

用法：
  train.py --steps 300                 # 合成数据跑 300 步
  train.py --steps 20000 --nightly     # 夜间长跑（自动续训）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from collections import Counter

from data_synth import SynthDataset, build_synth_vocab, encode_caption
from model import BrainNet

ROOT = Path("/srv/brainai")
CKPT = ROOT / "ckpt" / "model.pt"
LOG = ROOT / "logs" / "train.csv"
VOCAB = ROOT / "data" / "vocab.json"


class RealDS(torch.utils.data.Dataset):
    """从 data/real/*.npz 读真实片段 + 字幕文本。"""

    def __init__(self, vocab: dict[str, int], length: int = 16):
        frames, mels, caps = [], [], []
        for f in sorted((ROOT / "data" / "real").glob("clips_*.npz")):
            z = np.load(f, allow_pickle=True)
            fr = z["frames"]
            if fr.ndim == 5 and fr.shape[-1] == 3:      # (N,T,H,W,3) → (N,T,3,H,W)
                fr = fr.transpose(0, 1, 4, 2, 3)
            elif fr.ndim == 4:                          # (N,T,H,W) 灰度 → 复制三通道
                fr = np.stack([fr] * 3, axis=2)
            frames.append(fr)
            mels.append(z["mels"] if "mels" in z else np.zeros((z["frames"].shape[0], 1, 40, 64), np.float32))
            caps.extend([str(c) for c in z["captions"]])
        self.frames = np.concatenate(frames) if frames else np.zeros((0, 16, 3, 64, 64), np.uint8)
        self.kind = []
        for f in sorted((ROOT / "data" / "real").glob("clips_*.npz")):
            n = np.load(f, allow_pickle=True)["frames"].shape[0]
            self.kind += ["msrvtt" if "msrvtt" in f.name else "zh"] * n
        self.mels = np.concatenate(mels) if mels else np.zeros((0, 1, 40, 64), np.float32)
        self.caps = caps
        self.vocab, self.length = vocab, length

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        v = torch.from_numpy(np.ascontiguousarray(self.frames[i])).float() / 255.0   # (T,3,H,W)
        a = torch.from_numpy(self.mels[i]).float()
        return v, a, self.caps[i]


class ImgDS(torch.utils.data.Dataset):
    """Flickr8k 图片集：单帧重复 16 次，音频置零，专注图文对齐。"""

    def __init__(self, vocab: dict[str, int], length: int = 16):
        imgs, caps = [], []
        for f in sorted((ROOT / "data" / "image").glob("img_*.npz")):
            z = np.load(f, allow_pickle=True)
            imgs.append(z["images"])
            caps.extend([str(c) for c in z["captions"]])
        self.images = np.concatenate(imgs) if imgs else np.zeros((0, 3, 64, 64), np.uint8)
        self.caps = caps

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = torch.from_numpy(np.ascontiguousarray(self.images[i])).float() / 255.0   # (3,H,W)
        v = img.unsqueeze(0).repeat(16, 1, 1, 1)                                        # (16,3,H,W)
        a = torch.zeros(1, 40, 64)
        return v, a, self.caps[i]


def build_real_vocab(min_count: int = 2, cap: int = 3000) -> dict[str, int]:
    counter: Counter = Counter()
    for pattern in ("data/real/clips_*.npz", "data/image/img_*.npz"):
        for f in sorted(ROOT.glob(pattern)):
            z = np.load(f, allow_pickle=True)
            for c in z["captions"]:
                counter.update(str(c))
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for ch, n in counter.most_common(cap):
        if n >= min_count:
            vocab[ch] = len(vocab)
    return vocab


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    logits = a @ b.t() / temp
    labels = torch.arange(a.size(0), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit-seconds", type=float, default=0, help="超过这个时间就停（夜间用）")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--data", choices=["synth", "real"], default="synth")
    ap.add_argument("--mix", type=float, default=0.0, help="msrvtt 采样占比，0 表示不启用")
    ap.add_argument("--prefix-dropout", type=float, default=0.0,
                    help="按概率把解码器前缀置零，逼模型从画面/声音里取信息")
    ap.add_argument("--stage", choices=["img", "audio", "video", "all"], default="all",
                    help="训练阶段：img=单帧图文, audio=音频文本, video=视频, all=全部")
    args = ap.parse_args()

    torch.set_num_threads(4)
    if args.data == "real":
        vocab = build_real_vocab()
        if args.stage == "img" and any((ROOT / "data" / "image").glob("img_*.npz")):
            ds = ImgDS(vocab)
            print(f"图文数据 {len(ds)} 张，词表 {len(vocab)}")
        else:
            ds = RealDS(vocab)
            print(f"真实数据 {len(ds)} 个片段，词表 {len(vocab)}")
    else:
        vocab = build_synth_vocab()
        ds = SynthDataset(n=args.n)
    VOCAB.parent.mkdir(parents=True, exist_ok=True)
    VOCAB.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    sampler = None
    if args.mix > 0 and getattr(ds, "kind", None):
        from torch.utils.data import WeightedRandomSampler
        w = [args.mix if k == "msrvtt" else (1 - args.mix) for k in ds.kind]
        n_ms = sum(1 for k in ds.kind if k == "msrvtt")
        print(f"混合采样：msrvtt {n_ms} 条 / 中文 {len(ds.kind)-n_ms} 条，目标占比 {args.mix:.0%}")
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                    shuffle=sampler is None, num_workers=0, drop_last=True)

    model = BrainNet(vocab=len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    state = None
    if args.resume and CKPT.exists():
        state = torch.load(CKPT, map_location="cpu", weights_only=False)
        if state.get("vocab") not in (None, len(vocab)):
            print(f"词表变了（{state.get('vocab')} → {len(vocab)}），放弃旧权重，从头训")
            state = None
        if state is None:
            CKPT.unlink(missing_ok=True)
    if state is not None:
        try:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            start_step = state.get("step", 0)
            print(f"续训：从 step {start_step} 开始")
        except RuntimeError as e:
            print(f"旧权重与当前架构不兼容（{str(e).splitlines()[0][:60]}…），从头训练")
            CKPT.unlink(missing_ok=True)
            state = None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params/1e6:.2f}M · 数据 {len(ds)} 条 · batch {args.batch}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_log = not LOG.exists()
    f = LOG.open("a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_log:
        w.writerow(["step", "loss", "frame", "char", "nce", "sec_per_step"])

    model.train()
    t0 = time.time()
    step = start_step
    it = iter(dl)
    while step < start_step + args.steps:
        try:
            video, mels, captions = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        # 目标降采样到 32×32：解码器只需重建低分辨率，Pi 上快 4 倍
        b, t, c, h, wd = video.shape
        video = F.avg_pool2d(video.reshape(b * t, c, h, wd), 2).reshape(b, t, c, h // 2, wd // 2)
        tokens = torch.stack([encode_caption(c, vocab, length=36) for c in captions])
        st = args.stage
        vin = video
        ain = mels
        if st == "img":                       # 只用中间一帧，当图文对训练
            mid = video.size(1) // 2
            vin = video[:, mid:mid + 1].repeat(1, video.size(1), 1, 1, 1)
            ain = torch.zeros_like(mels)
        elif st == "audio":                   # 关掉画面，只练音频↔文本
            vin = torch.zeros_like(video)
        t_dec = tokens
        if args.prefix_dropout > 0 and torch.rand(1).item() < args.prefix_dropout:
            t_dec = torch.zeros_like(tokens)
            t_dec[:, 0] = tokens[:, 0]          # 只留起始符
        out = model(vin.permute(0, 2, 1, 3, 4), ain, tokens, t_dec)

        # 多分支 winner-takes-all：每个样本只用误差最小的那条分支回传
        per_head = torch.stack([F.mse_loss(fh, video, reduction="none").mean(dim=(1, 2, 3, 4))
                                for fh in out["frames"]], dim=1)   # (B, branches)
        wts = torch.softmax(-per_head / 0.05, dim=1).detach()      # 软 WTA：不让任何分支饿死
        l_frame = (wts * per_head).sum(1).mean()
        logits, target = out["chars"]
        n_valid = int((target != 0).sum())
        if n_valid > 0:
            l_char = F.cross_entropy(logits.reshape(-1, len(vocab)),
                                     target.reshape(-1), ignore_index=0)
        else:
            l_char = torch.zeros((), device=logits.device)
        terms, weights = [], []
        if st in ("img", "video", "all"):
            terms.append(l_frame); weights.append(0.3)
        if st in ("audio", "video", "all"):
            terms.append(F.l1_loss(out["mels_out"], mels)); weights.append(0.3)
            terms.append(info_nce(out["za"], out["zt"])); weights.append(0.5)
        if st in ("img", "video", "all"):
            terms.append(info_nce(out["zv"], out["zt"])); weights.append(0.5)
        if st in ("video", "all"):
            terms.append(info_nce(out["zv"], out["za"])); weights.append(0.3)
        terms.append(l_char); weights.append(0.5)
        l_nce = terms[-3] if len(terms) > 2 else terms[0]
        loss = sum(t * w for t, w in zip(terms, weights))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        if step % 10 == 0:
            dt = (time.time() - t0) / 10
            t0 = time.time()
            print(f"step {step:6d} [{args.stage}] loss {loss.item():.4f} "
                  f"(frame {l_frame.item():.4f} char {l_char.item():.3f} nce {l_nce.item():.3f}) "
                  f"{dt:.2f}s/step", flush=True)
            w.writerow([step, f"{loss.item():.4f}", f"{l_frame.item():.4f}",
                        f"{l_char.item():.4f}", f"{l_nce.item():.4f}", f"{dt:.2f}"])
            f.flush()
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "vocab": len(vocab), "vocab_map": vocab}, CKPT)

        if args.limit_seconds and (time.time() - t0) > 0:
            pass
        if args.limit_seconds:
            elapsed = time.time() - t0
            if elapsed > args.limit_seconds:
                print(f"达到时间上限 {args.limit_seconds}s，保存退出（step {step}）")
                break
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
