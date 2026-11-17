#!/usr/bin/env python3
"""评测"输出是否正常"：重建、跨模态检索、生成文字、可视化对照。

产出：
  logs/eval_<时间>.json   指标
  logs/eval_<时间>.png    可视化（原图 / 两条解码分支）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import wave

from audio_feat import mel_to_wav
from model import BrainNet
from train import RealDS, build_real_vocab, encode_caption

ROOT = Path("/srv/brainai")


def retrieval(z_a: torch.Tensor, z_b: torch.Tensor) -> tuple[float, float]:
    sim = z_a @ z_b.t()
    n = sim.size(0)
    ranks = sim.argsort(dim=1, descending=True)
    target = torch.arange(n).unsqueeze(1)
    r1 = (ranks[:, :1] == target).any(1).float().mean().item()
    r5 = (ranks[:, :5] == target).any(1).float().mean().item()
    return r1, r5


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def to_u8(img: torch.Tensor, scale: int = 4) -> np.ndarray:
    x = img.detach().cpu().numpy()
    if x.ndim == 3:
        x = x.transpose(1, 2, 0)
    x = np.clip(x, 0, 1)
    if x.ndim == 2:
        x = np.stack([x] * 3, -1)
    x = (x * 255).astype(np.uint8)
    return cv2.resize(x, (x.shape[1] * scale, x.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def id2ch(vocab: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in vocab.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ckpt", default=str(ROOT / "ckpt" / "model.pt"))
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    vocab = state.get("vocab_map") or build_real_vocab()
    branches = len([k for k in state["model"] if k.startswith("frame_heads.") and k.endswith(".fc.weight")]) or 2
    model = BrainNet(vocab=len(vocab), branches=branches)
    model.load_state_dict(state["model"])
    model.eval()

    ds = RealDS(vocab)
    if len(ds) == 0:
        print("没有真实数据，先用 data_video.py 抽片段")
        return 1

    vids, mels, caps, toks = [], [], [], []
    for i in range(min(args.n, len(ds))):
        v, a, c = ds[i]
        vids.append(v)
        mels.append(a)
        caps.append(c)
        toks.append(encode_caption(c, vocab, length=36))
    video = torch.stack(vids)
    mel = torch.stack(mels)
    tokens = torch.stack(toks)

    b, t, c, h, w = video.shape
    small = F.avg_pool2d(video.reshape(b * t, c, h, w), 2).reshape(b, t, c, h // 2, w // 2)

    with torch.no_grad():
        out = model(video.permute(0, 2, 1, 3, 4), mel, tokens)
        zj = model.fuse(out["zv"], out["za"], out["zt"])
        gen_ids = model.text_head.generate(zj, start_id=2, length=tokens.size(1))

    metrics: dict = {}
    per_branch = [F.mse_loss(fh, small).item() for fh in out["frames"]]
    metrics["重建MSE"] = [round(x, 5) for x in per_branch]
    metrics["重建PSNR"] = [round(psnr(fh, small), 2) for fh in out["frames"]]
    metrics["最优分支"] = int(np.argmin(per_branch))

    for name, (a_, b_) in {"视频→文本": (out["zv"], out["zt"]),
                           "音频→文本": (out["za"], out["zt"]),
                           "视频→音频": (out["zv"], out["za"])}.items():
        r1, r5 = retrieval(a_, b_)
        metrics[f"{name}_R@1"] = round(r1 * 100, 1)
        metrics[f"{name}_R@5"] = round(r5 * 100, 1)

    ref, hyp = tokens.numpy(), gen_ids.numpy()
    mask = ref != 0
    # 自回归生成是错位预测：把预测序列和"参考后一位"比较
    ref_shift = ref[:, 1:]
    hyp_shift = hyp[:, :-1]
    m2 = ref_shift != 0
    metrics["生成字符准确率"] = round(float(((hyp_shift == ref_shift) & m2).sum() / max(m2.sum(), 1)) * 100, 1)
    metrics["参数量M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (ROOT / "logs" / f"eval_{stamp}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    k = min(4, b)
    rows = [np.hstack([to_u8(small[i, t // 2]) for i in range(k)])]
    for fh in out["frames"]:
        rows.append(np.hstack([to_u8(fh[i, t // 2]) for i in range(k)]))
    canvas = np.vstack(rows)
    cv2.putText(canvas, "row1 GT | row2 branch0 | row3 branch1",
                (6, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out_png = ROOT / "logs" / f"eval_{stamp}.png"
    cv2.imwrite(str(out_png), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    m = id2ch(vocab)
    gen_text = "".join(m.get(int(i), "") for i in gen_ids[0].tolist()).replace("<eos>", "")[:20]

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("模型生成:", gen_text or "(空)")
    print("参考字幕:", caps[0])
    # 音频输出：生成梅尔谱 → Griffin-Lim 合成波形
    try:
        gen_mel = out["mels_out"][0, 0].detach().cpu().numpy()
        wav = mel_to_wav(gen_mel)
        wav_path = ROOT / "logs" / f"gen_{stamp}.wav"
        with wave.open(str(wav_path), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(wav.tobytes())
        ref_mel = mel[0, 0].detach().cpu().numpy()
        ref_path = ROOT / "logs" / f"ref_{stamp}.wav"
        with wave.open(str(ref_path), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(mel_to_wav(ref_mel).tobytes())
        metrics["音频输出"] = wav_path.name
        print("生成音频:", wav_path)
        print("参考音频:", ref_path)
    except Exception as e:
        print("音频合成失败:", str(e)[:80])

    print("图:", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
