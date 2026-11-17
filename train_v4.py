#!/usr/bin/env python3
"""v4 S1：图文对齐 + 看图说话 + 自评头。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_v4 import BrainV4

ROOT = Path("/srv/brainai")
FEATS = ROOT / "data" / "image" / "clip_feats.npz"
CKPT = ROOT / "ckpt" / "v4.pt"
LOG = ROOT / "logs" / "v4.csv"
L = 24


def encode(text: str, vocab: dict[str, int], length: int = L) -> torch.Tensor:
    ids = [vocab.get(ch, 1) for ch in text][: length - 1]
    ids = [2] + ids
    ids += [0] * (length - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def info_nce(a, b, temp=0.07):
    logits = a @ b.t() / temp
    y = torch.arange(a.size(0))
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.t(), y)) / 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--prefix-dropout", type=float, default=0.5)
    ap.add_argument("--cont", action="store_true", help="用连续解码器（非自回归）")
    args = ap.parse_args()

    torch.set_num_threads(4)
    z = np.load(FEATS, allow_pickle=True)
    feats = torch.from_numpy(z["feats"]).float()
    caps = [str(c) for c in z["captions"]]

    counter: dict[str, int] = {}
    for c in caps:
        for ch in c:
            counter[ch] = counter.get(ch, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for ch, n in sorted(counter.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            vocab[ch] = len(vocab)
    print(f"图片 {len(feats)} 张，词表 {len(vocab)}")

    model = BrainV4(vocab=len(vocab), rounds=args.rounds, cont_decoder=args.cont)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    if CKPT.exists() and not args.cont:
        st = torch.load(CKPT, map_location="cpu", weights_only=False)
        if st.get("vocab") == len(vocab):
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
            print(f"续训 step {step}")

    print("参数", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), "M")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    f = LOG.open("a")
    t0 = time.time()
    model.train()
    while step < args.steps:
        idx = torch.randint(0, len(feats), (args.batch,))
        clip = feats[idx]
        tok = torch.stack([encode(caps[i], vocab) for i in idx])
        t_dec = tok
        if torch.rand(1).item() < args.prefix_dropout:
            t_dec = torch.zeros_like(tok); t_dec[:, 0] = 2

        # scheduled sampling（仅自回归解码器需要）：训练后期逐步改用模型自己的预测当前缀
        ss_p = 0.0 if getattr(model, "use_cont", False) else (min(0.5, step / 6000 * 0.5) if step > 1000 else 0.0)
        if ss_p > 0:
            with torch.no_grad():
                pre = model(clip=clip, t_in=tok, t_dec=t_dec)
                lg0, _ = pre["chars"]
                pred = lg0.argmax(1)                       # (B,V,L-1)
                pred_seq = torch.cat([torch.full((tok.size(0), 1), 2, dtype=torch.long), pred], 1)
                use = (torch.rand(tok.shape) < ss_p) & (torch.arange(tok.size(1)).unsqueeze(0) > 0)
                t_dec = torch.where(use, pred_seq.clamp(0, len(vocab) - 1), t_dec)
        out = model(clip=clip, t_in=tok, t_dec=t_dec)
        l_nce = info_nce(F.normalize(out["zc"], dim=-1), F.normalize(out["zt"], dim=-1))
        if isinstance(out["chars"], torch.Tensor) and out["chars"].dim() == 0:
            l_char = out["chars"]
        else:
            lg, tg = out["chars"]
            n_valid = int((tg != 0).sum())
            l_char = (F.cross_entropy(lg.reshape(-1, len(vocab)), tg.reshape(-1), ignore_index=0)
                      if n_valid > 0 else torch.zeros((), device=lg.device))
        # 自评头：预测每个样本的字符误差
        with torch.no_grad():
            if isinstance(out["chars"], torch.Tensor) and out["chars"].dim() == 0:
                per = torch.full((tok.size(0),), l_char.item())
            else:
                per = F.cross_entropy(lg.reshape(-1, len(vocab)), tg.reshape(-1),
                                      ignore_index=0, reduction="none").view(lg.size(0), -1).mean(1)
        l_err = F.mse_loss(out["err_pred"], per)
        l_cycle = model.text_head.cycle_loss(out["z_dec"], model.text) if args.cont else torch.zeros(())
        loss = l_nce * 1.0 + l_char * 1.0 + l_err * 0.1 + l_cycle * 0.5

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); step += 1

        if step % 10 == 0:
            dt = (time.time() - t0) / 10; t0 = time.time()
            print(f"step {step:5d} loss {loss.item():.3f} nce {l_nce.item():.3f} "
                  f"char {l_char.item():.3f} cyc {float(l_cycle):.3f} {dt:.2f}s/step", flush=True)
            f.write(f"{step},{loss.item():.4f},{l_nce.item():.4f},{l_char.item():.4f}\n"); f.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "vocab": len(vocab)}, CKPT)
    f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
