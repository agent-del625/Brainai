#!/usr/bin/env python3
"""Brainai 统一自动评测入口。

按 docs/plan.md 第 1 项（问题一：缺评测自动化）落地：
一条命令跑出全部模态指标 → 写入 logs/baseline.json（最新）+ logs/baseline_<日期>.json（归档）。

每个评测项都是一个「探针」：资产（数据/权重）缺失时记 skip 和原因，不中断其它探针；
跑完后 baseline.json 里能看清每一项是 ok / skip / error。

用法：
  python3 eval_all.py                          # 默认根目录 /srv/brainai（或环境变量 BRAINAI_ROOT）
  python3 eval_all.py --root /srv/brainai --ckpt ckpt/model.pt --seed 0 --n 32
  python3 eval_all.py --only unified,caption   # 只跑指定探针
  python3 eval_all.py --selftest               # 无资产自检：验证指标函数与 JSON 落盘

指标口径与 Issue #1 基线对齐，见 docs/baseline.md。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_ROOT = os.environ.get("BRAINAI_ROOT", "/srv/brainai")
KNOWN_QA = {  # 逐字复刻 api_server.py 的规则表：当前 /api/ask 的真实行为
    "法国": "法国的首都是巴黎。",
    "france": "法国的首都是巴黎。",
    "2+2": "二加二等于四。",
    "罗密欧": "罗密欧与朱丽叶是莎士比亚写的。",
}


# ---------------------------------------------------------------- 指标工具
def chrf(pred: str, ref: str, n: int = 3) -> float:
    """字符 n-gram F 值（与 cap_eval_beam.py 同口径）。"""
    gp = [pred[i:i + n] for i in range(len(pred) - n + 1)]
    gr = [ref[i:i + n] for i in range(len(ref) - n + 1)]
    if not gp or not gr:
        return 0.0
    cp, cr = Counter(gp), Counter(gr)
    ov = sum((cp & cr).values())
    if ov == 0:
        return 0.0
    p = ov / len(gp)
    r = ov / len(gr)
    return 2 * p * r / (p + r)


def retrieval(z_a: np.ndarray, z_b: np.ndarray):
    """R@1 / R@5（百分比，与 eval.py 同口径）。"""
    sim = z_a @ z_b.T
    ranks = sim.argsort(axis=1)[:, ::-1]
    n = sim.shape[0]
    target = np.arange(n)[:, None]
    r1 = float((ranks[:, :1] == target).any(1).mean() * 100)
    r5 = float((ranks[:, :5] == target).any(1).mean() * 100)
    return round(r1, 1), round(r5, 1)


def psnr(a, b) -> float:
    mse = float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    return float("inf") if mse == 0 else round(10 * np.log10(1.0 / mse), 2)


class Skip(Exception):
    """探针因缺少资产而跳过。"""


# ---------------------------------------------------------------- 探针
def probe_unified(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    _ = seed
    """统一模型 BrainNet：重建 MSE/PSNR、三路检索 R@1/R@5、生成字符准确率。"""
    import torch
    import torch.nn.functional as F

    import train
    from eval import psnr as _psnr, retrieval as _retrieval
    from model import BrainNet

    ckpt = root / (ckpt_arg or "ckpt/model.pt")
    if not ckpt.exists():
        raise Skip(f"缺权重 {ckpt}")
    train.ROOT = root  # RealDS / build_real_vocab 用模块级 ROOT，指向当前根目录
    from train import RealDS, build_real_vocab, encode_caption

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    vocab = state.get("vocab_map") or build_real_vocab()
    branches = len([k for k in state["model"]
                    if k.startswith("frame_heads.") and k.endswith(".fc.weight")]) or 2
    model = BrainNet(vocab=len(vocab), branches=branches)
    model.load_state_dict(state["model"])
    model.eval()

    ds = RealDS(vocab)
    if len(ds) == 0:
        raise Skip("没有真实数据（先跑 data_video.py 抽片段）")

    vids, mels, toks = [], [], []
    for i in range(min(n, len(ds))):
        v, a, c = ds[i]
        vids.append(v)
        mels.append(a)
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

    per_branch = [F.mse_loss(fh, small).item() for fh in out["frames"]]
    metrics = {
        "重建MSE": [round(x, 5) for x in per_branch],
        "重建PSNR": [round(_psnr(fh, small), 2) for fh in out["frames"]],
        "最优分支": int(np.argmin(per_branch)),
    }
    for name, (a_, b_) in {"视频→文本": (out["zv"], out["zt"]),
                            "音频→文本": (out["za"], out["zt"]),
                            "视频→音频": (out["zv"], out["za"])}.items():
        r1, r5 = _retrieval(a_, b_)
        metrics[f"{name}_R@1"] = round(r1 * 100, 1)
        metrics[f"{name}_R@5"] = round(r5 * 100, 1)

    ref, hyp = tokens.numpy(), gen_ids.numpy()
    ref_shift = ref[:, 1:]
    hyp_shift = hyp[:, :-1]
    m2 = ref_shift != 0
    metrics["生成字符准确率"] = round(
        float(((hyp_shift == ref_shift) & m2).sum() / max(m2.sum(), 1)) * 100, 1)
    metrics["参数量M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    return metrics


def probe_caption(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    _ = ckpt_arg
    """图像描述 chrF（cap_merged 模型 + concept_clf 条件），复用 report_final.py 的架构。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    feat_p = root / "data" / "all_clip.npz"
    cap_p = root / "ckpt" / "cap_merged.pt"
    clf_p = root / "ckpt" / "concept_clf.pt"
    missing = [str(p) for p in (feat_p, cap_p, clf_p) if not p.exists()]
    if missing:
        raise Skip("缺资产：" + ", ".join(missing))

    d = np.load(feat_p, allow_pickle=True)
    FE = F.normalize(torch.from_numpy(d["feats"]).float(), dim=-1)
    CAPS = [str(c) for c in d["captions"]]
    IMG = torch.from_numpy(d["img_idx"])

    old = torch.load(clf_p, map_location="cpu", weights_only=False)
    on = [str(c) for c in old["names"]]

    class OClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(512, 384), nn.GELU(), nn.LayerNorm(384),
                                     nn.Dropout(0.1), nn.Linear(384, 192), nn.GELU(),
                                     nn.Linear(192, len(on)))

        def forward(self, x):
            return self.net(x)

    oclf = OClf()
    oclf.load_state_dict(old["model"])
    oclf.eval()
    with torch.no_grad():
        COND = torch.cat([FE, torch.sigmoid(oclf(FE))], -1)

    ce = torch.load(cap_p, map_location="cpu", weights_only=False)
    vocab = ce["vocab"]
    inv = {v: k for k, v in vocab.items()}

    class Cap(nn.Module):
        def __init__(self, h=256, emb=96):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(COND.shape[1], 384), nn.GELU(),
                                     nn.LayerNorm(384), nn.Linear(384, h), nn.GELU())
            self.emb = nn.Embedding(len(vocab), emb)
            self.cproj = nn.Linear(COND.shape[1], h)
            self.gru = nn.GRU(emb + h, h, batch_first=True)
            self.init = nn.Linear(h, h)
            self.out = nn.Linear(h, len(vocab))

        def h0(self, c):
            return self.init(self.enc(c)).unsqueeze(0)

    m = Cap()
    m.load_state_dict(ce["model"])
    m.eval()

    @torch.no_grad()
    def gen(cond, s=0):
        torch.manual_seed(s)
        h = m.h0(cond)
        tok = torch.tensor([[1]])
        cp = m.cproj(cond)
        out = []
        for _ in range(40):
            o, hn = m.gru(torch.cat([m.emb(tok), cp.unsqueeze(1)], -1), h)
            nxt = int(m.out(o[:, -1])[0].argmax())
            if nxt == 2:
                break
            out.append(nxt)
            tok = torch.tensor([[nxt]])
            h = hn
        return "".join(inv.get(x, "") for x in out)

    imgs = sorted(set(IMG.tolist()))
    rng = np.random.default_rng(seed)
    picks = rng.choice(imgs, size=min(n, len(imgs)), replace=False)
    scores = []
    for j, i in enumerate(picks):
        g = gen(COND[i:i + 1], s=seed + j)
        refs = [CAPS[k] for k in range(len(CAPS)) if int(IMG[k]) == i]
        scores.append(max(chrf(g, r) for r in refs))
    return {"caption_chrF": round(float(np.mean(scores)) * 100, 1), "样本数": len(scores)}


def probe_sound_vq(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    _ = ckpt_arg
    """声音 VQ 码本使用率 + 重建 MSE（复用 amae2.py 的模型与评测段）。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = root / "amae2.pt"
    data_p = root / "data" / "esc_mels.npz"
    if not ckpt.exists():
        raise Skip(f"缺权重 {ckpt}")
    if not data_p.exists():
        raise Skip(f"缺数据 {data_p}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    K = ck.get("K", 64)
    ZD = ck.get("zdim", 64)
    mu, sd = ck.get("mu", 0.0), ck.get("sd", 1.0)
    d = np.load(data_p, allow_pickle=True)
    M = d["mels"][:, None, :, :]
    X = torch.from_numpy((M - mu) / sd).float()

    class Enc(nn.Module):
        def __init__(self, zd=64):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(128, zd, 3, 1, 1))

        def forward(self, x):
            return self.c(x)

    class Dec(nn.Module):
        def __init__(self, zd=64):
            super().__init__()
            self.c = nn.Sequential(nn.ConvTranspose2d(zd, 128, 4, 2, 1), nn.GELU(),
                                   nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
                                   nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(32, 1, 3, 1, 1))

        def forward(self, z):
            return self.c(z)

    class VQ(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.emb = nn.Embedding(k, ZD)
            self.K = k

        def forward(self, r):
            B, C, H, W = r.shape
            zf = r.permute(0, 2, 3, 1).reshape(-1, C)
            zf = F.normalize(zf, dim=-1)
            e = F.normalize(self.emb.weight, dim=-1)
            dist = (zf ** 2).sum(1, keepdim=True) - 2 * zf @ e.T + (e ** 2).sum(1)[None]
            idx = dist.argmin(1)
            q = self.emb(idx)
            return q.view(B, H, W, C).permute(0, 3, 1, 2), idx

    enc = Enc(ZD)
    dec = Dec(ZD)
    vqs = [VQ(K) for _ in range(ck.get("nbook", 3))]
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])
    for v, sd in zip(vqs, ck["vq"]):
        v.load_state_dict(sd)
    enc.eval()
    dec.eval()

    def quantize(z):
        r = z
        qs, idxs = [], []
        for v in vqs:
            q, idx = v(r)
            qs.append(q)
            idxs.append(idx)
            r = r - q.detach()
        return sum(qs), qs, idxs

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    xt = X[idx]
    with torch.no_grad():
        z = enc(xt)
        zq, qs, idxs = quantize(z)
        usage = [int(len(torch.unique(ix))) for ix in idxs]
        metrics = {f"码本{bi}_使用率": f"{u}/{K}" for bi, u in enumerate(usage)}
        for nq in (1, 2, 3):
            rec = dec(sum(qs[:nq]))
            metrics[f"{nq}码本_MSE"] = round(F.mse_loss(rec, xt).item(), 4)
        re_z = enc(dec(sum(qs)))
        metrics["往返一致性"] = round(
            F.cosine_similarity(z.flatten(1), re_z.flatten(1), dim=-1).mean().item(), 3)
        rec = dec(sum(qs))
        for name, sl in (("低频", slice(0, 27)), ("中频", slice(27, 54)), ("高频", slice(54, 80))):
            metrics[f"{name}误差"] = round(
                F.mse_loss(rec[:, :, sl, :], xt[:, :, sl, :]).item(), 4)
    return metrics


def probe_sound_clf(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    _ = ckpt_arg
    """声音分类准确率（asem 量化 token vs 连续潜变量，复用 asem.py 的架构）。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ckpt = root / "asem.pt"
    data_p = root / "data" / "esc_mels.npz"
    if not ckpt.exists():
        raise Skip(f"缺权重 {ckpt}")
    if not data_p.exists():
        raise Skip(f"缺数据 {data_p}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    K, ZD, NT = ck["K"], ck["ZD"], ck["NT"]
    mu, sd = ck.get("mu", 0.0), ck.get("sd", 1.0)
    d = np.load(data_p, allow_pickle=True)
    M = d["mels"][:, None, :, :]
    Y = torch.from_numpy(d["target"]).long()
    X = torch.from_numpy((M - mu) / sd).float()
    NC = int(Y.max()) + 1

    class Enc(nn.Module):
        def __init__(self, zd=128):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),
                                   nn.Conv2d(128, zd, 3, 1, 1), nn.GELU())
            self.pool = nn.AdaptiveAvgPool2d((5, 8))

        def forward(self, x):
            z = self.pool(self.c(x))
            B, C, H, W = z.shape
            return z.permute(0, 2, 3, 1).reshape(B, H * W, C)

    class VQ(nn.Module):
        def __init__(self, k, zd):
            super().__init__()
            self.emb = nn.Embedding(k, zd)

        def forward(self, r):
            B, T, C = r.shape
            zf = r.reshape(-1, C)
            e = self.emb.weight
            dist = (zf ** 2).sum(1, keepdim=True) - 2 * zf @ e.T + (e ** 2).sum(1)[None]
            idx = dist.argmin(1)
            q = e[idx]
            return q.view(B, T, C), idx.view(B, T)

    enc = Enc(ZD)
    vqs = [VQ(K, ZD) for _ in range(ck.get("nbook", 2))]
    clf = nn.Sequential(nn.Flatten(), nn.Linear(NT * ZD, 512), nn.GELU(),
                        nn.Dropout(0.2), nn.Linear(512, NC))
    enc.load_state_dict(ck["enc"])
    clf.load_state_dict(ck["clf"])
    for v, sd in zip(vqs, ck["vq"]):
        v.load_state_dict(sd)
    enc.eval()
    clf.eval()

    def quantize(z):
        r = z
        qs = []
        for v in vqs:
            q, _ = v(r)
            qs.append(q)
            r = r - q.detach()
        return sum(qs)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    xv, yv = X[idx], Y[idx]
    with torch.no_grad():
        zv = enc(xv)
        zqv = quantize(zv)
        acc_q = float((clf(zqv.reshape(len(xv), -1)).argmax(-1) == yv).float().mean())
        acc_c = float((clf(zv.reshape(len(xv), -1)).argmax(-1) == yv).float().mean())
    return {"分类准确率_连续": round(acc_c * 100, 1), "分类准确率_量化": round(acc_q * 100, 1)}


def probe_qa(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    """问答内容准确率：固定测试集，按当前 /api/ask 行为（规则表）评分。

    口径说明：Issue 记 27%，report_overview.py 记 54.6%；本探针统一为
    「固定测试集上，期望答案子串命中率」，规则表口径可复现，接真模型后同口径复跑。
    """
    _ = (n, seed, ckpt_arg)
    test = [
        ("法国的首都是哪里？", "巴黎"),
        ("2+2 等于几？", "四"),
        ("罗密欧与朱丽叶是谁写的？", "莎士比亚"),
        ("太阳系有几大行星？", "八"),  # 规则表之外的开放题，当前应判错
    ]
    hits = 0
    detail = []
    for q, expect in test:
        ans = next((v for k, v in KNOWN_QA.items() if k in q), None) or "（模型能力有限，暂时无法回答这个问题）"
        ok = expect in ans
        hits += int(ok)
        detail.append({"question": q, "expected": expect, "answer": ans, "correct": ok})
    return {"问答内容准确率": round(hits / len(test) * 100, 1), "口径": "rule-table", "详情": detail}


def probe_latency(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    """端到端语音管道分段延迟（复用 stream_pipeline.py 的流式分句 + espeak TTS）。"""
    _ = (n, seed, ckpt_arg)
    from stream_pipeline import fake_token_stream, split_sentences, tts_local

    if not shutil.which("espeak-ng"):
        raise Skip("缺 espeak-ng，无法测 TTS 延迟")

    import tempfile
    t_start = time.time()
    first_audio = None
    total_tts = 0.0
    n_sent = 0
    for sent in split_sentences(fake_token_stream("What is the capital of France?")):
        out = tempfile.mktemp(suffix=".wav")
        lat = tts_local(sent, out)
        if first_audio is None:
            first_audio = time.time() - t_start
        total_tts += lat
        n_sent += 1
        if os.path.exists(out):
            os.remove(out)
    return {"首音延迟_ms": round((first_audio or 0) * 1000, 0),
            "总耗时_s": round(time.time() - t_start, 2),
            "句数": n_sent}


def probe_selftest(root: Path, n: int, seed: int, ckpt_arg: str | None) -> dict:
    _ = (n, ckpt_arg)
    """无资产自检：验证指标函数（chrF / retrieval / psnr）与落盘逻辑。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(8, 16)).astype(np.float32)
    b = rng.normal(size=(8, 16)).astype(np.float32)
    # 让 b 的第 0 行与 a 的第 0 行高度相关，保证 R@1 可判
    b[0] = a[0] + 0.01 * rng.normal(size=16).astype(np.float32)
    r1, r5 = retrieval(a, b)
    assert r1 >= 12.5 and r5 >= 12.5, (r1, r5)  # 随机 12.5%
    assert 0.0 <= chrf("abc", "abd") <= 1.0
    assert psnr(np.ones(4), np.ones(4)) == float("inf")
    return {"检索R@1_自检": r1, "检索R@5_自检": r5, "chrF_自检": round(chrf("abc", "abd"), 4)}


# ---------------------------------------------------------------- 主流程
PROBES = {
    "unified": probe_unified,
    "caption": probe_caption,
    "sound_vq": probe_sound_vq,
    "sound_clf": probe_sound_clf,
    "qa": probe_qa,
    "latency": probe_latency,
    "selftest": probe_selftest,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Brainai 统一自动评测")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="仓库根目录（含 data/ ckpt/ logs/）")
    ap.add_argument("--ckpt", default=None, help="统一模型权重路径（相对 --root），默认 ckpt/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=32, help="每探针抽样条数")
    ap.add_argument("--only", default=None, help="逗号分隔的探针名，只跑这些")
    ap.add_argument("--out", default=None, help="输出 JSON 路径，默认 <root>/logs/baseline.json")
    ap.add_argument("--selftest", action="store_true", help="只跑 selftest 探针")
    args = ap.parse_args()

    root = Path(args.root)
    names = list(PROBES)
    if args.selftest:
        names = ["selftest"]
    elif args.only:
        names = [x.strip() for x in args.only.split(",") if x.strip()]

    torch_seed_done = False
    results: dict[str, dict] = {}
    for name in names:
        if not torch_seed_done:
            try:
                import torch
                torch.manual_seed(args.seed)
                torch_seed_done = True
            except Exception:
                pass
        np.random.seed(args.seed)
        t0 = time.time()
        try:
            metrics = PROBES[name](root, args.n, args.seed, args.ckpt)
            results[name] = {"status": "ok", "metrics": metrics, "seconds": round(time.time() - t0, 1)}
        except Skip as e:
            results[name] = {"status": "skip", "reason": str(e), "metrics": {}}
        except Exception as e:  # 单探针失败不拖垮整次评测
            results[name] = {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:200]}",
                             "metrics": {}}

    flat = {}
    for name, r in results.items():
        if r["status"] == "ok":
            for k, v in r["metrics"].items():
                if not isinstance(v, (dict, list)):
                    flat.setdefault(name, {})[k] = v

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "ckpt": args.ckpt or "ckpt/model.pt",
        "seed": args.seed,
        "n": args.n,
        "probes": results,
        "metrics": flat,
    }

    out_p = Path(args.out) if args.out else root / "logs" / "baseline.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = out_p.parent / f"baseline_{stamp}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    out_p.write_text(text, encoding="utf-8")
    archive.write_text(text, encoding="utf-8")

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    skip = sum(1 for r in results.values() if r["status"] == "skip")
    err = sum(1 for r in results.values() if r["status"] == "error")
    print(json.dumps({name: {k: v for k, v in r.items() if k != "metrics"}
                      for name, r in results.items()}, ensure_ascii=False, indent=2))
    print(f"ok {ok} · skip {skip} · error {err}")
    print("baseline →", out_p)
    print("archive  →", archive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
