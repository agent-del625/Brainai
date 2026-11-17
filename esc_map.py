#!/usr/bin/env python3
"""ESC-50 → 概念声音原型：每个类别取几段音频，算平均梅尔谱。"""
import io, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from audio_feat import mel_spec

ROOT = Path("/srv/brainai")
FILES = [ROOT / "data/downloads/esc50_a.parquet", ROOT / "data/downloads/esc50_b.parquet"]

def main() -> int:
    per_cat = defaultdict(list)
    for f in FILES:
        t = pq.read_table(f)
        for i in range(t.num_rows):
            cat = t["category"][i].as_py()
            if len(per_cat[cat]) >= 3:
                continue
            raw = t["audio"][i].as_py()
            b = raw["bytes"] if isinstance(raw, dict) else raw
            try:
                data, sr = sf.read(io.BytesIO(b), dtype="float32")
            except Exception:
                continue
            if data.ndim > 1:
                data = data.mean(1)
            if sr != 16000:
                idx = (np.arange(int(len(data) * 16000 / sr)) * sr / 16000).astype(int)
                data = data[np.clip(idx, 0, len(data) - 1)]
            per_cat[cat].append(mel_spec(data[: 5 * 16000], n_mels=80, frames=128))
    print("类别数", len(per_cat))
    out = {"cats": sorted(per_cat), "mels": []}
    for c in out["cats"]:
        out["mels"].append(np.mean(per_cat[c], 0))
    np.savez_compressed(ROOT / "data/esc50_proto.npz",
                        cats=np.array(out["cats"]), mels=np.stack(out["mels"]))
    print("保存", (ROOT / "data/esc50_proto.npz"), np.stack(out["mels"]).shape)
    print("类别示例:", out["cats"][:12])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
