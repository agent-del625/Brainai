#!/usr/bin/env python3
"""合并物体概念（ImageNet 自动匹配）+ 场景概念（Places），去重后存统一原型。"""
from pathlib import Path
import numpy as np
ROOT = Path("/srv/brainai")
cp = np.load(ROOT/"data/concept_protos.npz", allow_pickle=True)
sp = np.load(ROOT/"data/scene_proto.npz", allow_pickle=True)
names, feats = {}, {}
for src in (cp, sp):
    for c, f in zip([str(x) for x in src["cats"]], src["feats"]):
        if c not in names:
            names[c] = 1; feats[c] = f
cats = sorted(names)
np.savez_compressed(ROOT/"data/concept_all.npz", cats=np.array(cats),
                    feats=np.stack([feats[c] for c in cats]).astype(np.float32))
print(f"合并后 {len(cats)} 个概念: {cats}")
