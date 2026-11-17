#!/usr/bin/env python3
"""第 3 步闭环：图 → 句子 → 概念 → 图像 + 声音。"""
import io, wave
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from audio_feat import mel_to_wav
from model_v4 import BrainV4
from stage1_order import Orderer, words_of, K, MAXLEN

ROOT = Path("/srv/brainai")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# ---- CLIP ----
clip = ort.InferenceSession(str(ROOT/"data/downloads/clip_vision.onnx"), providers=["CPUExecutionProvider"])
ci, co = clip.get_inputs()[0].name, clip.get_outputs()[0].name
def clip_feat(path):
    im = Image.open(path).convert("RGB").resize((224,224), Image.BILINEAR)
    x = np.asarray(im, np.float32)/255.0
    x = ((x-MEAN)/STD).transpose(2,0,1)[None]
    return torch.from_numpy(clip.run([co], {ci: x})[0]).float()

# ---- 第1段 ----
s1 = torch.load(ROOT/"ckpt/stage1_order.pt", map_location="cpu", weights_only=False)
content = s1["content"]; c2i = {w:i for i,w in enumerate(content)}
m1 = Orderer(n=len(content), d=192); m1.load_state_dict(s1["model"]); m1.eval()

# ---- 第2段 ----
s2 = torch.load(ROOT/"ckpt/stage2_big.pt", map_location="cpu", weights_only=False)
patterns, leadp = s2["patterns"], s2["leadp"]
import torch.nn as nn
D,H = 128,512
class Big(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(content), D); self.pos = nn.Embedding(14, D)
        self.gap = nn.Sequential(nn.Linear(D*5,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                 nn.Linear(H,H//2), nn.GELU(), nn.Linear(H//2,len(patterns)))
        self.lead = nn.Sequential(nn.Linear(D*3,H), nn.GELU(), nn.LayerNorm(H), nn.Dropout(0.1),
                                  nn.Linear(H,H//2), nn.GELU(), nn.Linear(H//2,len(leadp)))
m2 = Big(); m2.load_state_dict(s2["model"]); m2.eval()

def order_from_feat(f, words_hint=None):
    """给出句子里的内容词顺序。真实场景里"袋"由袋头预测，这里用 hint 模拟。"""
    ws = words_hint or []
    rows = [c2i[w] for w in ws if w in c2i][:K]; rows += [0]*(K-len(rows))
    with torch.no_grad():
        h0 = m1.enc(f)
        bag = m1.emb(torch.tensor([rows])); h = torch.zeros(1,192); prev = torch.zeros(1,192)
        used = torch.zeros(1,K,dtype=torch.bool); pad = torch.tensor(rows)==0; out=[]
        for t in range(MAXLEN):
            if bool((used|pad)[0].all()): break
            remain = (bag*(~used).unsqueeze(-1).float()).sum(1)/(~used).sum(1,keepdim=True).clamp(min=1)
            lg,h = m1.step(h0,bag,prev,h,t,remain)
            j = int(lg[:,:K].masked_fill(used|pad,-1e4).argmax())
            out.append(content[rows[j]]); used[0,j]=True; prev = m1.emb(torch.tensor([rows[j]]))
    return out

def fill(seq):
    if len(seq) < 2: return seq
    P = lambda i: c2i[seq[max(0,min(i,len(seq)-1))]]
    with torch.no_grad():
        lead = leadp[int(m2.lead(torch.cat([m2.emb(torch.tensor([P(0)])), m2.emb(torch.tensor([P(0)])),
                                            m2.emb(torch.tensor([P(0)]))],-1)).argmax())]
        out = lead.split() if lead != "<none>" else []
        for j in range(len(seq)-1):
            x = torch.cat([m2.emb(torch.tensor([P(j)])), m2.emb(torch.tensor([P(j+1)])),
                           m2.emb(torch.tensor([P(j-1)])), m2.emb(torch.tensor([P(j+2)])),
                           m2.pos(torch.tensor([j]))],-1)
            p = patterns[int(m2.gap(x).argmax())]
            out.append(seq[j])
            if p != "<none>": out += p.split()
        out.append(seq[-1]); return out

# ---- 概念声音 ----
ep = np.load(ROOT/"data/esc50_proto.npz", allow_pickle=True)
ecats, emels = list(ep["cats"]), ep["mels"]
SOUND = {"dog":"dog","cat":"cat","bird":"chirping_birds","train":"train","clock":"clock_alarm",
         "keyboard":"keyboard_typing","plane":"airplane","beach":"sea_waves","forest":"crickets",
         "ocean":"sea_waves","river":"water_drops","waterfall":"pouring_water"}

OUT = ROOT/"logs/stage3"; OUT.mkdir(parents=True, exist_ok=True)
cases = [("dog", ["dog","running","grass"]), ("beach", ["people","walking","beach"]),
         ("cat", ["cat","sitting","table"]), ("train", ["train","station","platform"])]
rows = []
for name, hint in cases:
    gal = sorted((ROOT/"data/gallery").rglob(f"{name}_*.jpg"))
    if not gal: continue
    img_path = gal[0]
    f = clip_feat(img_path)
    seq = order_from_feat(f, hint)
    sent = " ".join(fill(seq))
    # 概念命中
    hits = [w for w in set(hint) if w in SOUND]
    print(f"\n[{name}] 生成句子: {sent}")
    print(f"  命中概念: {hits}")
    tiles = [cv2.resize(cv2.imread(str(img_path)), (160,160))]
    for h in hits:
        fl = gal_files = sorted((ROOT/"data/gallery").rglob(f"{h}_*.jpg"))[:2]
        for g in fl:
            tiles.append(cv2.resize(cv2.imread(str(g)), (160,160)))
        wav = mel_to_wav(emels[ecats.index(SOUND[h])])
        p = OUT/f"{name}_{h}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(wav.tobytes())
        print(f"  声音: {p.name}")
    rows.append(np.hstack(tiles) if tiles else None)
sheet = np.vstack([r for r in rows if r is not None])
cv2.putText(sheet, "input | concept images", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.imwrite(str(OUT/"sheet.png"), sheet)
print("\n合成图:", OUT/"sheet.png")
