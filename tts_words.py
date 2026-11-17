#!/usr/bin/env python3
"""给概念词合成发音（MiniMax TTS），供"图片-词-声音"三元绑定训练。"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
import urllib.request

WORDS = ["dog","cat","ball","car","bike","boy","girl","man","woman","water","tree","grass",
         "snow","beach","boat","horse","bird","food","table","chair","phone","book","street",
         "bridge","fire","music","guitar","cake","coffee","window","door","shirt","hat","shoe",
         "train","plane","swim","run","jump","climb"]

URL = "https://api.minimaxi.com/v1/t2a_v2"
KEY = os.environ.get("MINIMAX_CN_API_KEY") or "sk-cp-k3tYD5d8cyt2Z4s-r3DeSE2N5EhcGt-dzU3PltLo4-GfoNdJOmwZnJ5xahFIo_cj5uDGNLmoL858rn7p7xA3jOBROKVDmd7ThrneHqBZ4VXp8RvtdF5FzB0"
OUT = Path("/srv/brainai/data/words"); OUT.mkdir(parents=True, exist_ok=True)

def tts(text: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 1000:
        return True
    body = json.dumps({"model": "speech-2.8-hd", "text": text, "stream": False,
                       "output_format": "hex",
                       "voice_setting": {"voice_id": "male-qn-qingse", "speed": 0.9, "pitch": 0},
                       "audio_setting": {"format": "mp3", "sample_rate": 16000}}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read())
        hx = (d.get("data") or {}).get("audio")
        if not hx:
            print("失败", text, str(d)[:120]); return False
        path.write_bytes(bytes.fromhex(hx))
        return True
    except Exception as e:
        print("异常", text, str(e)[:80]); return False

ok = 0
for w in WORDS:
    if tts(w, OUT / f"{w}.mp3"):
        ok += 1
print(f"完成 {ok}/{len(WORDS)}，目录 {OUT}")
