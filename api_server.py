#!/usr/bin/env python3
"""Brainai 模型服务：给 Svelte 前端提供 /api 接口（先接已有能力，逐步替换为真模型）。"""
import os, io, json, sys, numpy as np
from flask import Flask, request, jsonify, send_file
ROOT="/srv/brainai"
sys.path.insert(0,ROOT)
app=Flask(__name__)
_models={}
def get(name):
    """按需加载模型，避免启动时全部占用内存"""
    if name not in _models:
        from api_model import Captioner, ASR, TTS
        _models[name]={"captioner":Captioner,"asr":ASR,"tts":TTS}[name]()
    return _models[name]

@app.get("/api/health")
def health():
    return jsonify({"ok":True,"service":"brainai-api","version":"0.1"})

@app.post("/api/ask")
def ask():
    q=(request.get_json(force=True,silent=True) or {}).get("question","").strip()
    if not q: return jsonify({"error":"empty question"}),400
    # 先用确定知识表；后续接 sft 模型
    KNOWN={"法国":"法国的首都是巴黎。","france":"法国的首都是巴黎。",
           "2+2":"二加二等于四。","罗密欧":"罗密欧与朱丽叶是莎士比亚写的。"}
    ans=next((v for k,v in KNOWN.items() if k in q), None)
    if ans is None:
        ans="（模型能力有限，暂时无法回答这个问题）"
    return jsonify({"question":q,"answer":ans,"source":"rule-table"})

@app.post("/api/caption")
def caption():
    f=request.files.get("image")
    if not f: return jsonify({"error":"no image"}),400
    raw=f.read()
    try:
        cap=get("captioner").caption(raw)
        return jsonify({"caption":cap,"bytes":len(raw),"model":"vlm-13.6M"})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.post("/api/asr")
def asr():
    f=request.files.get("audio")
    if not f: return jsonify({"error":"no audio"}),400
    raw=f.read()
    try:
        text=get("asr").transcribe(raw)
        return jsonify({"text":text,"bytes":len(raw),"model":"zipformer-zh-en"})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.post("/api/tts")
def tts():
    data=request.get_json(force=True,silent=True) or {}
    text=(data.get("text") or "").strip()
    if not text: return jsonify({"error":"empty text"}),400
    try:
        wav=get("tts").synth(text)
        return send_file(io.BytesIO(wav),mimetype="audio/wav",as_attachment=False,download_name="tts.wav")
    except Exception as e:
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    port=int(os.environ.get("PORT","8800"))
    app.run(host="127.0.0.1",port=port,threaded=True)
