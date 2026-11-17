#!/usr/bin/env python3
"""模型加载层：图像描述（vlm 13.6M）+ 语音识别（sherpa zipformer）+ 语音合成（VITS）。"""
import os, io, re, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from pathlib import Path
ROOT=Path("/srv/brainai")
DEV="cpu"
ALLOWED=["<pad>","<bos>","<eos>","<unk>"]

# ---------- 图像描述 ----------
class _Block(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.h=h; self.dh=d//h
        self.qkv=nn.Linear(d,d*3); self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x,imp=None,mask=None):
        B,T,D=x.shape
        q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        q=q.view(B,T,self.h,self.dh).transpose(1,2); k=k.view(B,T,self.h,self.dh).transpose(1,2); v=v.view(B,T,self.h,self.dh).transpose(1,2)
        att=(q@k.transpose(-1,-2))/math.sqrt(self.dh)
        if mask is not None: att=att+mask
        if imp is not None: att=att+torch.log(imp.clamp(min=1e-3))[:,None,None,:]
        att=att.softmax(-1)
        o=(att@v).transpose(1,2).reshape(B,T,D)
        x=x+self.proj(o); return x+self.ff(self.ln2(x))

class VLM(nn.Module):
    def __init__(self,V,d=384,h=8,layers=6,P=4,L=26):
        super().__init__()
        self.P=P; self.L=L
        self.proj=nn.Linear(512,d*P); self.tok=nn.Embedding(V,d)
        self.pos=nn.Embedding(P+L+4,d); self.imp=nn.Parameter(torch.zeros(V))
        self.blocks=nn.ModuleList([_Block(d,h) for _ in range(layers)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,V,bias=False); self.head.weight=self.tok.weight
    def forward(self,f,x):
        B,T=x.shape
        pre=self.proj(f).view(B,self.P,-1)
        h=torch.cat([pre,self.tok(x)],1)+self.pos(torch.arange(self.P+T,device=x.device)[None])
        impx=torch.cat([torch.ones(B,self.P,device=x.device),self.imp[x].clamp(min=1e-3)],1)
        Tt=self.P+T
        causal=torch.triu(torch.full((Tt,Tt),float("-inf"),device=x.device),1)[None,None]
        for b in self.blocks: h=b(h,impx,causal)
        return self.head(self.ln(h)[:,self.P:])

class Captioner:
    def __init__(self,ckpt="/tmp/vlm_gpu.pt",clip="/srv/brainai/data/downloads/clip_vision.onnx"):
        ck=torch.load(ckpt,map_location="cpu",weights_only=False)
        self.vocab=ck["vocab"]; V=len(self.vocab)
        self.i2w={v:k for k,v in self.vocab.items()}
        self.m=VLM(V)
        sd=ck["model"]; cur=self.m.state_dict(); hit=0
        for k,v in sd.items():
            if k in cur and cur[k].shape==v.shape: cur[k]=v; hit+=1
        self.m.load_state_dict(cur,strict=False); self.m.eval()
        import onnxruntime as ort
        self.sess=ort.InferenceSession(clip,providers=["CPUExecutionProvider"])
        self.iname=self.sess.get_inputs()[0].name; self.oname=self.sess.get_outputs()[0].name
        print(f"[captioner] 载入 {hit} 个张量，词表 {V}",flush=True)
    def _feat(self,img_bytes):
        import cv2
        arr=cv2.imdecode(np.frombuffer(img_bytes,np.uint8),cv2.IMREAD_COLOR)
        rgb=cv2.cvtColor(cv2.resize(arr,(224,224)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        mean=np.array([0.48145466,0.4578275,0.40821073],np.float32); std=np.array([0.26862954,0.26130258,0.27577711],np.float32)
        x=((rgb-mean)/std).transpose(2,0,1)[None]
        f=self.sess.run([self.oname],{self.iname:x})[0]
        return torch.tensor(F.normalize(torch.from_numpy(f),dim=-1))
    @torch.no_grad()
    def caption(self,img_bytes,maxn=26,width=4):
        f=self._feat(img_bytes)
        beams=[([1],0.0)]; done=[]
        for _ in range(maxn):
            cand=[]
            for toks,sc in beams:
                x=torch.tensor([toks])
                pre=self.m.proj(f).view(1,self.m.P,-1)
                h=torch.cat([pre,self.m.tok(x)],1)+self.m.pos(torch.arange(self.m.P+x.size(1))[None])
                impx=torch.cat([torch.ones(1,self.m.P),self.m.imp[x].clamp(min=1e-3)],1)
                Tt=self.m.P+x.size(1)
                causal=torch.triu(torch.full((Tt,Tt),float("-inf")),1)[None,None]
                for b in self.m.blocks: h=b(h,impx,causal)
                lp=F.log_softmax(self.m.head(self.m.ln(h))[:,-1][0],-1)
                v,k=torch.topk(lp,width)
                for vv,kk in zip(v.tolist(),k.tolist()): cand.append((toks+[kk],sc+vv))
            cand.sort(key=lambda t:-t[1]/max(len(t[0])-1,1))
            beams=[]
            for toks,sc in cand[:width]:
                if toks[-1]==2: done.append((toks,sc))
                else: beams.append((toks,sc))
            if not beams: break
        if not done: done=[(b[0],b[1]) for b in beams]
        best=max(done,key=lambda t:t[1]/max(len(t[0])-1,1))[0]
        out="".join(self.i2w.get(x,"") for x in best[1:-1])
        return out.replace("  "," ").strip()

# ---------- 语音识别 ----------
class ASR:
    def __init__(self,model_dir="/srv/brainai/data/asr"):
        import sherpa_onnx
        p=Path(model_dir)
        self.rec=sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(p/"tokens.txt"),
            encoder=str(p/"encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(p/"decoder-epoch-99-avg-1.int8.onnx"),
            joiner=str(p/"joiner-epoch-99-avg-1.int8.onnx"),
            num_threads=4,sample_rate=16000,feature_dim=80,decoding_method="greedy_search")
        print("[asr] 就绪",flush=True)
    def transcribe(self,wav_bytes):
        import scipy.signal as ss
        y,sr=sf.read(io.BytesIO(wav_bytes),dtype="float32")
        if y.ndim>1: y=y.mean(1)
        if sr!=16000: y=ss.resample(y,int(len(y)*16000/sr)).astype(np.float32)
        s=self.rec.create_stream(); s.accept_waveform(16000,y.astype(np.float32))
        s.accept_waveform(16000,np.zeros(8000,dtype=np.float32)); s.input_finished()
        while self.rec.is_ready(s): self.rec.decode_stream(s)
        return self.rec.get_result(s)

# ---------- 语音合成 ----------
class TTS:
    def __init__(self,model_dir="/srv/brainai/data/vits"):
        import sherpa_onnx
        p=Path(model_dir)
        cfg=sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(p/"model.onnx"),tokens=str(p/"tokens.txt"),
                    lexicon=str(p/"lexicon.txt"),dict_dir=str(p/"dict")),
                num_threads=4),max_num_sentences=1)
        self.tts=sherpa_onnx.OfflineTts(cfg)
        print("[tts] 就绪 采样率",self.tts.sample_rate,flush=True)
    def synth(self,text,speed=1.0):
        a=self.tts.generate(text,sid=0,speed=speed)
        y=np.array(a.samples,dtype=np.float32)
        buf=io.BytesIO(); sf.write(buf,y,a.sample_rate,format="WAV"); buf.seek(0)
        return buf.read()
