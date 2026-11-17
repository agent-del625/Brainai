#!/usr/bin/env python3
import numpy as np, soundfile as sf
from pathlib import Path
d=np.load("/tmp/sgan_demo.npz",allow_pickle=True)
mels=d["mels"]; words=[str(w) for w in d["words"]]
sr=22050; n_fft=1024; hop=512; n_mels=80
import scipy.signal as ss
def h2m(h): return 2595*np.log10(1+h/700)
def m2h(m): return 700*(10**(m/2595)-1)
mm=np.linspace(h2m(0),h2m(sr/2),n_mels+2); hz=m2h(mm)
bins=np.floor((n_fft+1)*hz/sr).astype(int)
fb=np.zeros((n_mels,n_fft//2+1))
for i in range(n_mels):
    l,r=bins[i],bins[i+2]
    if r<=l: continue
    for j in range(l,min(r,n_fft//2+1)): fb[i,j]=1.0/(r-l)
out=Path("/home/del/dsh-workspace/out/audio2"); out.mkdir(parents=True,exist_ok=True)
for m,w in zip(mels,words):
    mag=fb.T@np.expm1(np.clip(m,0,10))
    ang=np.exp(2j*np.pi*np.random.rand(*mag.shape))
    _,y=ss.istft(mag*ang,nperseg=n_fft,noverlap=n_fft-hop)
    y=(y/max(np.abs(y).max(),1e-6)*0.9).astype(np.float32)
    sf.write(str(out/f"gan_{w}.wav"),y,sr)
print("已生成", [p.name for p in sorted(out.iterdir())])
