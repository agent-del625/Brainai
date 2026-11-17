#!/usr/bin/env python3
"""ESC-50 全部 2000 条音频 → mel 频谱（80x128），供声音生成模型训练。"""
import numpy as np, pyarrow.parquet as pq, soundfile as sf, io
from pathlib import Path
ROOT=Path("/srv/brainai")
def mel_spec(y,sr=22050,n_mels=80,hop=512,n_fft=1024,frames=128):
    import scipy.signal as ss
    f,t,Z=ss.stft(y,nperseg=n_fft,noverlap=n_fft-hop,boundary=None)
    mag=np.abs(Z)
    fb=np.zeros((n_mels,f.shape[0]))
    def h2m(h): return 2595*np.log10(1+h/700)
    def m2h(m): return 700*(10**(m/2595)-1)
    mm=np.linspace(h2m(0),h2m(sr/2),n_mels+2)
    hz=m2h(mm); bins=np.floor((n_fft+1)*hz/sr).astype(int)
    for i in range(n_mels):
        l,r=bins[i],bins[i+2]
        if r<=l: continue
        for j in range(l,min(r,f.shape[0])): fb[i,j]=1.0/(r-l)
    M=np.log1p(fb@mag)
    if M.shape[1]<frames: M=np.pad(M,((0,0),(0,frames-M.shape[1])))
    M=M[:,:frames]
    M=(M-M.min())/max(M.max()-M.min(),1e-6)
    return M.astype(np.float32)
mels=[]; cats=[]; tg=[]
for f in ("esc50_a.parquet","esc50_b.parquet"):
    t=pq.read_table(ROOT/"data/downloads"/f).to_pydict()
    for i in range(len(t["filename"])):
        a=t["audio"][i]
        b=a["bytes"] if isinstance(a,dict) else a
        try: y,sr=sf.read(io.BytesIO(b),dtype="float32")
        except Exception: continue
        if y.ndim>1: y=y.mean(1)
        if sr!=22050:
            import scipy.signal as ss
            y=ss.resample(y,int(len(y)*22050/sr))
        mels.append(mel_spec(y)); cats.append(t["category"][i]); tg.append(int(t["target"][i]))
    print(f,len(mels),flush=True)
np.savez_compressed(ROOT/"data/esc_mels.npz", mels=np.stack(mels), cats=np.array(cats), target=np.array(tg))
print("完成",len(mels),"条，mel",mels[0].shape)
