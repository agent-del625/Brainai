#!/usr/bin/env python3
"""用强分类器评估生成的声音，并把生成 mel 转成 wav。"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from pathlib import Path
ROOT=Path("/srv/brainai")
DEV="cpu"
dd=384
class VAE(nn.Module):
    def __init__(self,z=64,cd=384):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(1,32,3,2,1),nn.GELU(),nn.Conv2d(32,64,3,2,1),nn.GELU(),
                               nn.Conv2d(64,128,3,2,1),nn.GELU(),nn.Conv2d(128,128,3,2,1),nn.GELU(),
                               nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,128))
        self.mu=nn.Linear(128,z); self.lv=nn.Linear(128,z)
        self.cp=nn.Sequential(nn.Linear(cd,256),nn.GELU(),nn.Linear(256,256))
        self.fc=nn.Linear(z+256,128*10*16)
        self.dec=nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1),nn.GELU(),
                               nn.ConvTranspose2d(64,32,4,2,1),nn.GELU(),
                               nn.ConvTranspose2d(32,1,4,2,1),nn.Sigmoid())
    def decode(self,z,c):
        h=torch.cat([z,self.cp(c)],-1); h=self.fc(h).view(-1,128,10,16); return self.dec(h).squeeze(1)
class Clf(nn.Module):
    def __init__(self,NC=50):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,32,3,1,1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
                               nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
                               nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.GELU(),nn.MaxPool2d(2),
                               nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.GELU(),nn.AdaptiveAvgPool2d(1),
                               nn.Flatten(),nn.Dropout(0.3),nn.Linear(256,NC))
    def forward(self,x): return self.net(x.unsqueeze(1))
ck=torch.load("/tmp/sound_vae.pt",map_location="cpu",weights_only=False)
vae=VAE(); vae.load_state_dict(ck["vae"]); vae.eval()
clf=Clf(); clf.load_state_dict(torch.load("/tmp/mel_clf.pt",map_location="cpu")["model"]); clf.eval()
E=np.load(ROOT/"data/concept_emb.npz",allow_pickle=True)
words=[str(w) for w in E["esc_words"]]; EM=torch.from_numpy(E["esc_emb"]).float()
hits=0; tot=0; per=[]
with torch.no_grad():
    for ci in range(len(words)):
        c=EM[ci:ci+1].repeat(6,1); z=torch.randn(6,64)
        g=vae.decode(z,c); pred=clf(g).argmax(-1)
        h=int((pred==ci).sum()); hits+=h; tot+=6; per.append(h)
print(f"生成声音被正确识别 {hits}/{tot} = {hits/tot*100:.1f}%（随机 {100/50:.1f}%）")
best=np.argsort(per)[::-1][:8]
print("最容易认出的类别:", [(words[i],per[i]) for i in best])
# 生成 4 个示例并转 wav
def mel_to_wav(m,sr=22050,n_fft=1024,hop=512):
    import scipy.signal as ss
    n_mels=80
    def h2m(h): return 2595*np.log10(1+h/700)
    def m2h(m): return 700*(10**(m/2595)-1)
    mm=np.linspace(h2m(0),h2m(sr/2),n_mels+2); hz=m2h(mm)
    bins=np.floor((n_fft+1)*hz/sr).astype(int)
    fb=np.zeros((n_mels,n_fft//2+1))
    for i in range(n_mels):
        l,r=bins[i],bins[i+2]
        if r<=l: continue
        for j in range(l,min(r,n_fft//2+1)): fb[i,j]=1.0/(r-l)
    mmag=np.expm1(np.clip(m,0,10))
    mag=fb.T@mmag
    angles=np.exp(2j*np.pi*np.random.rand(*mag.shape))
    S=mag*angles
    _,y=ss.istft(S,nperseg=n_fft,noverlap=n_fft-hop)
    y=y/max(np.abs(y).max(),1e-6)*0.9
    return y.astype(np.float32)
outdir=ROOT/"logs/audio"; outdir.mkdir(exist_ok=True,parents=True)
picks=[0,1,2,3]   # 前四个类别
with torch.no_grad():
    demo=[]
    for ci in picks:
        z=torch.randn(1,64)
        g=vae.decode(z,EM[ci:ci+1])[0].numpy()
        demo.append((words[ci],g))
        w=mel_to_wav(g)
        sf.write(str(outdir/f"gen_{ci}_{words[ci]}.wav"),w,22050)
# 同时导出真实样本对照
d=np.load(ROOT/"data/esc_mels.npz",allow_pickle=True)
cats=[str(c) for c in d["cats"]]
for ci in picks:
    idx=[i for i,c in enumerate(cats) if c==words[ci]][:1]
    if idx:
        sf.write(str(outdir/f"real_{ci}_{words[ci]}.wav"),mel_to_wav(d["mels"][idx[0]]),22050)
print("wav 已写到", outdir)
import os
print([f for f in sorted(os.listdir(outdir))][:10])
