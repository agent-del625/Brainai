#!/usr/bin/env python3
"""描述生成 v9：一次修完——特征归一化 + 对比排序损失 + 缩小解码器 + 条件敏感度监控。"""
import re, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE,dim=-1)                      # ★修复 1：L2 归一化
ck=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in ck["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(ck["model"]); clf.eval()
for p in clf.parameters(): p.requires_grad=False
ACT=["running","jumping","sitting","standing","walking","playing","swimming","riding","holding","eating","driving","flying"]
GRD=["grass","sand","water","road","snow","beach","field","street"];SKY=["clouds","sky","sunny","night","trees","sunset"];CNT=["one","two","three","group","crowd"]
ALL=ACT+GRD+SKY+CNT
def lab(ws): return [1.0 if w in ws else 0.0 for w in ALL]
LA=torch.tensor([lab(set(re.findall(r"[a-z']+",c.lower()))) for c in CAPS])
class Attr(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(ALL)))
    def forward(self,x): return self.net(x)
at=Attr(); oa=torch.optim.AdamW(at.parameters(), lr=1e-3)
for s in range(1200):
    i=torch.randint(0,len(CAPS),(256,))
    l=F.binary_cross_entropy_with_logits(at(FE[IMG[i]]), LA[i]); oa.zero_grad(); l.backward(); oa.step()
at.eval()
for p in at.parameters(): p.requires_grad=False
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(clf(FE)), torch.sigmoid(at(FE))], -1)
chars=sorted({ch for c in CAPS for ch in c})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
L=40
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(c) for c in CAPS])
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):                     # ★修复 3：解码器缩小
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
    def forward(self,cond,toks):
        cp=self.cproj(cond).unsqueeze(1).expand(-1,toks.size(1),-1)
        out,_=self.gru(torch.cat([self.emb(toks),cp],-1),self.h0(cond))
        return self.out(out)
m=Cap(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
nimg=len(FE); perm=torch.randperm(nimg); ntr=int(nimg*0.9)
tr_img=set(perm[:ntr].tolist()); te_img=set(perm[ntr:].tolist())
tr=[i for i in range(len(CAPS)) if int(IMG[i]) in tr_img]
te=[i for i in range(len(CAPS)) if int(IMG[i]) in te_img]
print("训练对",len(tr),"测试对",len(te))
MARGIN=0.5
for step in range(1,7001):
    i=torch.tensor(np.random.choice(tr,96))
    cnd=COND[IMG[i]]
    drop=(torch.rand(cnd.size(0),1)<0.05).float(); cnd=cnd*(1-drop)
    logits=m(cnd, X[i][:,:-1])
    tgt=X[i][:,1:]
    ce=F.cross_entropy(logits.reshape(-1,len(vocab)), tgt.reshape(-1), ignore_index=0, reduction='none')
    ce=ce.view(tgt.shape)
    ms=(tgt!=0).float()
    per=(ce*ms).sum(1)/ms.sum(1)
    # ★修复 2：对比排序损失——配错图片必须更差
    wrong=torch.roll(cnd,1,0)
    logits_w=m(wrong, X[i][:,:-1])
    ce_w=F.cross_entropy(logits_w.reshape(-1,len(vocab)), tgt.reshape(-1), ignore_index=0, reduction='none').view(tgt.shape)
    per_w=(ce_w*ms).sum(1)/ms.sum(1)
    l_rank=F.relu(MARGIN+per-per_w).mean()
    loss=per.mean()+l_rank
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%500==0:
        with torch.no_grad():
            j=torch.tensor(np.random.choice(te,64))
            c2=COND[IMG[j]]; lg=m(c2,X[j][:,:-1]); t2=X[j][:,1:]
            a=F.cross_entropy(lg.reshape(-1,len(vocab)),t2.reshape(-1),ignore_index=0).item()
            lgw=m(torch.roll(c2,1,0),X[j][:,:-1])
            b=F.cross_entropy(lgw.reshape(-1,len(vocab)),t2.reshape(-1),ignore_index=0).item()
        print(f"step {step} ce {per.mean().item():.3f} rank {l_rank.item():.3f} | 正确 {a:.3f} 配错 {b:.3f} 敏感度 {b-a:+.3f}", flush=True)
def chrf(p,r,n=3):
    from collections import Counter
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp,cr=Counter(gp),Counter(gr); ov=sum((cp&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
@torch.no_grad()
def gen(cond,steps=40):
    h=m.h0(cond); tok=torch.tensor([[1]]); cp=m.cproj(cond); out=[]
    for _ in range(steps):
        o,hn=m.gru(torch.cat([m.emb(tok),cp.unsqueeze(1)],-1),h)
        nxt=m.out(o[:,-1]).argmax(-1,keepdim=True)
        if int(nxt)==2: break
        out.append(int(nxt)); tok=nxt; h=hn
    return "".join(inv.get(i,"") for i in out)
m.eval()
sel=te[:120]
sc=[chrf(gen(COND[IMG[i]:IMG[i]+1]), CAPS[i]) for i in sel]
outs={gen(COND[IMG[i]:IMG[i]+1]) for i in sel[:30]}
print(f"chrF 贪心 {np.mean(sc)*100:.1f}% | 30 张图产生 {len(outs)} 种不同句子")
for i in sel[:3]:
    print("  生成:",gen(COND[IMG[i]:IMG[i]+1])[:60]); print("  参考:",CAPS[i][:60])
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/cap10.pt")
