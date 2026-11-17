#!/usr/bin/env python3
"""中文描述生成：CLIP → 中文字符级解码器（多样化句子）。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load(ROOT/"data/zh_all.npz",allow_pickle=True)
TEXT=[str(t) for t in d["texts"]]; FEAT=torch.from_numpy(d["feats"]).float(); SP=d["split"]; IMG=d["img"]
FE=torch.from_numpy(d["feats"]).float()
FE=F.normalize(FE,dim=-1)
chars=sorted({ch for t in TEXT for ch in t})
vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
for ch in chars: vocab[ch]=len(vocab)
inv={v:k for k,v in vocab.items()}
L=26
def enc(s):
    ids=[1]+[vocab.get(ch,3) for ch in s][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
X=torch.tensor([enc(t) for t in TEXT])
SPt=torch.from_numpy(SP).bool()
print("中文样本",len(TEXT),"训练",int(SPt.sum()),"测试",int((~SPt).sum()),"字符表",len(vocab))
class Zh(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(512,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(512,h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
    def forward(self,c,t):
        cp=self.cproj(c).unsqueeze(1).expand(-1,t.size(1),-1)
        o,_=self.gru(torch.cat([self.emb(t),cp],-1),self.h0(c))
        return self.out(o)
m=Zh(); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
tr=torch.nonzero(SPt).flatten()
for step in range(1,6001):
    i=tr[torch.randint(0,len(tr),(96,))]
    c=FE[i]; xt=X[i]
    logits=m(c,xt[:,:-1])
    ce=F.cross_entropy(logits.reshape(-1,len(vocab)), xt[:,1:].reshape(-1), ignore_index=0, reduction='none').view(xt[:,1:].shape)
    loss=ce.mean()
    # 对比排序：配错图片必须更差
    cw=torch.roll(c,1,0)
    lg_w=m(cw,xt[:,:-1])
    ce_w=F.cross_entropy(lg_w.reshape(-1,len(vocab)), xt[:,1:].reshape(-1), ignore_index=0, reduction='none').view(ce.shape)
    loss=loss+F.relu(0.5+ce.mean(1)-ce_w.mean(1)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    if step%500==0: print(f"step {step} loss {loss.item():.3f}", flush=True)
@torch.no_grad()
def gen(c,seed=0,temp=0.8,k=20):
    torch.manual_seed(seed); h=m.h0(c); tok=torch.tensor([[1]]); cp=m.cproj(c); out=[]
    for _ in range(L):
        o,hn=m.gru(torch.cat([m.emb(tok),cp.unsqueeze(1)],-1),h)
        v,i=torch.topk(m.out(o[:,-1])[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)
def chrf(p,r,n=2):
    gp=[p[i:i+n] for i in range(len(p)-n+1)]; gr=[r[i:i+n] for i in range(len(r)-n+1)]
    if not gp or not gr: return 0.0
    cp_,cr=Counter(gp),Counter(gr); ov=sum((cp_&cr).values())
    if ov==0: return 0.0
    P=ov/len(gp); R=ov/len(gr); return 2*P*R/(P+R)
m.eval()
te=torch.nonzero(~SPt).flatten().tolist()
outs=[gen(FE[i:i+1],seed=j) for j,i in enumerate(te[:80])]
sc=np.mean([chrf(o,TEXT[i]) for o,i in zip(outs,te[:80])])
print(f"中文 chrF(2) {sc*100:.1f}%   80 张图产生 {len(set(outs))} 种句子")
for j in range(5):
    print("  生成:",outs[j]); print("  参考:",TEXT[te[j]])
torch.save({"model":m.state_dict(),"vocab":vocab}, ROOT/"ckpt/zh_all.pt")
