#!/usr/bin/env python3
"""token 流水线 v2：字兜底 + 重要性注意力掩码 + 受限输出词表。"""
import sys, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter, defaultdict
sys.path.insert(0,"/srv/brainai")
from tok_induct import Inducer
from pathlib import Path
ROOT=Path("/srv/brainai")

def build(sents, min_cov=6, max_len=10, min_freq=5):
    ind=Inducer(min_cov=min_cov, max_len=max_len); cov=ind.run(sents)
    items=sorted(cov.items(), key=lambda kv:(-kv[1], -len(kv[0])))
    bylen=defaultdict(list)
    for t,c in items: bylen[len(t)].append((t,c))
    def absorbed(t,c):
        for L in range(len(t)+1, max_len+1):
            for u,cu in bylen.get(L,[]):
                if t in u and cu>=c: return True
        return False
    keep=[(t,c) for t,c in items if not absorbed(t,c) and c>=min_freq and len(t)>=2]
    chars=Counter()
    for s in sents:
        for ch in s: chars[ch]+=1
    print(f"  候选 {len(cov)}  独立多字 token {len(keep)}  单字兜底 {len(chars)}")
    mx=max([c for _,c in keep]+[chars.most_common(1)[0][1]])
    vocab={"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
    imp=[0.0,0.0,0.0,0.0]
    for t,c in sorted(keep,key=lambda kv:-kv[1]):
        if t in vocab: continue
        vocab[t]=len(vocab); imp.append(c/mx)
    for ch,c in chars.most_common():
        if ch in vocab: continue
        vocab[ch]=len(vocab); imp.append(0.35*c/mx)      # 单字重要性打折
    long2short=sorted([t for t,_ in keep], key=len, reverse=True)
    def tok(s):
        out=[];i=0
        while i<len(s):
            hit=None
            for L in range(min(max_len,len(s)-i),1,-1):
                if s[i:i+L] in vocab: hit=s[i:i+L]; break
            if hit: out.append(hit); i+=len(hit)
            else: out.append(s[i]); i+=1
        return out
    seqs=[tok(s) for s in sents]
    unk=sum(1 for t in seqs for x in t if x not in vocab)
    print(f"  平均 {(sum(len(t) for t in seqs)/len(seqs)):.1f} token/句（字符 {sum(len(s) for s in sents)/len(sents):.1f}）  OOV {unk}")
    return vocab, np.array(imp,np.float32), seqs

class Block(nn.Module):
    """自注意力：重要性作为 key 侧的加性权重（低重要性 token 贡献更小）。"""
    def __init__(self,d,h):
        super().__init__()
        self.h=h; self.dh=d//h
        self.qkv=nn.Linear(d,d*3); self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d))
        self.ln1=nn.LayerNorm(d); self.ln2=nn.LayerNorm(d)
    def forward(self,x,imp):
        B,T,D=x.shape
        q,k,v=self.qkv(self.ln1(x)).chunk(3,-1)
        q=q.view(B,T,self.h,self.dh).transpose(1,2)
        k=k.view(B,T,self.h,self.dh).transpose(1,2)
        v=v.view(B,T,self.h,self.dh).transpose(1,2)
        att=(q@k.transpose(-1,-2))/math.sqrt(self.dh)          # (B,h,T,T)
        causal=torch.triu(torch.full((T,T),float("-inf")),1)
        bias=causal[None,None]+torch.log(imp.clamp(min=1e-3))[:,None,None,:]   # ★重要性进注意力
        att=att+bias
        att=att.softmax(-1)
        o=(att@v).transpose(1,2).reshape(B,T,D)
        x=x+self.proj(o); x=x+self.ff(self.ln2(x))
        return x

class TokLM(nn.Module):
    def __init__(self,Vn,d=128,h=4,layers=2,imp0=None,L=18,out_cut=0):
        super().__init__()
        self.tok=nn.Embedding(Vn,d); self.pos=nn.Embedding(L+1,d)
        self.imp=nn.Parameter(torch.tensor(imp0 if imp0 is not None else np.ones(Vn,np.float32)))
        self.blocks=nn.ModuleList([Block(d,h) for _ in range(layers)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,Vn,bias=False); self.head.weight=self.tok.weight
        self.out_cut=out_cut
    def forward(self,x):
        B,T=x.shape
        h=self.tok(x)+self.pos(torch.arange(T)[None])
        imp=self.imp[x].clamp(min=1e-3)
        for b in self.blocks: h=b(h,imp)
        logits=self.head(self.ln(h))+self.imp.unsqueeze(0).unsqueeze(0)
        if self.out_cut>0:
            top=torch.topk(self.imp,self.out_cut).indices
            return logits.index_select(-1,top), top
        return logits, None

def train(vocab, imp, seqs, tag, steps=3000, out_cut=0, L=18):
    i2t={v:k for k,v in vocab.items()}
    def enc(ts):
        ids=[1]+[vocab.get(t,3) for t in ts][:L-1]+[2]; return ids+[0]*(L+1-len(ids))
    X=torch.tensor([enc(s) for s in seqs])
    n=len(X); perm=torch.randperm(n); ntr=int(n*0.9); tr,te=perm[:ntr],perm[ntr:]
    m=TokLM(len(vocab),imp0=imp,L=L,out_cut=out_cut); opt=torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(1,steps+1):
        i=tr[torch.randint(0,len(tr),(64,))]
        lg,top=m(X[i][:,:-1])
        tgt=X[i][:,1:]
        if top is not None:
            keep=(tgt.unsqueeze(-1)==top.unsqueeze(0).unsqueeze(0)).any(-1)
            tgt2=torch.where(keep, tgt, torch.full_like(tgt,3))
            loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), tgt2.reshape(-1), ignore_index=0)
        else:
            loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)), tgt.reshape(-1), ignore_index=0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step%1000==0: print(f"  [{tag}] step {step} loss {loss.item():.3f} ppl {math.exp(min(loss.item(),15)):.1f}", flush=True)
    m.eval()
    with torch.no_grad():
        i=te[:300]; lg,_=m(X[i][:,:-1])
        if out_cut>0:
            top=torch.topk(m.imp,out_cut).indices
            tgt=X[i][:,1:]; keep=(tgt.unsqueeze(-1)==top.unsqueeze(0).unsqueeze(0)).any(-1)
            ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), torch.where(keep,tgt,torch.full_like(tgt,3)).reshape(-1), ignore_index=0).item()
        else:
            ll=F.cross_entropy(lg.reshape(-1,lg.size(-1)), X[i][:,1:].reshape(-1), ignore_index=0).item()
    print(f"  [{tag}] 测试困惑度 {math.exp(min(ll,15)):.1f}")
    return m,i2t
if __name__=="__main__":
    d=np.load(ROOT/"data/zh_all.npz",allow_pickle=True); ZH=[str(t) for t in d["texts"]]
    print("== 中文合成语料",len(ZH),"句"); v,i,s=build(ZH)
    m,i2t=train(v,i,s,"zh",steps=2000)
    @torch.no_grad()
    def gen(seed,temp=0.9,L=18):
        torch.manual_seed(seed); ids=[1]
        for _ in range(L):
            x=torch.tensor([ids])[:, -(L+1):]; lg,_=m(x); lg=lg[:,-1][0]
            val,idx=torch.topk(lg,20); nxt=int(idx[torch.multinomial(torch.softmax(val/temp,-1),1)])
            if nxt==2: break
            ids.append(nxt)
        return "".join(i2t.get(x,"?") for x in ids[1:])
    print("  生成:")
    for k in range(5): print("   ",gen(k))
    np.savez_compressed(ROOT/"data/tok_v2_zh.npz", seqs=np.array(["|".join(t) for t in s]))
    torch.save({"model":m.state_dict(),"vocab":v}, ROOT/"ckpt/toklm_zh.pt")
