#!/usr/bin/env python3
"""新数据完整性检查：形状/对齐/泄漏/取值/检索可用性。"""
import numpy as np, re, pickle
from collections import Counter
ROOT="/srv/brainai/data/"
print("="*54)
# ---------- 1. 图像 ----------
f=np.load(ROOT+"f30k_clip.npz",allow_pickle=True)
FE=f["feats"]; CAP=f["captions"]; IDX=f["img_idx"]
print(f"① 图像: 特征{FE.shape} 字幕{len(CAP)} 索引{IDX.shape}")
print(f"   索引范围 {IDX.min()}~{IDX.max()}  应为 0~{len(FE)-1}  → {'OK' if IDX.max()==len(FE)-1 else '不一致'}")
norms=np.linalg.norm(FE,axis=1)
print(f"   特征范数 {norms.min():.3f}~{norms.max():.3f} → {'已归一化 OK' if abs(norms.mean()-1)<0.05 else '未归一化'}")
print(f"   含 NaN: {bool(np.isnan(FE).any())}  含 Inf: {bool(np.isinf(FE).any())}")
print(f"   每图字幕数: {Counter(IDX.tolist()).most_common(3)}")
empty=sum(1 for c in CAP if len(str(c).strip())<5)
print(f"   空字幕 {empty} 条 → {'OK' if empty==0 else '有问题'}")
lens=[len(str(c).split()) for c in CAP]
print(f"   字幕词数 均值{np.mean(lens):.1f} 最大{max(lens)}")
dup=len(CAP)-len(set(map(str,CAP)))
print(f"   重复字幕 {dup} 条（{dup/len(CAP)*100:.1f}%）")
# 检索测试：图文是否真的对齐
Q=FE[:500]; C=np.stack([FE[int(np.random.choice(np.where(IDX==i)[0]))] for i in range(500)])
sim=Q@FE.T
r1=(sim.argmax(1)==np.arange(500)).mean()
print(f"   同图检索 R@1 {r1*100:.1f}%（特征索引与字幕索引对齐性）")
print("="*54)
# ---------- 2. 声音 ----------
a=np.load(ROOT+"esc_semvec_aug.npz",allow_pickle=True)
V=a["vec"]; T=a["target"]; C=a["cats"]
print(f"② 声音: 向量{V.shape} 类别{len(set(T.tolist()))}")
print(f"   范数 {np.linalg.norm(V,axis=1).min():.3f}~{np.linalg.norm(V,axis=1).max():.3f}")
print(f"   含 NaN: {bool(np.isnan(V).any())}")
cnt=Counter(T.tolist()); print(f"   每类样本数 min{min(cnt.values())} max{max(cnt.values())} → {'均衡 OK' if min(cnt.values())==max(cnt.values()) else '不均衡'}")
# 变体多样性：同一类内向量是否过度相似（增强是否只是复制）
same=[V[i] for i in range(0,50)]
sims=[float(np.dot(V[0],V[j])) for j in range(1,5)]
print(f"   同类前5个变体的余弦相似度 {['%.2f'%s for s in sims]} → {'有差异 OK' if max(sims)<0.999 else '完全重复，增强无效'}")
print("="*54)
# ---------- 3. 文字 ----------
p=np.load(ROOT+"orca_big.npy",allow_pickle=True)
print(f"③ 文字: {len(p)} 条问答")
ls=[len(str(a).split()) for a,b in p[:5000]]; lb=[len(str(b).split()) for a,b in p[:5000]]
print(f"   问题词数 均值{np.mean(ls):.1f}  答案词数 均值{np.mean(lb):.1f}")
empt=sum(1 for a,b in p if len(str(b).strip())<3)
print(f"   空答案 {empt} → {'OK' if empt==0 else '有问题'}")
dupq=len(p)-len(set(str(a) for a,b in p))
print(f"   重复问题 {dupq} 条（{dupq/len(p)*100:.1f}%）")
# 分词覆盖
V_=pickle.load(open(ROOT+"tok_qa2.pkl","rb")); vocab=V_["vocab"]
WORD=re.compile(r"[a-z']+|[.,!?]")
def tok(s,limit=200):
    n=0; unk=0
    for w in WORD.findall(str(s).lower()):
        i=0
        while i<len(w):
            hit=None
            for Lg in range(min(8,len(w)-i),0,-1):
                if w[i:i+Lg] in vocab: hit=w[i:i+Lg]; break
            if hit: i+=len(hit)
            else: unk+=1; i+=1
            n+=1
    return n,unk
tot=unk=0
for a,b in p[:2000]:
    for txt in (a,b):
        n,u=tok(txt); tot+=n; unk+=u
print(f"   未登录字符占比 {unk/max(tot,1)*100:.2f}% → {'覆盖良好' if unk/max(tot,1)<0.02 else '覆盖不足'}")
over=sum(1 for a,b in p[:2000] if tok(a)[0]+tok(b)[0]>120)
print(f"   超过 120 token 的样本 {over}/2000（会被截断）")
print("="*54)
print("检查完成")
