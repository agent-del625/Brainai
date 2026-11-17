#!/usr/bin/env python3
"""多模态大脑演示：一张照片 → 英文描述 + 中文长句 + 概念 + 简笔画 + 声音。"""
import re, glob, wave, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from audio_feat import mel_to_wav
ROOT=Path("/srv/brainai")

d=np.load(ROOT/"data/f8k_clip5.npz",allow_pickle=True)
FE0=torch.from_numpy(d["feats"]).float(); CAPS=[str(c) for c in d["captions"]]; IMG=torch.from_numpy(d["img_idx"])
FE=F.normalize(FE0,dim=-1)
cc=torch.load(ROOT/"ckpt/concept_clf.pt",map_location="cpu",weights_only=False)
cnames=[str(c) for c in cc["names"]]
class Clf(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,384),nn.GELU(),nn.LayerNorm(384),nn.Dropout(0.1),
                               nn.Linear(384,192),nn.GELU(),nn.Linear(192,len(cnames)))
    def forward(self,x): return self.net(x)
clf=Clf(); clf.load_state_dict(cc["model"]); clf.eval()
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
with torch.no_grad():
    COND=torch.cat([FE, torch.sigmoid(clf(FE)), torch.sigmoid(at(FE))], -1)
ck=torch.load(ROOT/"ckpt/cap10.pt",map_location="cpu",weights_only=False)
vocab=ck["vocab"]; inv={v:k for k,v in vocab.items()}
class Cap(nn.Module):
    def __init__(self,h=256,emb=96):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(COND.shape[1],384),nn.GELU(),nn.LayerNorm(384),nn.Linear(384,h),nn.GELU())
        self.emb=nn.Embedding(len(vocab),emb); self.cproj=nn.Linear(COND.shape[1],h)
        self.gru=nn.GRU(emb+h,h,batch_first=True); self.init=nn.Linear(h,h); self.out=nn.Linear(h,len(vocab))
    def h0(self,c): return self.init(self.enc(c)).unsqueeze(0)
cap=Cap(); cap.load_state_dict(ck["model"]); cap.eval()
@torch.no_grad()
def gen_caption(cond,seed=0,temp=0.7,k=20,steps=40):
    torch.manual_seed(seed); h=cap.h0(cond); tok=torch.tensor([[1]]); cp=cap.cproj(cond); out=[]
    for _ in range(steps):
        o,hn=cap.gru(torch.cat([cap.emb(tok),cp.unsqueeze(1)],-1),h)
        v,i=torch.topk(cap.out(o[:,-1])[0],k)
        nxt=int(i[torch.multinomial(torch.softmax(v/temp,-1),1)])
        if nxt==2: break
        out.append(nxt); tok=torch.tensor([[nxt]]); h=hn
    return "".join(inv.get(x,"") for x in out)

# 简笔画：CLIP → 积木
pk=torch.load(ROOT/"ckpt/p2s.pt",map_location="cpu",weights_only=False)
cent=torch.as_tensor(pk["cent"]).float(); N,G,RES,SB,RB,K=[int(x) for x in pk["cfg"]]
class P2S(nn.Module):
    def __init__(self,h=384):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,h),nn.GELU())
        self.q=nn.Parameter(torch.randn(N,h)*0.02)
        self.h_id=nn.Linear(h,K); self.h_cell=nn.Linear(h,G*G); self.h_res=nn.Linear(h,RES*RES)
        self.h_sc=nn.Linear(h,SB); self.h_rot=nn.Linear(h,RB)
    def forward(self,f):
        h=self.net(f).unsqueeze(1)+self.q.unsqueeze(0)
        return self.h_id(h),self.h_cell(h),self.h_res(h),self.h_sc(h),self.h_rot(h)
p2s=P2S(); p2s.load_state_dict(pk["model"]); p2s.eval()
def draw_sketch(f,seed=0,size=200):
    torch.manual_seed(seed)
    with torch.no_grad():
        outs=p2s(f); probs=[torch.softmax(o[0],-1) for o in outs]
    img=np.zeros((size,size),np.uint8)
    for t in range(N):
        pid,cell,res,sb,rb=[int(torch.multinomial(probs[j][t],1)) for j in range(5)]
        gx,gy=divmod(cell,G); rx,ry=divmod(res,RES)
        cx=(gx+(rx+0.5)/RES)/G*2-1; cy=(gy+(ry+0.5)/RES)/G*2-1
        half=(sb+0.5)/SB*0.6; th=(rb+0.5)/RB*2*np.pi-np.pi
        c_,s_=np.cos(th),np.sin(th); sh=cent[pid].numpy()
        sh=np.stack([sh[:,0]*c_-sh[:,1]*s_, sh[:,0]*s_+sh[:,1]*c_],1)*half+np.array([cx,cy])
        q=np.stack([(sh[:,0]+1)*size/2-0.5,(sh[:,1]+1)*size/2-0.5],1)
        for a,b in zip(q,q[1:]): cv2.line(img,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),255,2)
    return 255-img

# 声音：概念 → ESC-50 原型
ep=np.load(ROOT/"data/esc50_proto.npz",allow_pickle=True); ecats=[str(c) for c in ep["cats"]]; emels=ep["mels"]
SND={"dog":"dog","cat":"cat","train":"train","airplane":"airplane","beach":"sea_waves","ocean":"sea_waves",
     "river":"water_drops","rain":"rain","forest":"crickets","chirping_birds":"chirping_birds","crickets":"crickets",
     "cow":"cow","sheep":"sheep","hen":"hen","pig":"pig","frog":"frog","rooster":"rooster","insects":"insects","waterfall":"pouring_water"}
def sound_for(concept):
    k=SND.get(concept)
    if k is None or k not in ecats: return None
    return mel_to_wav(emels[ecats.index(k)][0])

CN={"dog":"一只狗","cat":"一只猫","bird":"一只鸟","train":"一列火车","airplane":"一架飞机","beach":"海滩",
    "ocean":"大海","river":"河流","waterfall":"瀑布","forest":"森林","cow":"一头牛","sheep":"一只羊",
    "hen":"一只母鸡","pig":"一头猪","frog":"一只青蛙","rooster":"一只公鸡","insects":"一些昆虫"}
CN_ACT={"running":"奔跑","jumping":"跳跃","sitting":"坐着","standing":"站着","walking":"行走","playing":"玩耍",
        "swimming":"游泳","riding":"骑行","holding":"拿着东西","eating":"吃东西","driving":"行驶","flying":"飞行"}
CN_GRD={"grass":"绿色的草坪","sand":"沙地","water":"水边","road":"道路","snow":"雪地","beach":"海滩","field":"田野","street":"街道"}
CN_SKY={"clouds":"蓝天白云","sky":"开阔的天空","sunny":"阳光明媚","night":"夜色","trees":"树木","sunset":"夕阳"}

raw=[]
for f in sorted(glob.glob(str(ROOT/"data/downloads/f8k_*.parquet"))):
    raw.extend(pq.read_table(f).to_pydict()["image"])
n=len(FE); perm=torch.randperm(n); te=perm[int(n*0.9):].tolist()[:6]

font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",15)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",21)
W,Hh=380,560
tiles=[]
OUT=ROOT/"logs/demo"; OUT.mkdir(parents=True,exist_ok=True)
for idx,i in enumerate(te):
    f=FE0[i:i+1]
    with torch.no_grad():
        pc=torch.sigmoid(clf(f))[0]; pa=torch.sigmoid(at(f))[0]
    concept=cnames[int(pc.argmax())]
    en=gen_caption(COND[i:i+1],seed=idx)
    sk=draw_sketch(f,seed=idx)
    wav=sound_for(concept)
    if wav is not None:
        with wave.open(str(OUT/f"photo{idx}_{concept}.wav"),"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(wav.tobytes())
    def best(mapping,thr=0.10):
        for j in pa.argsort(descending=True).tolist():
            if ALL[j] in mapping and pa[j]>thr: return mapping[ALL[j]]
        return None
    cn=f"{CN.get(concept,concept)}"
    a=best(CN_ACT); g=best(CN_GRD); s=best(CN_SKY)
    cn_sent=f"照片里{cn}"
    if a: cn_sent+=f"正在{a}"
    if g: cn_sent+=f"，位于{g}上"
    if s: cn_sent+=f"，背景是{s}"
    cn_sent+="。"

    arr=cv2.imdecode(np.frombuffer(raw[i]["bytes"],np.uint8),cv2.IMREAD_COLOR)
    im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((180,140))
    skv=Image.fromarray(sk).resize((180,140))
    canvas=Image.new("RGB",(W,Hh),"white"); dr=ImageDraw.Draw(canvas)
    canvas.paste(im,(10,10)); canvas.paste(skv,(190,10))
    y=162
    dr.text((10,y),"概念："+concept,font=font,fill=(0,110,0)); y+=24
    dr.text((10,y),"英文：",font=font,fill=(180,0,0)); y+=20
    for line in [en[k:k+42] for k in range(0,len(en),42)][:2]:
        dr.text((10,y),line,font=font,fill=(0,0,0)); y+=20
    y+=6
    dr.text((10,y),"中文：",font=font,fill=(180,0,0)); y+=20
    for line in [cn_sent[k:k+22] for k in range(0,len(cn_sent),22)][:3]:
        dr.text((10,y),line,font=font,fill=(0,0,0)); y+=20
    y+=6
    dr.text((10,y),f"声音：{SND.get(concept,'(无对应)')}",font=font,fill=(0,0,150)); y+=20
    dr.text((10,y),"左上=真实照片  右上=模型画的",font=font,fill=(120,120,120))
    tiles.append(canvas)
cols=3; rows=(len(tiles)+cols-1)//cols
sheet=Image.new("RGB",(W*cols,Hh*rows+60),"white"); dr=ImageDraw.Draw(sheet)
dr.text((16,14),"多模态大脑：一张照片 → 说（英/中）+ 认（概念）+ 画（简笔画）+ 出声",font=fontb,fill=(20,20,20))
for k,t in enumerate(tiles): sheet.paste(t,((k%cols)*W,60+(k//cols)*Hh))
sheet.save(str(ROOT/"logs/demo/sheet.png"))
print("图:", ROOT/"logs/demo/sheet.png", sheet.size)
print("音频:", sorted([p.name for p in OUT.glob("*.wav")]))
