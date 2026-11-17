#!/usr/bin/env python3
"""渲染「词 → 简笔画」：每个概念 3 张，看模型是不是真会画。"""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
d=np.load("/tmp/sketch_txt_outs.npz",allow_pickle=True)
PEN=1024;BIN=32;E=0.35
def decode(toks,S=64):
    pts=[];x=y=0.5;cur=[(x,y)]
    for t in toks:
        t=int(t)
        if t>=PEN:
            if cur: pts.append(cur); cur=[(x,y)]
            continue
        qx,qy=divmod(t,BIN); x+=(qx/(BIN-1))*2*E-E; y+=(qy/(BIN-1))*2*E-E
        cur.append((x,y))
    if cur: pts.append(cur)
    img=np.zeros((S,S),np.uint8)
    for s in pts:
        for i in range(len(s)-1):
            p1=(int(np.clip(s[i][0],0,1)*(S-1)),int(np.clip(s[i][1],0,1)*(S-1)))
            p2=(int(np.clip(s[i+1][0],0,1)*(S-1)),int(np.clip(s[i+1][1],0,1)*(S-1)))
            cv2.line(img,p1,p2,255,1)
    return img
words=list(d.files)
W,Hh=120,146
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",15)
sheet=Image.new("RGB",(W*3*4,Hh*3+50),"white"); dr=ImageDraw.Draw(sheet)
dr.text((10,12),"词 → 简笔画（每行一个概念，三列是不同的画法）",font=font,fill=(0,0,0))
for wi,w in enumerate(words[:11]):
    r,c=divmod(wi,4)
    for k in range(3):
        img=decode(d[w][k])
        sheet.paste(Image.fromarray(img).resize((110,110)).convert("RGB"),(c*W*3//1+k*W//1+5, 46+r*Hh))
    dr.text((c*W*3+8,46+r*Hh+112),w,font=font,fill=(180,0,0))
sheet.save(ROOT/"logs/sketch_txt.png")
print("图:",ROOT/"logs/sketch_txt.png",sheet.size)
