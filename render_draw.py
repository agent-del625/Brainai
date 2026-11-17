#!/usr/bin/env python3
"""渲染模型生成的笔画，与原图并排对比。"""
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
d=np.load("/tmp/draw_outs.npz",allow_pickle=True)
seqs=d["seqs"]; imgs=d["imgs"]; labels=d["labels"]
S=64; BIN=32; PEN=1024; E=0.35
def decode(toks):
    """连续路径解码：抬笔不清空位置，仍按顺序连成折线（QuickDraw 简化数据每笔点很少）。"""
    strokes=[]; cur=[]; x=y=0.5
    for t in toks:
        if t>=PEN:                  # 抬笔：另起一笔，但位置连续
            if len(cur)>=1: strokes.append(cur)
            cur=[(x,y)]
            continue
        qx,qy=divmod(int(t),BIN)
        dx=(qx/(BIN-1))*2*E-E; dy=(qy/(BIN-1))*2*E-E
        x+=dx; y+=dy
        cur.append((x,y))
    if len(cur)>=1: strokes.append(cur)
    return strokes
def render(strokes,S=64):
    img=np.zeros((S,S),np.uint8)
    for s in strokes:
        for i in range(len(s)-1):
            p1=(int(np.clip(s[i][0],0,1)*(S-1)),int(np.clip(s[i][1],0,1)*(S-1)))
            p2=(int(np.clip(s[i+1][0],0,1)*(S-1)),int(np.clip(s[i+1][1],0,1)*(S-1)))
            cv2.line(img,p1,p2,255,1)
    return img
def chamfer(a,b):
    ya,xa=np.nonzero(a>50); yb,xb=np.nonzero(b>50)
    if len(xa)==0 or len(xb)==0: return 99.0
    A=np.stack([xa,ya],1).astype(np.float32); B=np.stack([xb,yb],1).astype(np.float32)
    D=((A[:,None,:]-B[None,:,:])**2).sum(-1)
    return float(np.sqrt(D.min(1)).mean()+np.sqrt(D.min(0)).mean())
W,H=150,300
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",13)
tiles=[]
for i in range(12):
    st=decode(seqs[i]); gen=render(st)
    cd=chamfer(gen,imgs[i])
    tile=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(tile)
    tile.paste(Image.fromarray(imgs[i]).resize((128,128)).convert("RGB"),(11,6))
    tile.paste(Image.fromarray(gen).resize((128,128)).convert("RGB"),(11,140-0)) if False else None
    tiles.append((imgs[i],gen,cd))
sheet=Image.new("RGB",(W*4,(H)*3+40),"white"); dr=ImageDraw.Draw(sheet)
dr.text((10,10),"上排：原图  下排：模型逐笔生成（chamfer 越小越好）",font=font,fill=(0,0,0))
for k,(orig,gen,cd) in enumerate(tiles):
    cx=(k%4)*W; cy=40+(k//4)*H
    sheet.paste(Image.fromarray(orig).resize((128,128)).convert("RGB"),(cx+11,cy))
    sheet.paste(Image.fromarray(gen).resize((128,128)).convert("RGB"),(cx+11,cy+136))
    dr.text((cx+11,cy+270),f"chamfer {cd:.1f}",font=font,fill=(180,0,0))
sheet=Image.new("RGB",(W*4,H*3+44),"white"); dr=ImageDraw.Draw(sheet)
dr.text((10,10),"每组：上=原图  下=模型逐笔生成",font=font,fill=(0,0,0))
for k,(orig,gen,cd) in enumerate(tiles):
    cx=(k%4)*W; cy=44+(k//4)*H
    sheet.paste(Image.fromarray(orig).resize((128,128)).convert("RGB"),(cx+11,cy))
    sheet.paste(Image.fromarray(gen).resize((128,128)).convert("RGB"),(cx+11,cy+136))
    dr.text((cx+11,cy+272),f"chamfer {cd:.1f}",font=font,fill=(180,0,0))
sheet.save("/srv/brainai/logs/draw_cmp.png")
cds=[tiles[k][2] for k in range(len(tiles))]
print("chamfer 平均 %.2f px（越小越好）"%np.mean(cds))
print("图:", "/srv/brainai/logs/draw_cmp.png", sheet.size)
