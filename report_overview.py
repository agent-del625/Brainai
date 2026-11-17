#!/usr/bin/env python3
"""能力总览：把已验证的指标和与真实 LLM 的差距画在一张图上。"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT=Path("/srv/brainai")
font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",16)
fontb=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",24)
fonts=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",13)
W,H=1100,760
img=Image.new("RGB",(W,H),"white"); dr=ImageDraw.Draw(img)
dr.text((24,18),"树莓派多模态模型 · 能力总览",font=fontb,fill=(20,20,20))
dr.text((24,52),"数据截至 2026-11-15 · 所有指标均按「图像/视频」划分，无泄漏",font=fonts,fill=(110,110,110))
caps=[
 ("看图说话","chrF 59.7%",0.60,"束搜索 + 判断重排"),
 ("识物（150 类主体）","77%",0.77,"显著性排序 + 候选约束"),
 ("看视频说话","chrF 39.4%",0.39,"16 帧即 token"),
 ("判断/重排能力","+3.7 点",0.35,"偏好判断器（on-policy 训练）"),
 ("问答（格式已学）","内容准确 54.6%",0.55,"40M + 23.6 万条指令"),
 ("语言模型","困惑度 3.7",0.78,"52.8 万句，13.6M 参数"),
 ("简笔画生成","chamfer 24 px",0.30,"逐笔自回归"),
 ("持续学习新 token","困惑度 3073→11.5",0.80,"只训 25% 参数"),
]
y=92
for name,val,ratio,note in caps:
    dr.text((24,y),name,font=font,fill=(30,30,30))
    bw=380; bx=300
    dr.rectangle([bx,y+4,bx+bw,y+22],fill=(238,238,238))
    col=(40,140,70) if ratio>=0.6 else ((200,150,40) if ratio>=0.35 else (200,60,60))
    dr.rectangle([bx,y+4,bx+int(bw*ratio),y+22],fill=col)
    dr.text((bx+bw+14,y+2),val,font=fonts,fill=(40,40,40))
    dr.text((bx,y+28),note,font=fonts,fill=(130,130,130))
    y+=72
# 差距区
dr.rectangle([24,y+6,W-24,y+120],outline=(200,60,60),width=2)
dr.text((36,y+14),"与真实大模型的差距（这才是重点）",font=font,fill=(200,60,60))
lines=[
 "参数：我们 1,400 万  vs  Llama-3 80 亿 —— 相差 570 倍",
 "预训练语料：1,000 万 token  vs  15 万亿 —— 相差 15,000 倍",
 "指令数据：23.6 万条  vs  1,000 万条 —— 相差 42 倍",
 "结论：格式能学，事实装不下；推理能力目前为零",
]
yy=y+42
for t in lines:
    dr.text((36,yy),t,font=fonts,fill=(70,70,70)); yy+=19
img.save(ROOT/"logs/overview.png")
print("图:",ROOT/"logs/overview.png")
