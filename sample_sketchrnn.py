#!/usr/bin/env python3
"""从 Sketch-RNN 采样：给定类别，一步步画出简笔画。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
from pathlib import Path
ROOT = Path("/srv/brainai"); M, H, S = 20, 384, 64

class MDN(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.lstm = nn.LSTM(5+64, H, num_layers=2, batch_first=True)
        self.cls = nn.Embedding(ncls, 64)
        self.z = nn.Linear(H, M*6); self.pen = nn.Linear(H, 3)
    def forward(self, x, c):
        cc = self.cls(c).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.lstm(torch.cat([x, cc], -1))
        z = self.z(h); pen = self.pen(h)
        pi, mu1, mu2, s1, s2, rho = z.split(M, -1)
        return (F.log_softmax(pi,-1), mu1, mu2, torch.exp(s1), torch.exp(s2), torch.tanh(rho), pen)

ck = torch.load(ROOT/"ckpt/sketchrnn.pt", map_location="cpu", weights_only=False)
cats = [str(c) for c in ck["cats"]]; m = MDN(len(cats)); m.load_state_dict(ck["model"]); m.eval()

def sample(cid, steps=100, temp=0.5):
    canvas = np.zeros((S,S), np.float32)
    x = torch.zeros(1,1,5); px = py = 0.0
    pen_down = False
    with torch.no_grad():
        for t in range(steps):
            pi, mu1, mu2, s1, s2, rho, pen = m(x, torch.tensor([cid]))
            pi = pi[0,0]; k = int(torch.multinomial(torch.exp(pi/temp), 1))
            dx = float(mu1[0,0,k] + s1[0,0,k]*torch.randn(()))
            dy = float(mu2[0,0,k] + s2[0,0,k]*torch.randn(()))
            p3 = int(pen[0,0].argmax())
            x = torch.tensor([[[dx, dy, 1.0 if p3==0 else 0.0, 1.0 if p3==1 else 0.0, 0.0]]])
            nx, ny = px + dx/10, py + dy/10
            if pen_down and p3 == 0:
                cv2.line(canvas,(int((px+1)*S/2),int((py+1)*S/2)),
                                (int((nx+1)*S/2),int((ny+1)*S/2)),1.0,1)
            pen_down = (p3 == 0)
            px, py = nx, ny
            if p3 == 2: break
    return canvas

OUT = ROOT/"logs/turtle"; OUT.mkdir(parents=True, exist_ok=True)
tiles = []
for c in cats:
    im = sample(cats.index(c), 120, temp=0.4)
    cv2.imwrite(str(OUT/f"srn_{c}.png"), (255*(1-im)).astype(np.uint8))
    tiles.append((255*(1-im)).astype(np.uint8))
cv2.imwrite(str(OUT/"sketchrnn_samples.png"), np.hstack(tiles))
print("图:", OUT/"sketchrnn_samples.png")
