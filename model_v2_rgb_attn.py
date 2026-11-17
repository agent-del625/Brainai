"""卷积 + 跨模态注意力的多模态模型（不用 transformer 主干）。

视频 (RGB) → 3D 卷积堆栈 → 时空 token 序列 (N=256, C=128)
文本      → 字符 Embedding + 膨胀卷积 → 字符特征 (L,128)
跨模态注意力：Q=文本, K/V=视频 token  → 让"河流"这类词去画面里找对应区域
音频      → 2D 卷积（梅尔谱）→ z_a
融合      → 三路 FiLM 门控
输出      → 双解码分支（winner-takes-all）+ 字符预测
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3d_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.GELU(),
    )


class VideoEncoder(nn.Module):
    """RGB 视频 → 时空 token 序列。"""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv3d_block(3, 24), nn.MaxPool3d((1, 2, 2)),      # T16 H32
            conv3d_block(24, 48), nn.MaxPool3d((2, 2, 2)),     # T8  H16
            conv3d_block(48, 96), nn.MaxPool3d((2, 2, 2)),     # T4  H8
            conv3d_block(96, dim),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B,3,T,H,W) → tokens (B, N, dim)。"""
        h = self.net(v)                               # (B,dim,4,8,8)
        b, c = h.shape[:2]
        return h.reshape(b, c, -1).transpose(1, 2)    # (B,256,dim)


class AudioEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(96, dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.net(a)).flatten(1))


class TextEncoder(nn.Module):
    """返回 token 级特征（供注意力用）和池化表示。"""

    def __init__(self, vocab: int, dim: int = 128, emb: int = 48):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 96, 5, padding=2), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=8, dilation=4), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=16, dilation=8), nn.GELU(),
        )
        self.proj = nn.Linear(96, dim)
        self.tproj = nn.Linear(96, dim)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (t != 0).float()
        h = self.conv(self.emb(t).transpose(1, 2)).transpose(1, 2)   # (B,L,96)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.tproj(h), self.proj(pooled)


class CrossModalAttention(nn.Module):
    """Q 来自文本，K/V 来自视频：让"河流"这类词去画面里找对应区域。"""

    def __init__(self, dim: int = 128, heads: int = 4, n_tokens: int = 256):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)
        self.heads = heads
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_tokens: torch.Tensor, video_tokens: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        b, l, d = text_tokens.shape
        kv_in = video_tokens + self.pos[:, : video_tokens.size(1)]
        k, v = self.kv(kv_in).chunk(2, dim=-1)
        q = self.q(text_tokens)
        h = self.heads
        q = q.view(b, l, h, d // h).transpose(1, 2)
        k = k.view(b, -1, h, d // h).transpose(1, 2)
        v = v.view(b, -1, h, d // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (d // h) ** 0.5
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, l, d)
        o = self.out(o)
        w = text_mask.unsqueeze(-1)
        pooled = (o * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.norm(pooled)


class Fusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 6, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )

    def forward(self, zv, za, zt) -> torch.Tensor:
        gv = torch.sigmoid(self.gate(torch.cat([za, zt], -1)))
        ga = torch.sigmoid(self.gate(torch.cat([zv, zt], -1)))
        gt = torch.sigmoid(self.gate(torch.cat([zv, za], -1)))
        v, a, t = zv * gv, za * ga, zt * gt
        return self.mlp(torch.cat([v, a, t, v * a, a * t, v * t], dim=-1))


class FrameDecoder(nn.Module):
    """一条解码分支：z → (B,T,3,32,32)。"""

    def __init__(self, dim: int = 128, t: int = 16):
        super().__init__()
        self.t = t
        self.fc = nn.Linear(dim, 96 * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose3d(24, 3, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 96, 4, 4, 4)
        y = self.net(h)[:, :, : self.t]                    # (B,3,T,32,32)
        return torch.sigmoid(y).permute(0, 2, 1, 3, 4)     # (B,T,3,32,32)


class TextDecoder(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, emb: int = 48):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.film = nn.Linear(dim, emb * 2)
        self.net = nn.Sequential(
            nn.Conv1d(emb, 96, 5, padding=2), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(96, vocab, 1),
        )

    def forward(self, tokens: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.emb(tokens).transpose(1, 2)
        scale, shift = self.film(z).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.net(h)


class BrainNet(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, t: int = 16, branches: int = 2):
        super().__init__()
        self.video = VideoEncoder(dim)
        self.audio = AudioEncoder(dim)
        self.text = TextEncoder(vocab, dim)
        self.xattn = CrossModalAttention(dim)
        self.fuse = Fusion(dim)
        self.frame_heads = nn.ModuleList([FrameDecoder(dim, t) for _ in range(branches)])
        self.text_head = TextDecoder(vocab, dim)

    def forward(self, v: torch.Tensor, a: torch.Tensor, t_in: torch.Tensor) -> dict:
        vtok = self.video(v)                      # (B,N,dim)
        zv = F.normalize(vtok.mean(1), dim=-1)
        za = F.normalize(self.audio(a), dim=-1)
        tmask = (t_in != 0).float()
        ttok, zt = self.text(t_in)
        zt = F.normalize(zt, dim=-1)
        zx = self.xattn(ttok, vtok, tmask)        # 文本注意力对齐到视频
        zj = self.fuse(zv, za, F.normalize(zt + zx, dim=-1))
        return {
            "zv": zv, "za": za, "zt": zt,
            "frames": [h(zj) for h in self.frame_heads],
            "chars": self.text_head(t_in, zj),
        }
