"""卷积+膨胀卷积的多模态模型：不用 transformer、不做 token 序列注意力。

视频：3D 卷积逐层降采样（空间 64→8，时间 16→4），全局池化后得 z_v
文本：字符 Embedding + 一维膨胀卷积（1/2/4/8），掩码平均得 z_t
耦合：FiLM 门控（一路调制另一路）+ 拼接 MLP 得 z_j
输出：① 重建视频帧 ② 逐位置字符预测 ③ 跨模态对比（InfoNCE）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block3d(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm3d(cout),
        nn.GELU(),
    )


class VideoEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block3d(1, 24), nn.MaxPool3d((1, 2, 2)),      # T16 H64 → H32
            conv_block3d(24, 48), nn.MaxPool3d((2, 2, 2)),     # T8  H16
            conv_block3d(48, 96), nn.MaxPool3d((2, 2, 2)),     # T4  H8
            conv_block3d(96, 128),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(128, dim)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        if v.dim() == 5 and v.shape[1] == 1:      # (B,1,T,H,W) 已是卷积格式
            pass
        elif v.dim() == 5 and v.shape[2] == 1:    # (B,T,1,H,W) → 转成 (B,1,T,H,W)
            v = v.permute(0, 2, 1, 3, 4)
        h = self.net(v)
        return self.proj(self.pool(h).flatten(1))


class AudioEncoder(nn.Module):
    """梅尔频谱 (B,1,40,64) → 2D 卷积降采样 → z_a。"""

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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        mask = (t != 0).float().unsqueeze(1)          # (B,1,L)
        h = self.conv(self.emb(t).transpose(1, 2))    # (B,96,L)
        h = (h * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        return self.proj(h)


class Fusion(nn.Module):
    """三路 FiLM 门控：视频 / 音频 / 文本互相调制后拼接。"""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 6, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim),
        )

    def forward(self, zv: torch.Tensor, za: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
        gv = torch.sigmoid(self.gate(torch.cat([za, zt], -1)))
        ga = torch.sigmoid(self.gate(torch.cat([zv, zt], -1)))
        gt = torch.sigmoid(self.gate(torch.cat([zv, za], -1)))
        v, a, t = zv * gv, za * ga, zt * gt
        return self.mlp(torch.cat([v, a, t, v * a, a * t, v * t], dim=-1))


class FrameDecoder(nn.Module):
    """把联合潜变量还原成 (1,16,64,64) 的帧序列。"""

    def __init__(self, dim: int = 128, t: int = 16, h: int = 64):
        super().__init__()
        self.t, self.h = t, h
        self.fc = nn.Linear(dim, 96 * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose3d(96, 48, 4, stride=2, padding=1), nn.GELU(),    # 8
            nn.ConvTranspose3d(48, 24, 4, stride=2, padding=1), nn.GELU(),    # 16
            nn.ConvTranspose3d(24, 1, 4, stride=2, padding=1),                # 32
        )
        self.out_t = t

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 96, 4, 4, 4)
        y = self.net(h)                     # (B,1,64,64,64)
        y = y[:, :, : self.out_t]           # 取前 16 帧 → (B,1,T,64,64)
        return torch.sigmoid(y).permute(0, 2, 1, 3, 4)   # → (B,T,1,64,64)


class TextDecoder(nn.Module):
    """条件于联合潜变量，逐位置预测字符（卷积，无自回归）。"""

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
        h = self.emb(tokens).transpose(1, 2)          # (B,emb,L)
        scale, shift = self.film(z).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.net(h)                            # (B,V,L)


class BrainNet(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, t: int = 16, h: int = 64):
        super().__init__()
        self.video = VideoEncoder(dim)
        self.audio = AudioEncoder(dim)
        self.text = TextEncoder(vocab, dim)
        self.fuse = Fusion(dim)
        self.frame_head = FrameDecoder(dim, t, h)
        self.text_head = TextDecoder(vocab, dim)

    def forward(self, v: torch.Tensor, a: torch.Tensor, t_in: torch.Tensor) -> dict:
        zv = F.normalize(self.video(v), dim=-1)
        za = F.normalize(self.audio(a), dim=-1)
        zt = F.normalize(self.text(t_in), dim=-1)
        zj = self.fuse(zv, za, zt)
        return {
            "zv": zv, "za": za, "zt": zt,
            "frames": self.frame_head(zj),
            "chars": self.text_head(t_in, zj),
        }
