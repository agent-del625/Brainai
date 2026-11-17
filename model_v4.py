"""BrainAI v4：全局工作空间 + 模态专用编解码 + 自评头。

与 v3 的关键区别：
- 感知端用注意力池化（不再平均池化，避免条件向量塌缩）
- 跨模态融合用「工作空间」：4 个槽位 × N 轮交叉注意力迭代
- 文本解码器只条件于工作空间（看不到文本编码，杜绝抄答案）
- 额外的自评头：预测自己的误差，给训练和推理反馈
- 路由器：决定这次输出哪种模态
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """可学习 query 的注意力池化：把 token 序列压成一个有区分度的向量。"""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q.expand(tokens.size(0), -1, -1)              # (B,1,D)
        att = (q @ self.proj(tokens).transpose(1, 2)) / tokens.size(-1) ** 0.5
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(1).bool(), -1e4)
        w = att.softmax(-1)
        return self.norm((w @ tokens).squeeze(1))


class Conv3dBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm3d(cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class VideoEncoder(nn.Module):
    """(B,3,T,64,64) → tokens (B,256,D) + 跳连特征。"""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            Conv3dBlock(3, 24), nn.MaxPool3d((1, 2, 2)),
            Conv3dBlock(24, 48), nn.MaxPool3d((2, 2, 2)),
            Conv3dBlock(48, 96), nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(96, dim, 3, padding=1),
        )
        self.pool = AttnPool(dim)

    def forward(self, v):
        x = self.net[0](v)
        skip = F.adaptive_avg_pool3d(x, (4, 4, 4))
        h = self.net[1:](x)
        b, c = h.shape[:2]
        tok = h.reshape(b, c, -1).transpose(1, 2)
        return tok, self.pool(tok), skip


class AudioEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, dim, 3, padding=1),
        )
        self.pool = AttnPool(dim)

    def forward(self, a):
        h = self.net(a)                       # (B,D,5,8)
        b, c = h.shape[:2]
        tok = h.reshape(b, c, -1).transpose(1, 2)
        return tok, self.pool(tok)


class TextEncoder(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, emb: int = 48):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb, 96, 5, padding=2), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=8, dilation=4), nn.GELU(),
        )
        self.proj = nn.Conv1d(96, dim, 1)
        self.pool = AttnPool(dim)

    def encode_emb(self, emb_seq: torch.Tensor) -> torch.Tensor:
        """emb_seq: (B,emb,L) 连续向量 → 句向量（用于"反向思考"）。"""
        h = self.conv(emb_seq)
        tok = self.proj(h).transpose(1, 2)
        return self.pool(tok)

    def forward(self, t):
        mask = (t != 0)
        h = self.conv(self.emb(t).transpose(1, 2))
        tok = self.proj(h).transpose(1, 2)     # (B,L,D)
        return tok, self.pool(tok, mask), mask


class GlobalWorkspace(nn.Module):
    """4 个槽位，N 轮迭代：每轮先从所有模态读，再槽位间交流。"""

    def __init__(self, dim: int = 128, slots: int = 4, rounds: int = 12, heads: int = 4):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, slots, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.rounds = rounds

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.size(0)
        s = self.slots.expand(b, -1, -1)
        for _ in range(self.rounds):
            s = self.norm1(s + self.cross(s, tokens, tokens, need_weights=False)[0])
            s = self.norm2(s + self.self_attn(s, s, s, need_weights=False)[0])
            s = self.norm3(s + self.ff(s))
        return s


class TextDecoder(nn.Module):
    """因果卷积语言头：条件于工作空间状态。"""

    def __init__(self, vocab: int, dim: int = 128, emb: int = 64, ch: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.inp = nn.Conv1d(emb + dim, ch, 1)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv1d(ch, vocab, 1),
        )

    def logits(self, tokens, z):
        e = self.emb(tokens).transpose(1, 2)
        zz = z.unsqueeze(-1).expand(-1, -1, e.size(-1))
        return self.net(self.inp(torch.cat([e, zz], 1)))

    def forward(self, tokens, z, target=None):
        lg = self.logits(tokens, z)
        tgt = tokens if target is None else target
        return lg[:, :, :-1], tgt[:, 1:]

    @torch.no_grad()
    def generate(self, z, start_id=2, length=24, no_repeat: int = 3, temp: float = 1.0):
        """贪心/温度采样 + 禁止 n-gram 重复（治"啊啊啊啊"这类退化）。"""
        out = torch.full((z.size(0), 1), start_id, dtype=torch.long)
        for _ in range(length - 1):
            logits = self.logits(out, z)[:, :, -1]                # (B,V)
            if no_repeat > 0 and out.size(1) >= no_repeat:
                n = no_repeat
                last = out[:, -n:].tolist()
                for b in range(out.size(0)):
                    seen = set()
                    for i in range(out.size(1) - n):
                        if out[b, i:i + n].tolist() == last[b]:
                            nxt_id = out[b, i + n].item()
                            seen.add(nxt_id)
                    for sid in seen:
                        logits[b, sid] = -1e4
            if temp != 1.0:
                nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], 1)
        return out


class TextDecoderCont(nn.Module):
    """连续解码：一次前向直接输出 L×d 的连续向量序列，不做逐字 argmax。

    字符只是一个"量化落点"——训练时回归到固定的字符嵌入，推理时取最近邻。
    没有自回归反馈，所以不会出现"啊啊啊啊"式自激振荡。
    """

    def __init__(self, vocab: int, dim: int = 128, d: int = 64, length: int = 24, ch: int = 192):
        super().__init__()
        # 固定的目标嵌入（不参与训练，避免目标漂移）
        g = torch.Generator().manual_seed(7)
        emb = torch.randn(vocab, d, generator=g)
        emb[0] = 0.0
        self.register_buffer("target_emb", torch.nn.functional.normalize(emb, dim=-1))
        self.length = length
        self.pos = nn.Parameter(torch.randn(1, length, ch) * 0.02)
        self.inp = nn.Linear(dim, ch)
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.GELU(),
            nn.Conv1d(ch, d, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.inp(z).unsqueeze(-1).expand(-1, -1, self.length) + self.pos.transpose(1, 2)
        out = self.net(h).transpose(1, 2)              # (B,L,d)
        return torch.nn.functional.normalize(out, dim=-1)

    def loss(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        pred = self.forward(z)                          # (B,L,d)
        tgt = self.target_emb[tokens]                   # (B,L,d)
        cos = (pred * tgt).sum(-1)
        mask = (tokens != 0).float()
        return ((1 - cos) * mask).sum() / mask.sum().clamp(min=1)

    def cycle_loss(self, z: torch.Tensor, text_enc, temp: float = 0.2) -> torch.Tensor:
        """正向解码出的文字，反向再编码应回到同一个 z。"""
        pred = self.forward(z)                          # (B,L,d)
        soft = (pred @ self.target_emb.t() / temp).softmax(-1)   # (B,L,V)
        emb = soft @ text_enc.emb.weight                # (B,L,emb)
        z_re = text_enc.encode_emb(emb.transpose(1, 2))  # (B,dim)
        return (1 - F.cosine_similarity(z_re, z, dim=-1)).mean()

    @torch.no_grad()
    def generate(self, z: torch.Tensor) -> torch.Tensor:
        pred = self.forward(z)                          # (B,L,d)
        sim = pred @ self.target_emb.t()                # (B,L,V)
        return sim.argmax(-1)


class AudioDecoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(dim, 64 * 5 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 64, 5, 4)
        return F.adaptive_avg_pool2d(torch.sigmoid(self.net(h)), (40, 64))


class BrainV4(nn.Module):
    def __init__(self, vocab: int, dim: int = 128, clip_dim: int = 512,
                 slots: int = 4, rounds: int = 12, cont_decoder: bool = False):
        super().__init__()
        self.clip_proj = nn.Sequential(nn.Linear(clip_dim, dim), nn.LayerNorm(dim))
        self.video = VideoEncoder(dim)
        self.audio = AudioEncoder(dim)
        self.text = TextEncoder(vocab, dim)
        self.gws = GlobalWorkspace(dim, slots, rounds)
        self.use_cont = cont_decoder
        self.text_head = TextDecoderCont(vocab, dim) if self.use_cont else TextDecoder(vocab, dim)
        self.audio_head = AudioDecoder(dim)
        # 自评头：预测"这次输出错了多少"
        self.critic = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        # 路由头：选择输出模态（0 文本 1 音频 2 图像 3 全部）
        self.router = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, v=None, a=None, t_in=None, clip=None, t_dec=None):
        toks, zs = [], []
        if clip is not None:
            zc = self.clip_proj(clip)
            toks.append(zc.unsqueeze(1)); zs.append(zc)
        if v is not None:
            vt, zv, skip = self.video(v)
            toks.append(vt); zs.append(zv)
        if a is not None:
            at, za = self.audio(a)
            toks.append(at); zs.append(za)
        if t_in is not None:
            tt, zt, mask = self.text(t_in)
            toks.append(tt); zs.append(zt)

        # 文本解码只能用"非文本"模态构成的工作空间，否则等于抄答案
        nontext = []
        if clip is not None:
            nontext.append(self.clip_proj(clip).unsqueeze(1))
        if v is not None:
            nontext.append(self.video(v)[0])
        if a is not None:
            nontext.append(self.audio(a)[0])
        all_tok = torch.cat(toks, 1)
        slots = self.gws(all_tok)                       # (B,slots,D)
        zj = slots.mean(1)                              # 工作空间整体状态（含文本，用于检索/路由）
        z_dec = self.gws(torch.cat(nontext, 1)).mean(1) if nontext else zj.detach()

        out = {"slots": slots, "z": zj}
        if t_in is not None:
            if self.use_cont:
                out["chars"] = self.text_head.loss(z_dec, t_in)
            else:
                out["chars"] = self.text_head(t_dec if t_dec is not None else t_in, z_dec, target=t_in)
        out["z_dec"] = z_dec
        out["mels"] = self.audio_head(z_dec)
        out["err_pred"] = self.critic(zj).squeeze(-1)
        out["route"] = self.router(zj)
        if clip is not None:
            out["zc"] = zs[0]
        if t_in is not None:
            out["zt"] = zs[-1]
        return out


class FailureMemory:
    """记录失败的输出，避免模型在返工时反复给出同一个错误答案。"""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.seen: dict[tuple, int] = {}

    def add(self, seq) -> None:
        key = tuple(int(x) for x in seq)
        self.seen[key] = self.seen.get(key, 0) + 1
        if len(self.seen) > self.capacity:
            self.seen.pop(next(iter(self.seen)))

    def repeated(self, seq) -> bool:
        return tuple(int(x) for x in seq) in self.seen

    @torch.no_grad()
    def generate_avoiding(self, decoder, z, tries: int = 3, **kw):
        """生成；如果撞上历史失败答案，就加噪重采。"""
        out = decoder.generate(z, **kw)
        for _ in range(tries):
            bad = [i for i in range(out.size(0)) if self.repeated(out[i])]
            if not bad:
                break
            zz = z[bad]
            out[bad] = decoder.generate(zz + 0.05 * torch.randn_like(zz), **kw)
        for i in range(out.size(0)):
            self.add(out[i])
        return out
