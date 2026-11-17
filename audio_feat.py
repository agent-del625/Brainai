"""音频特征：把 8 秒单声道 16kHz 波形压成 (1, 40, 64) 的梅尔频谱张量。

不依赖 librosa，纯 numpy 实现（树莓派装 librosa 太重）。
"""
from __future__ import annotations

import numpy as np

SR = 16000
N_FFT = 400
N_MELS = 40
T_FRAMES = 64


def _hz_to_mel(f: np.ndarray | float):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
    fmin, fmax = 0.0, sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c += 1
        if r <= c:
            r = c + 1
        for k in range(l, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, min(r, fb.shape[1])):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = mel_filterbank()
_WIN = np.hanning(N_FFT).astype(np.float32)


def mel_spec(pcm: np.ndarray, sr: int = SR, frames: int = T_FRAMES,
             n_mels: int = N_MELS) -> np.ndarray:
    """pcm: int16/float 一维波形 → (1, 40, frames) 的 log 梅尔谱，取值约 [0,1]。"""
    x = np.asarray(pcm)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    target = frames * (len(x) // frames if len(x) >= frames else 1)
    if target < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
        target = N_FFT
    hop = max(1, len(x) // frames)
    fb = _FB if n_mels == N_MELS else mel_filterbank(n_mels)
    out = np.zeros((n_mels, frames), dtype=np.float32)
    for i in range(frames):
        start = i * hop
        seg = x[start:start + N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        spec = np.abs(np.fft.rfft(seg * _WIN)) ** 2
        out[:, i] = fb @ spec[: fb.shape[1]]
    out = np.log1p(out * 1000.0)
    hi = float(out.max()) or 1.0
    return (out / hi).astype(np.float32)[None, :, :]


def synthetic_audio(direction: int, speed: float, amp: float, seconds: float = 8.0) -> np.ndarray:
    """给合成数据造一段"运动声"：方向决定基频，速度决定滑音，亮度决定幅度。"""
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    base = 220.0 * (1.25 ** direction)
    freq = base * (1.0 + 0.15 * speed * np.sin(2 * np.pi * 0.4 * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    wave = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return (amp * 0.3 * wave * env).astype(np.float32)


def mel_to_wav(mel: np.ndarray, iters: int = 24, hop: int = 512, n_fft: int = 1024) -> np.ndarray:
    """梅尔谱 (40,64) → 波形 int16。用 Griffin-Lim 迭代估计相位（无需训练）。"""
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    frames = int(round(8.0 * SR / hop))
    # 时间维插值到足够分辨率
    t_old = np.linspace(0, 1, m.shape[1])
    t_new = np.linspace(0, 1, frames)
    m2 = np.stack([np.interp(t_new, t_old, row) for row in m])
    # 反梅尔：用滤波器组的伪逆
    fb = mel_filterbank(m.shape[0], n_fft)
    inv = np.linalg.pinv(fb)                     # (freq, 40)
    mag = np.maximum(inv @ m2, 1e-6)             # (freq, frames)
    spec = mag.astype(np.complex64)
    for _ in range(iters):
        wav = _istft(spec, hop, n_fft)
        spec = _stft(wav, hop, n_fft)
        spec = mag * np.exp(1j * np.angle(spec))
    wav = _istft(spec, hop, n_fft)
    peak = np.max(np.abs(wav)) or 1.0
    return (wav / peak * 0.9 * 32767).astype(np.int16)


def _stft(x: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    n = 1 + (len(x) - n_fft) // hop
    out = np.zeros((n_fft // 2 + 1, n), dtype=np.complex64)
    for i in range(n):
        seg = x[i * hop: i * hop + n_fft] * win
        out[:, i] = np.fft.rfft(seg)
    return out


def _istft(spec: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
    win = np.hanning(n_fft).astype(np.float32)
    n = spec.shape[1]
    out = np.zeros(hop * (n - 1) + n_fft, dtype=np.float32)
    for i in range(n):
        seg = np.fft.irfft(spec[:, i], n=n_fft).astype(np.float32) * win
        out[i * hop: i * hop + n_fft] += seg
    return out
