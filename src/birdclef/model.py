import numpy as np
import timm
import torch
import torch.nn as nn

from . import config as C


def mel_filters(sr=C.SR, n_fft=C.N_FFT, n_mels=C.N_MELS, f_min=C.F_MIN, f_max=C.F_MAX):
    """HTK mel filterbank [n_fft // 2 + 1, n_mels]."""
    mel = np.linspace(*(2595 * np.log10(1 + np.array([f_min, f_max]) / 700)), n_mels + 2)
    hz = 700 * (10 ** (mel / 2595) - 1)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)[:, None]
    lo, mid, hi = hz[:-2], hz[1:-1], hz[2:]
    return np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None).astype(np.float32)


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(C.N_FFT), persistent=False)
        self.register_buffer("filters", torch.from_numpy(mel_filters()), persistent=False)

    def forward(self, wave):  # [B, samples] -> [B, 1, mels, samples // HOP], standardised per sample
        spec = torch.stft(wave, C.N_FFT, C.HOP, window=self.window, return_complex=True).abs() ** 2
        spec = spec[..., : wave.shape[-1] // C.HOP]  # frame k centred on sample k * HOP: 32 frames per second
        db = 10 * torch.log10((spec.transpose(1, 2) @ self.filters).clamp_min(1e-10)).transpose(1, 2)
        db = torch.maximum(db, db.amax((1, 2), keepdim=True) - C.TOP_DB)
        return ((db - db.mean((1, 2), keepdim=True)) / (db.std((1, 2), keepdim=True) + 1e-6)).unsqueeze(1)


class GeM(nn.Module):
    """Generalised mean over one dimension, with a learnt exponent."""

    def __init__(self, dim, p=3.0, eps=1e-6):
        super().__init__()
        self.dim, self.eps, self.p = dim, eps, nn.Parameter(torch.tensor(p))

    def forward(self, x):
        return x.clamp_min(self.eps).pow(self.p).mean(self.dim).pow(1 / self.p)


class Net(nn.Module):
    """Waveform -> (clip logits [B, classes], frame logits [B, seconds, classes]).

    Log-mel at 32 frames per second, a timm CNN (time and frequency downsampled by 32), GeM over frequency, then a
    sound event detection head: logits per frame (one per second) and an attention over frames that pools them into
    the clip logits. `embed` projects the average-pooled backbone features to Perch's embedding size, for
    distillation."""

    def __init__(
        self,
        n_classes,
        backbone="tf_efficientnet_b0.ns_jft_in1k",
        pretrained=True,
        dropout=0.25,
        embed_dim=1536,
        drop_path=0.0,
    ):
        super().__init__()
        self.mel = LogMel()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="", drop_path_rate=drop_path
        )
        self.pool = GeM(dim=2)
        width = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        self.frame = nn.Linear(width, n_classes)
        self.attention = nn.Linear(width, n_classes)
        self.embed = nn.Linear(width, embed_dim)

    def features(self, wave):
        with torch.autocast(wave.device.type, enabled=False):
            x = self.mel(wave.float())
        return self.backbone(x)  # [B, channels, mels / 32, seconds]

    def head(self, f):
        x = self.dropout(self.pool(f).transpose(1, 2))  # [B, frames, channels]
        frame = self.frame(x)
        weights = self.attention(x).float().softmax(1)  # over frames, per class
        return (weights * frame.float()).sum(1), frame

    def forward(self, wave):
        return self.head(self.features(wave))

    def forward_with_embedding(self, wave):
        f = self.features(wave)
        return *self.head(f), self.embed(f.mean((2, 3)))


def masked_ce(logits, target, mask):
    """Cross-entropy with soft targets (normalized to sum to 1); masked classes are left out of the softmax."""
    logits = logits.float().masked_fill(mask == 0, -1e4)
    target = target / target.sum(1, keepdim=True).clamp_min(1e-6)
    return -(target * logits.log_softmax(1)).sum(1).mean()


def cosine_loss(pred, target, valid):
    """Mean of 1 - cosine similarity over the samples with a target embedding."""
    loss = 1 - nn.functional.cosine_similarity(pred.float(), target.float(), dim=1)
    return (loss * valid).sum() / valid.sum().clamp_min(1)
