from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import ForecastOutput


def _compatible_heads(d_model: int, requested: int) -> int:
    """Return the largest requested-or-smaller head count dividing d_model."""
    for heads in range(min(requested, d_model), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def _masked_revin(
    x: Tensor, mask: Optional[Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-window reversible normalization using observed values only."""
    observed = torch.isfinite(x) if mask is None else (mask.bool() & torch.isfinite(x))
    values = torch.where(observed, torch.nan_to_num(x), torch.zeros_like(x))
    weights = observed.to(values.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    location = (values * weights).sum(dim=1, keepdim=True) / count
    variance = ((values - location).square() * weights).sum(dim=1, keepdim=True) / count
    scale = variance.add(1e-5).sqrt().clamp_min(0.1)
    normalized = torch.where(observed, (values - location) / scale, torch.zeros_like(values))
    return normalized, location, scale


def _denormalize_output(
    mean_normalized: Tensor,
    std_normalized: Tensor,
    location: Tensor,
    scale: Tensor,
) -> ForecastOutput:
    mean = mean_normalized * scale + location
    std = std_normalized.clamp_min(1e-4) * scale.clamp_min(0.1)
    return ForecastOutput(mean=mean, std=std.clamp_min(1e-4))


class PatchTSTForecaster(nn.Module):
    """Channel-independent PatchTST with a Gaussian prediction head.

    The backbone follows the supervised PatchTST design: patch tokenization,
    shared channel-independent Transformer encoding, and a flattened direct head.
    """

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 48,
        patch_len: int = 16,
        stride: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = min(patch_len, seq_len)
        self.stride = max(1, stride)
        self.num_patches = 1 + math.ceil(
            max(0, seq_len - self.patch_len) / self.stride
        )
        padded_length = (self.num_patches - 1) * self.stride + self.patch_len
        self.pad_right = padded_length - seq_len
        heads = _compatible_heads(d_model, n_heads)
        self.patch_embedding = nn.Linear(self.patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.num_patches * d_model, 2 * pred_len)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        normalized, location, scale = _masked_revin(x, mask)
        batch, _, variables = normalized.shape
        sequence = normalized.transpose(1, 2)
        if self.pad_right:
            sequence = F.pad(sequence, (0, self.pad_right), mode="replicate")
        patches = sequence.unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_embedding(patches.reshape(batch * variables, self.num_patches, self.patch_len))
        tokens = self.encoder(tokens + self.position)
        raw = self.head(self.norm(tokens).flatten(1))
        raw = raw.view(batch, variables, self.pred_len, 2)
        mean_norm = raw[..., 0].transpose(1, 2)
        std_norm = (F.softplus(raw[..., 1]) + 1e-4).transpose(1, 2)
        return _denormalize_output(mean_norm, std_norm, location, scale)


class ITransformerForecaster(nn.Module):
    """iTransformer variate-token encoder with a Gaussian prediction head."""

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 48,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        heads = _compatible_heads(d_model, n_heads)
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.variate_id = nn.Parameter(torch.zeros(1, num_vars, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * pred_len)
        nn.init.normal_(self.variate_id, std=0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        normalized, location, scale = _masked_revin(x, mask)
        tokens = self.variate_embedding(normalized.transpose(1, 2)) + self.variate_id
        encoded = self.norm(self.encoder(tokens))
        raw = self.head(encoded)
        batch, variables, _ = raw.shape
        raw = raw.view(batch, variables, self.pred_len, 2)
        mean_norm = raw[..., 0].transpose(1, 2)
        std_norm = (F.softplus(raw[..., 1]) + 1e-4).transpose(1, 2)
        return _denormalize_output(mean_norm, std_norm, location, scale)


class _DecomposedTemporalMixer(nn.Module):
    def __init__(self, length: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.length = length
        width = max(hidden, length)
        self.seasonal = nn.Sequential(
            nn.Linear(length, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, length)
        )
        self.trend = nn.Sequential(
            nn.Linear(length, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, length)
        )
        self.norm = nn.LayerNorm(length)

    def forward(self, x: Tensor) -> Tensor:
        window = min(7, self.length)
        if window % 2 == 0:
            window -= 1
        padded = F.pad(x, (window // 2, window // 2), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=window, stride=1)
        seasonal = x - trend
        return self.norm(x + self.seasonal(seasonal) + self.trend(trend))


class TimeMixerForecaster(nn.Module):
    """Compact probabilistic TimeMixer adaptation with decomposed multiscale mixing."""

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 48,
        num_scales: int = 3,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scales = [2 ** i for i in range(max(1, num_scales)) if seq_len // (2 ** i) >= 4]
        self.lengths = [seq_len // factor for factor in self.scales]
        self.mixers = nn.ModuleList(
            [
                nn.Sequential(
                    *[_DecomposedTemporalMixer(length, d_model, dropout) for _ in range(n_layers)]
                )
                for length in self.lengths
            ]
        )
        self.predictors = nn.ModuleList(
            [nn.Linear(length, 2 * pred_len) for length in self.lengths]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        normalized, location, scale = _masked_revin(x, mask)
        channel_first = normalized.transpose(1, 2)
        means, stds = [], []
        for factor, mixer, predictor in zip(self.scales, self.mixers, self.predictors):
            current = channel_first if factor == 1 else F.avg_pool1d(channel_first, factor, factor)
            mixed = mixer(current)
            raw = predictor(mixed).view(x.shape[0], self.num_vars, self.pred_len, 2)
            means.append(raw[..., 0].transpose(1, 2))
            stds.append((F.softplus(raw[..., 1]) + 1e-4).transpose(1, 2))
        weights = torch.softmax(self.scale_logits, dim=0)
        mean_norm = sum(weight * mean for weight, mean in zip(weights, means))
        second = sum(
            weight * (std.square() + mean.square())
            for weight, mean, std in zip(weights, means, stds)
        )
        std_norm = (second - mean_norm.square()).clamp_min(1e-6).sqrt()
        return _denormalize_output(mean_norm, std_norm, location, scale)


class VarTCNBlock(nn.Module):
    """Official VPNet local cross-variate/patch depthwise TCN block."""

    def __init__(
        self,
        hidden_size: int,
        kernel_vars: int = 7,
        kernel_patches: int = 3,
        ff_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        kernel_vars = max(1, int(kernel_vars) | 1)
        kernel_patches = max(1, int(kernel_patches) | 1)
        self.depthwise = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(kernel_vars, kernel_patches),
            padding=(kernel_vars // 2, kernel_patches // 2),
            groups=hidden_size,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(hidden_size)
        self.expand = nn.Conv2d(hidden_size, hidden_size * ff_ratio, 1)
        self.contract = nn.Conv2d(hidden_size * ff_ratio, hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.norm(self.depthwise(x))
        residual = self.dropout(F.gelu(self.expand(F.gelu(residual))))
        residual = self.dropout(self.contract(residual))
        return x + residual


class VPNetForecaster(nn.Module):
    """ICLR-2026 VPNet backbone adapted with a Gaussian output head.

    Patch encode/decode, the variate-patch 2-D field, local VarTCN blocks, and
    the per-variate latent predictor follow the authors' official implementation.
    """

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 96,
        patch_len: int = 8,
        middle_size: int = 256,
        n_layers: int = 2,
        kernel_vars: int = 7,
        kernel_patches: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.1,
        reconstruction_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % patch_len or pred_len % patch_len:
            raise ValueError("VPNet requires seq_len and pred_len divisible by patch_len")
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.num_patches = seq_len // patch_len
        self.num_future_patches = pred_len // patch_len
        self.hidden_size = d_model
        self.reconstruction_weight = reconstruction_weight
        self.encoder = nn.Sequential(
            nn.Linear(patch_len, middle_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, d_model),
            nn.LayerNorm(d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, middle_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(middle_size, patch_len),
        )
        self.var_tcn = nn.Sequential(
            *[
                VarTCNBlock(
                    d_model,
                    kernel_vars=kernel_vars,
                    kernel_patches=kernel_patches,
                    ff_ratio=ff_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        input_size = d_model * self.num_patches
        output_size = d_model * self.num_future_patches
        predictor_width = max(middle_size, input_size // 2)
        self.future_predictor = nn.Sequential(
            nn.Linear(input_size, predictor_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_width, output_size),
        )
        self.scale_predictor = nn.Linear(input_size, pred_len)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        normalized, location, scale = _masked_revin(x, mask)
        batch = x.shape[0]
        patches = normalized.transpose(1, 2).reshape(
            batch, self.num_vars, self.num_patches, self.patch_len
        )
        encoded = self.encoder(patches)
        reconstruction = self.decoder(encoded)
        field = encoded.permute(0, 3, 1, 2).contiguous()
        mixed = self.var_tcn(field).permute(0, 2, 3, 1).contiguous()
        flattened = mixed.reshape(batch * self.num_vars, -1)
        future_latent = self.future_predictor(flattened).reshape(
            batch, self.num_vars, self.num_future_patches, self.hidden_size
        )
        mean_norm = self.decoder(future_latent).reshape(
            batch, self.num_vars, self.pred_len
        ).transpose(1, 2)
        std_norm = (F.softplus(self.scale_predictor(flattened)) + 1e-4).reshape(
            batch, self.num_vars, self.pred_len
        ).transpose(1, 2)
        output = _denormalize_output(mean_norm, std_norm, location, scale)
        output.auxiliary_loss = self.reconstruction_weight * F.mse_loss(reconstruction, patches)
        return output
