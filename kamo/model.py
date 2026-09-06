from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ForecastOutput:
    mean: Tensor
    std: Tensor
    diagnostics: Optional[Dict[str, Tensor]] = None
    auxiliary_loss: Optional[Tensor] = None


class VariableStateMixer(nn.Module):
    """Configurable cross-variable residual used for the mixer ablation."""

    def __init__(
        self,
        num_vars: int,
        rank: int,
        adjacency: Optional[np.ndarray],
        mixer_type: str = "low_rank",
    ) -> None:
        super().__init__()
        if mixer_type not in {"none", "low_rank", "dense"}:
            raise ValueError("mixer_type must be one of: none, low_rank, dense")
        self.mixer_type = mixer_type
        self.rank = max(0, min(int(rank), num_vars))
        if mixer_type == "low_rank":
            if self.rank == 0:
                raise ValueError("low_rank mixer requires rank >= 1; use mixer_type='none'")
            self.left = nn.Parameter(
                torch.randn(num_vars, self.rank) / math.sqrt(num_vars)
            )
            self.right = nn.Parameter(
                torch.randn(self.rank, num_vars) / math.sqrt(self.rank)
            )
        elif mixer_type == "dense":
            self.dense = nn.Parameter(torch.randn(num_vars, num_vars) / math.sqrt(num_vars))
        self.raw_gate = nn.Parameter(torch.tensor(-4.0)) if mixer_type != "none" else None
        self.graph_gate = (
            nn.Parameter(torch.tensor(-4.0)) if mixer_type != "none" else None
        )
        if adjacency is None:
            self.register_buffer("adjacency", torch.empty(0), persistent=False)
        else:
            matrix = torch.as_tensor(adjacency, dtype=torch.float32)
            degree = matrix.sum(-1, keepdim=True).clamp_min(1e-6)
            self.register_buffer("adjacency", matrix / degree, persistent=True)

    def forward(self, state: Tensor) -> Tensor:
        out = state
        if self.mixer_type == "low_rank":
            low_rank = torch.einsum("bdh,dr,rv->bvh", state, self.left, self.right)
            out = state + torch.sigmoid(self.raw_gate) * low_rank / math.sqrt(self.rank)
        elif self.mixer_type == "dense":
            dense = torch.einsum("bdh,dv->bvh", state, self.dense)
            out = state + torch.sigmoid(self.raw_gate) * dense / math.sqrt(state.shape[1])
        if self.mixer_type != "none" and self.adjacency.numel() > 0:
            graph = torch.einsum("vw,bwh->bvh", self.adjacency, state)
            out = out + torch.sigmoid(self.graph_gate) * (graph - state)
        return out


class KAMO(nn.Module):
    """KAMO: bounded-influence observations plus robust innovation encoding.

    A robust direct multi-horizon backbone supplies the forecast. A causal median/MAD
    filter gives every observation a Huber reliability. A GRU summarizes bounded
    innovation features and can only add a conservative, zero-start residual correction
    to the backbone. The encoder choice is fixed from the matched server ablation.
    """

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 48,
        num_scales: int = 3,
        mixer_rank: int = 16,
        adjacency: Optional[np.ndarray] = None,
        robust_window: int = 7,
        huber_delta: float = 3.0,
        dropout: float = 0.1,
        mixer_type: str = "low_rank",
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.robust_window = max(3, int(robust_window) | 1)
        self.huber_delta = float(huber_delta)
        default_windows = [1, 3, 7, 15, 31]
        self.windows = default_windows[:max(1, min(num_scales, len(default_windows)))]

        self.base_projection = nn.Linear(seq_len, pred_len)
        self.base_raw_std = nn.Parameter(torch.full((pred_len,), -1.0))
        self.variable_log_scale = nn.Parameter(torch.zeros(num_vars))

        # [level residual, difference, clipped innovation, surprise, reliability,
        #  observation mask, missing age, multiscale levels]
        feature_dim = 7 + len(self.windows)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.innovation_encoder = nn.GRU(feature_dim, d_model, batch_first=True)
        self.state_norm = nn.LayerNorm(d_model)
        self.state_mixer = VariableStateMixer(
            num_vars, mixer_rank, adjacency, mixer_type=mixer_type
        )
        self.residual_body = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(d_model, pred_len)
        self.scale_head = nn.Linear(d_model, pred_len)
        self.raw_residual_gate = nn.Parameter(torch.full((pred_len,), -2.5))
        self.raw_uncertainty_gain = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def optimizer_groups(self, learning_rate: float):
        base_ids = {
            id(self.base_projection.weight),
            id(self.base_projection.bias),
            id(self.base_raw_std),
            id(self.variable_log_scale),
        }
        base, residual = [], []
        for parameter in self.parameters():
            (base if id(parameter) in base_ids else residual).append(parameter)
        return [
            {"params": base, "lr": learning_rate},
            {"params": residual, "lr": 0.35 * learning_rate},
        ]

    @staticmethod
    def _forward_fill(x: Tensor, mask: Tensor) -> Tensor:
        # The former implementation launched one CUDA kernel per context step.
        # Carry the most recent observed index forward in one vectorized pass,
        # then gather its value.  Leading gaps retain the original zero fill.
        positions = torch.arange(
            x.shape[1], dtype=torch.long, device=x.device
        ).view(1, -1, 1)
        last_observed = torch.where(
            mask,
            positions,
            torch.full_like(positions, -1),
        ).cummax(dim=1).values
        filled = torch.gather(x, 1, last_observed.clamp_min(0))
        return torch.where(last_observed >= 0, filled, torch.zeros_like(filled))

    @staticmethod
    def _causal_average(values: Tensor, window: int) -> Tensor:
        if window <= 1:
            return values
        signal = values.transpose(1, 2)
        padded = F.pad(signal, (window - 1, 0), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=window, stride=1).transpose(1, 2)

    @staticmethod
    def _missing_age(mask: Tensor) -> Tensor:
        # For a missing entry, t - last_observed is exactly the loop-based
        # missing run length.  A virtual observation at index -1 preserves the
        # previous behavior for leading gaps.
        positions = torch.arange(
            mask.shape[1], dtype=torch.long, device=mask.device
        ).view(1, -1, 1)
        last_observed = torch.where(
            mask,
            positions,
            torch.full_like(positions, -1),
        ).cummax(dim=1).values
        age = (positions - last_observed).to(torch.float32)
        return age / max(1, mask.shape[1])

    def _robustify(self, values: Tensor, mask: Tensor):
        window = min(self.robust_window, values.shape[1])
        if window % 2 == 0:
            window = max(1, window - 1)
        signal = values.transpose(1, 2)
        padded = F.pad(signal, (window - 1, 0), mode="replicate")
        neighborhoods = padded.unfold(-1, window, 1)
        neighborhood_median = neighborhoods.median(dim=-1, keepdim=True).values
        median = neighborhood_median.squeeze(-1).transpose(1, 2)
        mad = (neighborhoods - neighborhood_median).abs()
        mad = mad.median(dim=-1).values.transpose(1, 2)
        scale = 1.4826 * mad + 0.05
        standardized = (values - median) / scale
        absolute = standardized.abs()
        huber_weight = torch.clamp(self.huber_delta / absolute.clamp_min(1e-6), max=1.0)
        reliability = mask.float() * huber_weight
        clipped = standardized.clamp(-self.huber_delta, self.huber_delta)
        filtered = median + scale * clipped
        # After a missing gap, a valid returning measurement may be far from the
        # carried-forward level. Treat it as recovery, not as an impulse. This keeps
        # missing-data adaptation separate from outlier rejection.
        missing_signal = (~mask).float().transpose(1, 2)
        missing_padded = F.pad(missing_signal, (window - 1, 0), value=0.0)
        recent_missing = F.max_pool1d(
            missing_padded, kernel_size=window, stride=1
        ).transpose(1, 2).bool()
        recovery = mask & recent_missing
        filtered = torch.where(recovery, values, filtered)
        reliability = torch.where(recovery, torch.ones_like(reliability), reliability)
        return filtered, reliability, standardized, median, scale

    def _assemble_innovation_features(
        self,
        filtered: Tensor,
        mask: Tensor,
        standardized: Tensor,
        reliability: Tensor,
        feature_reliability: Tensor,
        median: Tensor,
        robust_scale: Tensor,
        include_robust_diagnostics: bool = True,
    ):
        level = filtered[:, -1:, :]
        centered = filtered - level
        difference = torch.zeros_like(filtered)
        difference[:, 1:] = filtered[:, 1:] - filtered[:, :-1]
        clipped_innovation = standardized.clamp(-self.huber_delta, self.huber_delta)
        clipped_innovation = clipped_innovation / self.huber_delta
        surprise = torch.log1p(standardized.square()).clamp(max=4.0) / 4.0
        if not include_robust_diagnostics:
            clipped_innovation = torch.zeros_like(clipped_innovation)
            surprise = torch.zeros_like(surprise)
        missing_age = self._missing_age(mask)
        scales = [self._causal_average(centered, window) for window in self.windows]
        features = torch.stack(
            [
                centered,
                difference,
                clipped_innovation,
                surprise,
                feature_reliability,
                mask.float(),
                missing_age,
                *scales,
            ],
            dim=-1,
        )
        diagnostics = {
            "robust_values": filtered,
            "local_median": median,
            "robust_scale": robust_scale,
            "standardized_innovation": standardized,
            "surprise": surprise,
            "reliability": reliability,
            "missing_age": missing_age,
        }
        return self.feature_norm(features), filtered, reliability, diagnostics

    def _innovation_features(
        self, values: Tensor, mask: Tensor, disable_clipping: bool = False
    ):
        filtered, reliability, standardized, median, robust_scale = self._robustify(values, mask)
        if disable_clipping:
            # Pure clipping ablation: retain the same reliability scores and every
            # downstream reliability pathway, but feed the un-winsorized values
            # to the forecasting backbone and residual encoder.
            filtered = values
        return self._assemble_innovation_features(
            filtered,
            mask,
            standardized,
            reliability,
            reliability,
            median,
            robust_scale,
            True,
        )

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> ForecastOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, context, variables]")
        batch, context, variables = x.shape
        if context != self.seq_len or variables != self.num_vars:
            raise ValueError(
                f"Expected [batch,{self.seq_len},{self.num_vars}], got {tuple(x.shape)}"
            )
        observed = torch.isfinite(x) if mask is None else (mask.bool() & torch.isfinite(x))
        values = self._forward_fill(torch.nan_to_num(x), observed)
        features, robust_values, reliability, diagnostic_values = self._innovation_features(
            values, observed
        )

        level = robust_values[:, -1:, :]
        centered = robust_values - level
        base_mean = self.base_projection(centered.transpose(1, 2)).transpose(1, 2) + level

        sequence = features.permute(0, 2, 1, 3).reshape(
            batch * variables, context, features.shape[-1]
        )
        _, hidden = self.innovation_encoder(sequence)
        encoded = hidden[-1]
        state = encoded.reshape(batch, variables, self.d_model)
        state = self.state_mixer(self.state_norm(state))
        state = self.residual_body(state)

        residual = self.mean_head(state).transpose(1, 2)
        gate = torch.sigmoid(self.raw_residual_gate)[None, :, None]
        context_reliability = reliability[:, -min(12, context):].mean(dim=1, keepdim=True)
        residual_trust = 0.1 + 0.9 * context_reliability
        mean = base_mean + gate * residual_trust * residual

        scale_adjustment = self.scale_head(state).transpose(1, 2)
        base_raw = self.base_raw_std[None, :, None] + self.variable_log_scale[None, None, :]
        corruption_fraction = 1.0 - reliability.mean(dim=1, keepdim=True)
        raw_std = (
            base_raw
            + gate * torch.tanh(scale_adjustment)
            + F.softplus(self.raw_uncertainty_gain) * corruption_fraction
        )
        std = F.softplus(raw_std) + 1e-4

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                **diagnostic_values,
                "prior": values,
                "base_mean": base_mean,
                "residual": gate * residual,
                "residual_gate": torch.sigmoid(self.raw_residual_gate),
                "residual_trust": residual_trust,
                "corruption_fraction": corruption_fraction,
            }
        return ForecastOutput(mean, std.expand_as(mean), diagnostics)


def _compatible_heads(d_model: int, requested: int) -> int:
    for heads in range(min(d_model, requested), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


class ReliabilityPatchBranch(nn.Module):
    """Channel-independent patch encoder with explicit data-quality channels.

    Each patch contains the robustly normalized signal, Huber reliability,
    observation mask, and normalized time-since-observation.  Consequently a
    missing block is represented explicitly rather than being confused with a
    long constant segment produced by forward filling.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        patch_len: int,
        stride: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
        feature_channels: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = min(int(patch_len), seq_len)
        self.stride = max(1, min(int(stride), self.patch_len))
        self.num_patches = 1 + math.ceil(
            max(0, seq_len - self.patch_len) / self.stride
        )
        padded_length = (self.num_patches - 1) * self.stride + self.patch_len
        self.pad_right = padded_length - seq_len
        heads = _compatible_heads(d_model, n_heads)
        self.embedding = nn.Linear(feature_channels * self.patch_len, d_model)
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

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # features: [batch, time, variables, channels]
        batch, _, variables, channels = features.shape
        sequence = features.permute(0, 2, 3, 1)
        if self.pad_right:
            sequence = F.pad(sequence, (0, self.pad_right), mode="replicate")
        patches = sequence.unfold(-1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(
            batch * variables, self.num_patches, channels * self.patch_len
        )
        encoded = self.norm(self.encoder(self.embedding(patches) + self.position))
        raw = self.head(encoded.flatten(1)).reshape(
            batch, variables, self.pred_len, 2
        )
        mean = raw[..., 0].transpose(1, 2)
        std = (F.softplus(raw[..., 1]) + 1e-4).transpose(1, 2)
        state = encoded.mean(dim=1).reshape(batch, variables, -1)
        return mean, std, state


class RoleReliabilityController(nn.Module):
    """Parameter-matched shared-versus-separate reliability controller.

    Both modes execute the same network and therefore have exactly the same
    trainable parameter count.  ``shared_learned`` averages the five correction
    logits before applying them to every role; ``separate_learned`` leaves the
    five logits unconstrained.  This isolates the sharing constraint instead of
    confounding it with model size.
    """

    ROLE_NAMES = ("normalization", "encoding", "routing", "residual", "scale")

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.ROLE_NAMES)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        base_reliability: Tensor,
        observed: Tensor,
        missing_age: Tensor,
        surprise: Tensor,
        mode: str,
    ) -> Tensor:
        if mode not in {"shared_learned", "separate_learned", "mask_only"}:
            raise ValueError(f"Unsupported learned reliability mode: {mode}")
        observed_float = observed.to(base_reliability.dtype)
        if mode == "mask_only":
            # The matched robust-preprocessing control sees missingness/age but no
            # outlier-reliability or surprise signal.
            anchor = torch.where(
                observed,
                torch.full_like(base_reliability, 0.80),
                torch.full_like(base_reliability, 1e-3),
            )
            controller_reliability = observed_float
            controller_surprise = torch.zeros_like(surprise)
        else:
            anchor = base_reliability
            controller_reliability = base_reliability
            controller_surprise = surprise
        features = torch.stack(
            [controller_reliability, observed_float, missing_age, controller_surprise],
            dim=-1,
        )
        correction = 2.0 * torch.tanh(self.network(features))
        if mode == "shared_learned":
            correction = correction.mean(dim=-1, keepdim=True).expand_as(correction)
        anchor_logit = torch.logit(anchor.clamp(1e-4, 1.0 - 1e-4))[..., None]
        roles = torch.sigmoid(anchor_logit + correction)
        return roles * observed_float[..., None]


class KAMOV5(KAMO):
    """KAMO-v5: bounded influence plus a reliability-routed patch backbone.

    The original KAMO robustifier and innovation correction are retained.  The
    weak single linear forecasting backbone is replaced by multiple patch scales
    and a linear safety branch.  A per-series router uses corruption summaries to
    combine fine, medium, and coarse temporal resolutions.  All predictions are
    made in a reliability-weighted reversible normalization space.
    """

    def __init__(
        self,
        num_vars: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 48,
        num_scales: int = 3,
        mixer_rank: int = 16,
        adjacency: Optional[np.ndarray] = None,
        robust_window: int = 7,
        huber_delta: float = 3.0,
        dropout: float = 0.1,
        patch_scales: Tuple[int, ...] = (8, 16, 32),
        patch_layers: int = 1,
        n_heads: int = 4,
        ff_mult: int = 4,
        ablation: str = "none",
        mixer_type: str = "low_rank",
        reliability_mode: str = "fixed_shared",
        reliability_hidden: int = 16,
        recent_window: int = 12,
        residual_trust_floor: float = 0.1,
        scale_adjustment_coefficient: float = 0.2,
        reliability_strength: float = 1.0,
        uncertainty_reliability_strength: float = 1.0,
    ) -> None:
        super().__init__(
            num_vars=num_vars,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            num_scales=num_scales,
            mixer_rank=mixer_rank,
            adjacency=adjacency,
            robust_window=robust_window,
            huber_delta=huber_delta,
            dropout=dropout,
            mixer_type=mixer_type,
        )
        valid_scales = tuple(
            sorted({max(2, min(int(scale), seq_len)) for scale in patch_scales})
        )
        if not valid_scales:
            raise ValueError("patch_scales must contain at least one positive length")
        valid_ablations = {
            "none",
            "no_clipping",
            "no_reliability_normalization",
            "no_reliability_routing",
            "uniform_router",
            "no_reliability_residual_gate",
            "no_unreliability_scale",
            "no_reliability_gate_scale",
        }
        if ablation not in valid_ablations:
            raise ValueError(
                f"Unknown KAMO-v5 ablation {ablation!r}; choose from {sorted(valid_ablations)}"
            )
        self.ablation = ablation
        if reliability_mode not in {
            "fixed_shared", "shared_learned", "separate_learned", "mask_only"
        }:
            raise ValueError(
                "reliability_mode must be fixed_shared, shared_learned, "
                "separate_learned or mask_only"
            )
        if recent_window < 1:
            raise ValueError("recent_window must be positive")
        if not 0.0 <= residual_trust_floor <= 1.0:
            raise ValueError("residual_trust_floor must be in [0,1]")
        if scale_adjustment_coefficient < 0:
            raise ValueError("scale_adjustment_coefficient must be non-negative")
        if not 0.0 <= reliability_strength <= 1.0:
            raise ValueError("reliability_strength must be in [0,1]")
        if not 0.0 <= uncertainty_reliability_strength <= 1.0:
            raise ValueError("uncertainty_reliability_strength must be in [0,1]")
        self.reliability_mode = reliability_mode
        self.recent_window = int(recent_window)
        self.residual_trust_floor = float(residual_trust_floor)
        self.scale_adjustment_coefficient = float(scale_adjustment_coefficient)
        # Runtime controls are deliberately non-parametric.  They can therefore be
        # selected on validation data for an already trained checkpoint without
        # changing its weights or invalidating legacy state dictionaries.
        self.reliability_strength = float(reliability_strength)
        self.uncertainty_reliability_strength = float(
            uncertainty_reliability_strength
        )
        self.reliability_controller = (
            None
            if reliability_mode == "fixed_shared"
            else RoleReliabilityController(max(4, int(reliability_hidden)))
        )
        self.patch_scales = valid_scales
        self.patch_branches = nn.ModuleList(
            [
                ReliabilityPatchBranch(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    d_model=d_model,
                    patch_len=patch_len,
                    stride=max(1, patch_len // 2),
                    n_layers=patch_layers,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for patch_len in self.patch_scales
            ]
        )
        router_hidden = max(12, d_model // 2)
        self.router = nn.Sequential(
            nn.Linear(4, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, 1 + len(self.patch_scales)),
        )
        nn.init.zeros_(self.router[-1].weight)
        initial = torch.full((1 + len(self.patch_scales),), -0.5)
        # Start from the conventional medium (approximately length-16) patch
        # solution while leaving every branch trainable from the first step.
        medium = min(
            range(len(self.patch_scales)),
            key=lambda index: abs(self.patch_scales[index] - 16),
        )
        initial[0] = -1.0
        initial[1 + medium] = 1.5
        with torch.no_grad():
            self.router[-1].bias.copy_(initial)

    def optimizer_groups(self, learning_rate: float):
        residual_modules = [
            self.feature_norm,
            self.innovation_encoder,
            self.state_norm,
            self.state_mixer,
            self.residual_body,
            self.mean_head,
            self.scale_head,
        ]
        residual_ids = {
            id(parameter)
            for module in residual_modules
            for parameter in module.parameters()
        }
        residual_ids.update({id(self.raw_residual_gate), id(self.raw_uncertainty_gain)})
        backbone, residual = [], []
        for parameter in self.parameters():
            (residual if id(parameter) in residual_ids else backbone).append(parameter)
        return [
            {"params": backbone, "lr": learning_rate},
            {"params": residual, "lr": 0.35 * learning_rate},
        ]

    @staticmethod
    def _reliability_normalize(
        values: Tensor, observed: Tensor, reliability: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        weights = reliability * observed.to(values.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        location = (values * weights).sum(dim=1, keepdim=True) / count
        variance = ((values - location).square() * weights).sum(dim=1, keepdim=True) / count
        scale = variance.add(1e-5).sqrt().clamp_min(0.1)
        normalized = torch.where(
            observed, (values - location) / scale, torch.zeros_like(values)
        )
        return normalized.clamp(-8.0, 8.0), location, scale

    @staticmethod
    def _soften_reliability(
        reliability: Tensor, observed: Tensor, strength: float
    ) -> Tensor:
        """Interpolate reliability toward the observation mask.

        ``strength=1`` preserves the trained reliability signal, whereas
        ``strength=0`` keeps missingness information but removes outlier-driven
        attenuation.  This is a validation-selectable effect-size control, not a
        learned test-time switch.
        """
        observed_float = observed.to(reliability.dtype)
        return observed_float + float(strength) * (reliability - observed_float)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> ForecastOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, context, variables]")
        batch, context, variables = x.shape
        if context != self.seq_len or variables != self.num_vars:
            raise ValueError(
                f"Expected [batch,{self.seq_len},{self.num_vars}], got {tuple(x.shape)}"
            )
        observed = torch.isfinite(x) if mask is None else (mask.bool() & torch.isfinite(x))
        values = self._forward_fill(torch.nan_to_num(x), observed)
        robust_values, reliability, standardized, median, robust_scale = self._robustify(
            values, observed
        )
        if self.ablation == "no_clipping":
            robust_values = values
        missing_age_for_controller = self._missing_age(observed)
        surprise_for_controller = (
            torch.log1p(standardized.square()).clamp(max=4.0) / 4.0
        )
        if self.reliability_controller is None:
            role_reliability = reliability[..., None].expand(
                *reliability.shape, len(RoleReliabilityController.ROLE_NAMES)
            )
        else:
            role_reliability = self.reliability_controller(
                reliability,
                observed,
                missing_age_for_controller,
                surprise_for_controller,
                self.reliability_mode,
            )
        (
            normalization_role,
            encoding_role,
            routing_role,
            residual_role,
            scale_role,
        ) = role_reliability.unbind(dim=-1)
        normalization_role = self._soften_reliability(
            normalization_role, observed, self.reliability_strength
        )
        routing_role = self._soften_reliability(
            routing_role, observed, self.reliability_strength
        )
        residual_role = self._soften_reliability(
            residual_role, observed, self.reliability_strength
        )
        innovation_features, _, _, diagnostic_values = self._assemble_innovation_features(
            robust_values,
            observed,
            standardized,
            reliability,
            encoding_role,
            median,
            robust_scale,
            self.reliability_mode != "mask_only",
        )
        normalization_reliability = (
            observed.to(reliability.dtype)
            if self.ablation == "no_reliability_normalization"
            else normalization_role
        )
        normalized, location, scale = self._reliability_normalize(
            robust_values, observed, normalization_reliability
        )
        missing_age = diagnostic_values["missing_age"]
        patch_features = torch.stack(
            [normalized, encoding_role, observed.to(x.dtype), missing_age], dim=-1
        )

        branch_means, branch_stds = [], []
        # A direct linear branch remains as a low-variance safety path.
        linear_mean = self.base_projection(normalized.transpose(1, 2)).transpose(1, 2)
        linear_std = (
            F.softplus(
                self.base_raw_std[None, :, None]
                + self.variable_log_scale[None, None, :]
            )
            + 1e-4
        ).expand_as(linear_mean)
        branch_means.append(linear_mean)
        branch_stds.append(linear_std)
        for branch in self.patch_branches:
            branch_mean, branch_std, _ = branch(patch_features)
            branch_means.append(branch_mean)
            branch_stds.append(branch_std)

        routing_reliability = (
            observed.to(routing_role.dtype)
            if self.ablation == "no_reliability_routing"
            else routing_role
        )
        recent_routing = routing_reliability[:, -min(self.recent_window, context):]
        surprise = (
            torch.zeros_like(diagnostic_values["surprise"])
            if self.reliability_mode == "mask_only"
            else diagnostic_values["surprise"]
        )
        routing_features = torch.stack(
            [
                1.0 - routing_reliability.mean(dim=1),
                1.0 - recent_routing.mean(dim=1),
                missing_age.amax(dim=1),
                surprise.mean(dim=1),
            ],
            dim=-1,
        )
        if self.ablation == "uniform_router":
            router_weights = torch.full(
                (batch, variables, 1 + len(self.patch_scales)),
                1.0 / (1 + len(self.patch_scales)),
                dtype=x.dtype,
                device=x.device,
            )
        else:
            router_weights = torch.softmax(self.router(routing_features), dim=-1)
        mean_stack = torch.stack(branch_means, dim=-1)
        std_stack = torch.stack(branch_stds, dim=-1)
        weights = router_weights[:, None, :, :]
        mean_norm = (weights * mean_stack).sum(dim=-1)
        second = (weights * (std_stack.square() + mean_stack.square())).sum(dim=-1)
        variance_norm = (second - mean_norm.square()).clamp_min(1e-6)

        sequence = innovation_features.permute(0, 2, 1, 3).reshape(
            batch * variables, context, innovation_features.shape[-1]
        )
        _, hidden = self.innovation_encoder(sequence)
        state = hidden[-1].reshape(batch, variables, self.d_model)
        state = self.state_mixer(self.state_norm(state))
        state = self.residual_body(state)
        residual = self.mean_head(state).transpose(1, 2)
        gate = torch.sigmoid(self.raw_residual_gate)[None, :, None]
        recent_residual = residual_role[:, -min(self.recent_window, context):]
        context_reliability = recent_residual.mean(dim=1, keepdim=True)
        residual_trust = (
            torch.ones_like(context_reliability)
            if self.ablation in {
                "no_reliability_residual_gate",
                "no_reliability_gate_scale",
            }
            else self.residual_trust_floor
            + (1.0 - self.residual_trust_floor) * context_reliability
        )
        mean_norm = mean_norm + gate * residual_trust * residual

        scale_adjustment = self.scale_head(state).transpose(1, 2)
        corruption_fraction = 1.0 - scale_role.mean(dim=1, keepdim=True)
        scale_fraction = (
            torch.zeros_like(corruption_fraction)
            if self.ablation in {
                "no_unreliability_scale",
                "no_reliability_gate_scale",
            }
            else self.uncertainty_reliability_strength * corruption_fraction
        )
        std_base_norm = variance_norm.sqrt()
        scale_log_signal = torch.tanh(scale_adjustment)
        std_norm = std_base_norm * torch.exp(
            self.scale_adjustment_coefficient * scale_log_signal
        )
        uncertainty_gain = F.softplus(self.raw_uncertainty_gain)
        std_norm = std_norm * (1.0 + uncertainty_gain * scale_fraction)
        mean = mean_norm * scale + location
        std = std_norm.clamp_min(1e-4) * scale.clamp_min(0.1)

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                **diagnostic_values,
                "normalized_input": normalized,
                "location": location,
                "normalization_scale": scale,
                "router_weights": router_weights,
                "branch_mean": mean_stack * scale[..., None] + location[..., None],
                "residual": gate * residual,
                "residual_gate": torch.sigmoid(self.raw_residual_gate),
                "residual_trust": residual_trust,
                "scale_fraction": scale_fraction,
                "corruption_fraction": corruption_fraction,
                "std_base_norm": std_base_norm,
                "scale_log_signal": scale_log_signal,
                "uncertainty_gain": uncertainty_gain,
                "context_unreliability": corruption_fraction,
                "role_reliability": role_reliability,
                "routing_reliability": routing_reliability,
            }
        return ForecastOutput(mean, std.clamp_min(1e-4), diagnostics)


class DLinear(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, pred_len: int) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.pred_len = pred_len
        self.projection = nn.Linear(seq_len, pred_len)
        self.raw_std = nn.Parameter(torch.full((pred_len,), -1.0))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.zeros_like(x))
        level = x[:, -1:, :]
        centered = x - level
        mean = self.projection(centered.transpose(1, 2)).transpose(1, 2) + level
        std = (F.softplus(self.raw_std) + 1e-4)[None, :, None].expand_as(mean)
        return ForecastOutput(mean, std, None)


class GRUForecaster(nn.Module):
    def __init__(self, num_vars: int, pred_len: int, hidden: int = 48) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.pred_len = pred_len
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * pred_len)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **_) -> ForecastOutput:
        if mask is not None:
            x = torch.where(mask.bool(), x, torch.zeros_like(x))
        batch, length, variables = x.shape
        sequence = x.transpose(1, 2).reshape(batch * variables, length, 1)
        _, hidden = self.gru(sequence)
        raw = self.head(hidden[-1]).reshape(batch, variables, self.pred_len, 2)
        mean = raw[..., 0].transpose(1, 2)
        std = (F.softplus(raw[..., 1]) + 1e-4).transpose(1, 2)
        return ForecastOutput(mean, std, None)
