from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RobustifierOutput:
    values: Tensor
    observed: Tensor
    reliability: Tensor
    standardized_innovation: Tensor
    local_median: Tensor
    local_scale: Tensor
    recovery: Tensor


def forward_fill(values: Tensor, observed: Tensor) -> Tensor:
    """Causal forward fill with zero initialization for leading gaps."""
    positions = torch.arange(
        values.shape[1], dtype=torch.long, device=values.device
    ).view(1, -1, 1)
    last_observed = torch.where(
        observed,
        positions,
        torch.full_like(positions, -1),
    ).cummax(dim=1).values
    filled = torch.gather(values, 1, last_observed.clamp_min(0))
    return torch.where(last_observed >= 0, filled, torch.zeros_like(filled))


def median_mad_robustify(
    x: Tensor,
    mask: Optional[Tensor] = None,
    window: int = 7,
    huber_delta: float = 3.0,
) -> RobustifierOutput:
    """Apply the same causal median/MAD clipping rule used by KAMO-v5.

    This function is intentionally parameter free.  It is shared by the
    plug-in robustified baselines and tested against KAMO-v5's internal
    implementation so the baseline comparison changes only the forecasting
    architecture after preprocessing.
    """
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, context, variables]")
    observed = torch.isfinite(x) if mask is None else (mask.bool() & torch.isfinite(x))
    values = forward_fill(torch.nan_to_num(x), observed)
    width = min(max(1, int(window) | 1), values.shape[1])
    if width % 2 == 0:
        width = max(1, width - 1)

    signal = values.transpose(1, 2)
    padded = F.pad(signal, (width - 1, 0), mode="replicate")
    neighborhoods = padded.unfold(-1, width, 1)
    neighborhood_median = neighborhoods.median(dim=-1, keepdim=True).values
    median = neighborhood_median.squeeze(-1).transpose(1, 2)
    mad = (neighborhoods - neighborhood_median).abs().median(dim=-1).values
    mad = mad.transpose(1, 2)
    scale = 1.4826 * mad + 0.05
    standardized = (values - median) / scale
    reliability = observed.to(x.dtype) * torch.clamp(
        float(huber_delta) / standardized.abs().clamp_min(1e-6), max=1.0
    )
    clipped = standardized.clamp(-float(huber_delta), float(huber_delta))
    filtered = median + scale * clipped

    missing_signal = (~observed).to(x.dtype).transpose(1, 2)
    missing_padded = F.pad(missing_signal, (width - 1, 0), value=0.0)
    recent_missing = F.max_pool1d(
        missing_padded, kernel_size=width, stride=1
    ).transpose(1, 2).bool()
    recovery = observed & recent_missing
    filtered = torch.where(recovery, values, filtered)
    reliability = torch.where(recovery, torch.ones_like(reliability), reliability)
    return RobustifierOutput(
        values=filtered,
        observed=observed,
        reliability=reliability,
        standardized_innovation=standardized,
        local_median=median,
        local_scale=scale,
        recovery=recovery,
    )


class RobustifiedInputWrapper(nn.Module):
    """Prepend KAMO's fixed robustifier to an unchanged forecasting backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        robust_window: int = 7,
        huber_delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.robust_window = int(robust_window)
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
        **kwargs,
    ):
        robust = median_mad_robustify(
            x,
            mask=mask,
            window=self.robust_window,
            huber_delta=self.huber_delta,
        )
        output = self.backbone(
            robust.values,
            mask=robust.observed,
            return_diagnostics=False,
            **kwargs,
        )
        if return_diagnostics:
            output.diagnostics = {
                "robust_values": robust.values,
                "reliability": robust.reliability,
                "standardized_innovation": robust.standardized_innovation,
                "local_median": robust.local_median,
                "robust_scale": robust.local_scale,
                "recovery": robust.recovery.to(x.dtype),
            }
        return output


class ReliabilityConditionedWrapper(nn.Module):
    """Portable reliability conditioning for an unchanged forecasting backbone.

    In contrast with :class:`RobustifiedInputWrapper`, this adapter uses the
    Huber reliability both to estimate the reversible normalization statistics
    and to calibrate predictive uncertainty.  The forecasting backbone itself
    is not modified, which makes the transfer experiment identical for PatchTST
    and iTransformer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        robust_window: int = 7,
        huber_delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.robust_window = int(robust_window)
        self.huber_delta = float(huber_delta)
        self.raw_uncertainty_gain = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
        **kwargs,
    ):
        robust = median_mad_robustify(
            x,
            mask=mask,
            window=self.robust_window,
            huber_delta=self.huber_delta,
        )
        weights = robust.reliability * robust.observed.to(x.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        location = (robust.values * weights).sum(dim=1, keepdim=True) / count
        variance = (
            (robust.values - location).square() * weights
        ).sum(dim=1, keepdim=True) / count
        scale = variance.add(1e-5).sqrt().clamp_min(0.1)
        normalized = torch.where(
            robust.observed,
            (robust.values - location) / scale,
            torch.zeros_like(robust.values),
        ).clamp(-8.0, 8.0)
        output = self.backbone(
            normalized,
            mask=robust.observed,
            return_diagnostics=False,
            **kwargs,
        )
        corruption_fraction = 1.0 - robust.reliability.mean(dim=1, keepdim=True)
        output.mean = output.mean * scale + location
        output.std = (
            output.std
            * scale.clamp_min(0.1)
            * (1.0 + F.softplus(self.raw_uncertainty_gain) * corruption_fraction)
        ).clamp_min(1e-4)
        if return_diagnostics:
            output.diagnostics = {
                "robust_values": robust.values,
                "reliability": robust.reliability,
                "normalized_input": normalized,
                "location": location,
                "normalization_scale": scale,
                "corruption_fraction": corruption_fraction,
            }
        return output


class ReliabilitySoftInputWrapper(nn.Module):
    """Parameter-free portable reliability adapter for external backbones.

    The legacy :class:`ReliabilityConditionedWrapper` performs an additional
    reversible normalization and predictive-scale transformation.  Those
    operations are suitable only when the downstream architecture is designed
    around them and can disturb the native RevIN/probabilistic head of an
    otherwise unchanged PatchTST or iTransformer.

    This adapter transfers only the common reliability signal.  It blends the
    causally filled observation with the bounded-influence robust value at each
    time-variable location::

        x_soft = reliability * x_filled + (1 - reliability) * x_robust

    Reliable observations therefore remain unchanged, while unreliable or
    missing locations move smoothly toward the robust estimate.  The backbone
    keeps its own normalization and prediction head, and its mean/std outputs
    are returned without post-hoc modification.  The adapter has no trainable
    parameters, giving an exactly parameter-matched transfer comparison.
    """

    def __init__(
        self,
        backbone: nn.Module,
        robust_window: int = 7,
        huber_delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.robust_window = int(robust_window)
        self.huber_delta = float(huber_delta)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
        **kwargs,
    ):
        robust = median_mad_robustify(
            x,
            mask=mask,
            window=self.robust_window,
            huber_delta=self.huber_delta,
        )
        filled = forward_fill(torch.nan_to_num(x), robust.observed)
        reliability = robust.reliability.clamp(0.0, 1.0)
        soft_values = (
            reliability * filled + (1.0 - reliability) * robust.values
        )
        output = self.backbone(
            soft_values,
            mask=robust.observed,
            return_diagnostics=False,
            **kwargs,
        )
        if return_diagnostics:
            output.diagnostics = {
                "filled_values": filled,
                "robust_values": robust.values,
                "soft_values": soft_values,
                "reliability": reliability,
                "standardized_innovation": robust.standardized_innovation,
                "local_median": robust.local_median,
                "robust_scale": robust.local_scale,
                "recovery": robust.recovery.to(x.dtype),
            }
        return output
