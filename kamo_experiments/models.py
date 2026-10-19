"""Official mean backbones with an explicitly documented Gaussian scale head.

Never import these classes under the earlier ``DLinearForecaster`` name: that
checkpoint architecture is different. The four vendored backbones are unmodified.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from kamo.model import ForecastOutput
from .vendor import cyclenet, dlinear, nlinear, tqnet


NEW_MODEL_NAMES = ("dlinear", "nlinear", "cyclenet", "tqnet")
NEW_MODELS = NEW_MODEL_NAMES
DEFAULT_CYCLES = {"ETTh1": 24, "ETTh2": 24, "weather": 144,
                  "electricity": 168, "exchange": 20}


def _observed_input(x: Tensor, mask: Optional[Tensor]):
    """Window-mean fill, using only observed history; preserve every observed value."""
    if x.ndim != 3:
        raise ValueError("Expected x with shape [batch, history, variables]")
    observed = torch.isfinite(x)
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError("mask must have the same shape as x")
        observed = observed & mask.bool()
    clean = torch.where(observed, x, torch.zeros_like(x))
    count = observed.sum(dim=1, keepdim=True).to(x.dtype).clamp_min(1.0)
    location = clean.sum(dim=1, keepdim=True) / count
    filled = torch.where(observed, x, location.expand_as(x))
    variance = ((filled - location).square() * observed.to(x.dtype)).sum(
        dim=1, keepdim=True) / count
    scale = (variance + 1e-5).sqrt().clamp_min(0.1)
    return filled, location, scale


class OfficialGaussianForecaster(nn.Module):
    """Keep upstream mean forecast; attach common channel-independent scale head.

    The scale head is Linear(history, horizon), softplus, and rescaling by the
    observed historical window standard deviation. It uses no future targets.
    These are probabilistic adaptations, not the point-loss results in the papers.
    ``cycle_index`` is the absolute start index of each historical input window.
    """
    def __init__(self, name, num_vars, seq_len, pred_len, d_model=128,
                 dropout=0.1, individual=False, cycle_len=None,
                 cycle_model_type="mlp", use_revin=True):
        super().__init__()
        if name not in NEW_MODEL_NAMES:
            raise ValueError("Unknown new baseline: %s" % name)
        if min(num_vars, seq_len, pred_len) < 1:
            raise ValueError("num_vars, seq_len, and pred_len must be positive")
        if name == "tqnet" and seq_len % 4:
            raise ValueError("Official TQNet requires seq_len divisible by 4")
        if name in ("cyclenet", "tqnet") and (cycle_len is None or cycle_len < 1):
            raise ValueError("CycleNet/TQNet require a positive declared cycle_len")
        if use_revin and name in ("cyclenet", "tqnet") and seq_len < 2:
            raise ValueError("Official instance variance requires seq_len >= 2")
        self.name = name
        self.num_vars = int(num_vars)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.cycle_len = int(cycle_len) if cycle_len is not None else None
        config = SimpleNamespace(enc_in=self.num_vars, seq_len=self.seq_len,
                                 pred_len=self.pred_len, individual=bool(individual),
                                 d_model=int(d_model), dropout=float(dropout),
                                 cycle=self.cycle_len, model_type=cycle_model_type,
                                 use_revin=bool(use_revin))
        module = {"dlinear": dlinear, "nlinear": nlinear,
                  "cyclenet": cyclenet, "tqnet": tqnet}[name]
        self.backbone = module.Model(config)
        self.scale_head = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None,
                cycle_index=None, window_start=None, **kwargs):
        filled, location, scale = _observed_input(x, mask)
        if x.shape[1:] != (self.seq_len, self.num_vars):
            raise ValueError("Input shape does not match model seq_len/num_vars")
        if self.name in ("cyclenet", "tqnet"):
            phase = cycle_index if cycle_index is not None else window_start
            if phase is None:
                raise ValueError("%s requires actual global window starts as cycle_index; "
                                 "batch order or constant zero is invalid" % self.name)
            phase = torch.as_tensor(phase, device=x.device)
            if phase.dtype.is_floating_point and not torch.equal(phase, phase.round()):
                raise ValueError("cycle_index must contain integer window starts")
            phase = phase.long().reshape(-1)
            if phase.numel() != x.shape[0]:
                raise ValueError("cycle_index must have one start per batch item")
            mean = self.backbone(filled, phase.remainder(self.cycle_len))
        else:
            mean = self.backbone(filled)
        scale_input = ((filled - location) / scale).transpose(1, 2)
        std = (F.softplus(self.scale_head(scale_input)).transpose(1, 2) + 1e-4) * scale
        return ForecastOutput(mean=mean, std=std.clamp_min(1e-4))


def build_new_model(name, args, bundle):
    """Factory compatible with argparse.Namespace and DatasetBundle.

    Relevant args: seq_len, pred_len, d_model, dropout, linear_individual,
    cycle_len, cycle_model_type, use_revin. Only the architecture-relevant
    arguments should be included in a model's hyperparameter search.
    """
    name = str(name).lower()
    if name not in NEW_MODEL_NAMES:
        raise ValueError("Unsupported new baseline: %s" % name)
    dataset = getattr(args, "dataset", getattr(bundle, "name", ""))
    cycle_len = getattr(args, "cycle_len", None)
    if cycle_len is None:
        cycle_len = DEFAULT_CYCLES.get(dataset)
    return OfficialGaussianForecaster(
        name=name, num_vars=bundle.num_vars, seq_len=args.seq_len,
        pred_len=args.pred_len, d_model=getattr(args, "d_model", 128),
        dropout=getattr(args, "dropout", 0.1),
        individual=getattr(args, "linear_individual", False),
        cycle_len=cycle_len, cycle_model_type=getattr(args, "cycle_model_type", "mlp"),
        use_revin=getattr(args, "use_revin", True),
    )


def new_model_search_space(name, dataset):
    """Model-specific effective knobs; runner sets an equal trial budget.

    Optimizer keys are ``lr`` and ``weight_decay``. No dropout or d_model
    placebo trials are generated for either linear model. Cycle choice uses
    validation only, never test statistics. These spaces are recommendations,
    not permission to compare different numbers of completed trials.
    """
    space = {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "weight_decay": [0.0, 1e-4, 1e-3]}
    if name in ("dlinear", "nlinear"):
        space["linear_individual"] = [False, True]
    elif name in ("cyclenet", "tqnet"):
        cycle = DEFAULT_CYCLES.get(dataset)
        if cycle is None:
            raise ValueError("Declare a cycle before searching %s on %s" % (name, dataset))
        space["cycle_len"] = [5, 20] if dataset == "exchange" else [cycle]
        space["d_model"] = [64, 128, 256]
        if name == "cyclenet":
            space["cycle_model_type"] = ["mlp"]
        else:
            space["dropout"] = [0.0, 0.1, 0.3]
    else:
        raise ValueError("Unsupported new baseline: %s" % name)
    return space
