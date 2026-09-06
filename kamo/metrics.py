from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(z)
    except ImportError:
        return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def gaussian_crps_values(
    y: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Element-wise CRPS for a diagonal Gaussian predictive distribution."""
    std = np.maximum(std, 1e-6)
    z = (y - mean) / std
    return std * (
        z * (2.0 * _normal_cdf(z) - 1.0)
        + 2.0 * _normal_pdf(z)
        - 1.0 / math.sqrt(math.pi)
    )


def gaussian_crps(
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    score = gaussian_crps_values(y, mean, std)
    if mask is not None:
        score = score[np.asarray(mask, dtype=bool)]
    if score.size == 0:
        raise ValueError("CRPS is undefined because the evaluation mask is empty")
    return float(np.mean(score))


def regression_metrics(
    y: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-6)
    if y.shape != mean.shape or y.shape != std.shape:
        raise ValueError(
            f"y, mean and std must have identical shapes, got "
            f"{y.shape}, {mean.shape}, {std.shape}"
        )
    observed = np.isfinite(y) & np.isfinite(mean) & np.isfinite(std)
    if mask is not None:
        observed &= np.broadcast_to(np.asarray(mask, dtype=bool), y.shape)
    if not np.any(observed):
        raise ValueError("Metrics are undefined because the evaluation mask is empty")
    error = mean - y
    mse = float(np.mean(error[observed] ** 2))
    mae = float(np.mean(np.abs(error[observed])))
    nll_values = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * (error / std) ** 2
    nll = float(np.mean(nll_values[observed]))
    out = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "nll": nll,
        "crps": gaussian_crps(y, mean, std, observed),
    }
    quantiles = {
        0.80: 1.2815515655,
        0.90: 1.6448536270,
        0.95: 1.9599639845,
        0.99: 2.5758293035,
    }
    for level, z in quantiles.items():
        lower, upper = mean - z * std, mean + z * std
        key = int(level * 100)
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        alpha = 1.0 - level
        interval_score = (
            width
            + (2.0 / alpha) * np.maximum(lower - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - upper, 0.0)
        )
        out[f"picp_{key}"] = float(np.mean(covered[observed]))
        out[f"mpiw_{key}"] = float(np.mean(width[observed]))
        out[f"interval_score_{key}"] = float(np.mean(interval_score[observed]))
        out[f"calibration_error_{key}"] = abs(out[f"picp_{key}"] - level)
    return out
