"""Full, ordered evaluation; no first-N-batches truncation.

The compact NPZ contains every window's masked average score and observed count.
Optional prediction shards contain every target/mean/std/mask (can be very large).
Corruptions are keyed by the original dataset window index and family, not by
training seed, batch number, GPU RNG state, or method. Severity sweeps share draws.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import zlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch.utils.data import SequentialSampler, Subset

from kamo.engine import corrupt_context, unpack_batch


CORE_CONDITIONS = (
    "clean", "gaussian:0.10", "impulse:0.05:5", "random_missing:0.20",
    "block_missing:0.20", "level_shift:1.0", "gradual_drift:1.0",
    "sensor_freeze:0.30:0.25", "heavy_tail:0.10:20",
    "burst_impulse:0.10:5", "correlated_impulse:0.05:5",
    "variable_block_missing:0.05:0.30", "impulse_missing:0.05:0.20:5",
)
SEVERITY_CONDITIONS = (
    "clean", "gaussian:0", "gaussian:0.05", "gaussian:0.10", "gaussian:0.20", "gaussian:0.40",
    "impulse:0.05:0", "impulse:0.05:1", "impulse:0.05:3", "impulse:0.05:5", "impulse:0.05:10",
    "random_missing:0", "random_missing:0.10", "random_missing:0.20", "random_missing:0.40", "random_missing:0.60",
    "block_missing:0", "block_missing:0.10", "block_missing:0.20", "block_missing:0.40", "block_missing:0.60",
    "level_shift:0", "level_shift:0.25", "level_shift:0.50", "level_shift:1", "level_shift:2", "level_shift:4",
    "gradual_drift:0", "gradual_drift:0.25", "gradual_drift:0.50", "gradual_drift:1", "gradual_drift:2",
    "sensor_freeze:0:0.25", "sensor_freeze:0.10:0.25", "sensor_freeze:0.30:0.25", "sensor_freeze:0.60:0.25",
    "heavy_tail:0:20", "heavy_tail:0.05:20", "heavy_tail:0.10:20", "heavy_tail:0.20:20",
)
DEFAULT_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def condition_slug(condition: str) -> str:
    return condition.replace(":", "_").replace("/", "_")


def ordered_window_metadata(loader):
    """Resolve Subset indices back to original timestamps/patient episodes."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Evaluation requires shuffle=False and drop_last=False")
    dataset = loader.dataset
    index = np.arange(len(dataset), dtype=np.int64)
    while isinstance(dataset, Subset):
        index = np.asarray(dataset.indices, dtype=np.int64)[index]
        dataset = dataset.dataset
    result = {"window_id": index}
    if hasattr(dataset, "window_ids"):
        result["window_id"] = np.asarray(dataset.window_ids, dtype=np.int64)[index]
    if hasattr(dataset, "window_starts"):
        result["time_start"] = np.asarray(dataset.window_starts, dtype=np.int64)[index]
    if hasattr(dataset, "group_ids"):
        result["group_id"] = np.asarray(dataset.group_ids, dtype=np.int64)[index]
    if hasattr(dataset, "indices"):
        ids = np.asarray(dataset.indices)[index]
        if ids.ndim == 2 and ids.shape[1] == 2:
            result["group_id"] = ids[:, 0].astype(np.int64)
            result["time_start"] = ids[:, 1].astype(np.int64)
    if "time_start" not in result:
        result["time_start"] = (np.asarray(dataset.starts)[index].astype(np.int64)
                                if hasattr(dataset, "starts") else index.copy())
    return result


def _batch_fields(batch, starts):
    if isinstance(batch, dict):
        x, y = batch["x"], batch["y"]
        x_mask = batch.get("x_mask", torch.isfinite(x)).bool()
        y_mask = batch.get("y_mask", torch.isfinite(y)).bool()
        window_start = batch.get("window_start", torch.as_tensor(starts))
        return x, y, x_mask, y_mask, window_start.long()
    x, y, x_mask, y_mask = unpack_batch(batch)
    return x, y, x_mask, y_mask, torch.as_tensor(starts).long()


def _validate_condition(condition):
    if condition == "clean":
        return
    parts = condition.split(":")
    if len(parts) < 2:
        raise ValueError("Corruption must be clean or family:severity[:parameter]")
    numbers = [float(v) for v in parts[1:]]
    if any(not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError("Corruption parameters must be finite and nonnegative")
    probabilities = {"impulse", "random_missing", "block_missing", "sensor_freeze",
                     "burst_impulse", "correlated_impulse", "variable_block_missing", "impulse_missing"}
    if parts[0] in probabilities and numbers[0] > 1:
        raise ValueError("Corruption rate/fraction must be between zero and one")


def deterministic_corruption(x, condition, window_ids, corruption_seed=73021):
    """CPU generation guarantees pairing even across devices and batch sizes."""
    _validate_condition(condition)
    x = x.detach().cpu()
    if condition == "clean":
        return x, torch.ones_like(x, dtype=torch.bool)
    parts = condition.split(":")
    # The legacy block generator makes a one-step block at rate zero: fix here.
    if float(parts[1]) == 0 and parts[0] in {"block_missing", "burst_impulse"}:
        return x, torch.ones_like(x, dtype=torch.bool)
    if parts[0] == "variable_block_missing" and all(float(value) == 0 for value in parts[1:]):
        return x, torch.ones_like(x, dtype=torch.bool)
    family = zlib.crc32(parts[0].encode("ascii"))
    corrupted, masks = [], []
    for i, window_id in enumerate(window_ids):
        seed = (int(corruption_seed) + family + 1000003 * int(window_id)) % (2 ** 63 - 1)
        sample, mask = corrupt_context(x[i:i + 1], condition, seed=seed)
        corrupted.append(sample)
        masks.append(mask)
    return torch.cat(corrupted, dim=0), torch.cat(masks, dim=0)


def _checked_output(output, target, target_mask):
    mean = output.mean.detach().cpu().numpy().astype(np.float64)
    std = output.std.detach().cpu().numpy().astype(np.float64)
    if mean.shape != target.shape or std.shape != target.shape:
        raise ValueError("Prediction shape must exactly equal target shape")
    if not np.isfinite(target[target_mask]).all():
        raise ValueError("An observed target is nonfinite")
    if not np.isfinite(mean[target_mask]).all() or not np.isfinite(std[target_mask]).all():
        raise FloatingPointError("Nonfinite prediction at an observed target; evaluation aborted")
    if np.any(std[target_mask] <= 0):
        raise FloatingPointError("Predictive standard deviation must be positive")
    return mean, std


def _masked_average(value, mask):
    flat_mask = mask.reshape(len(mask), -1)
    count = flat_mask.sum(axis=1)
    total = np.where(mask, value, 0).reshape(len(mask), -1).sum(axis=1, dtype=np.float64)
    return np.divide(total, count, out=np.full(len(mask), np.nan), where=count > 0)


def _window_scores(y, mean, std, mask, levels):
    # Missing cells are zeroed before arithmetic to avoid inf/nan warnings.
    y = np.where(mask, y, 0.0)
    mean = np.where(mask, mean, 0.0)
    std = np.where(mask, np.maximum(std, 1e-8), 1.0)
    error = y - mean
    z = error / std
    scores = {
        "mse": _masked_average(error ** 2, mask),
        "mae": _masked_average(np.abs(error), mask),
        "nll": _masked_average(0.5 * math.log(2 * math.pi) + np.log(std) + 0.5 * z ** 2, mask),
        "crps": _masked_average(std * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2)
                                        / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)), mask),
    }
    for level in levels:
        q = float(ndtri((1 + level) / 2))
        radius = q * std
        key = int(round(100 * level))
        scores["picp_%d" % key] = _masked_average(np.abs(error) <= radius, mask)
        scores["mpiw_%d" % key] = _masked_average(2 * radius, mask)
        scores["interval_score_%d" % key] = _masked_average(
            2 * radius + 2 / (1 - level) * np.maximum(np.abs(error) - radius, 0), mask)
    return scores


def _diagnostic_window_summaries(diagnostics, batch_size):
    result = {}
    allowed = ("reliability", "routing_reliability", "role_reliability", "router_weights",
               "residual_trust", "scale_fraction", "corruption_fraction", "context_unreliability",
               "std_base_norm", "scale_log_signal", "normalization_scale", "surprise", "missing_age")
    for key in allowed:
        value = (diagnostics or {}).get(key)
        if value is None or value.ndim < 2 or value.shape[0] != batch_size:
            continue
        array = value.detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(array).all():
            raise FloatingPointError("Nonfinite diagnostic: " + key)
        flat = array.reshape(batch_size, -1)
        result["diag_%s_mean" % key] = flat.mean(axis=1)
        for label, q in (("q05", 0.05), ("q50", 0.50), ("q95", 0.95)):
            result["diag_%s_%s" % (key, label)] = np.quantile(flat, q, axis=1)
        if key in {"router_weights", "role_reliability"}:
            for branch in range(array.shape[-1]):
                result["diag_%s_component_%d" % (key, branch)] = array[..., branch].reshape(batch_size, -1).mean(axis=1)
    return result


@torch.no_grad()
def fit_std_temperature(model, calibration_loader, device, minimum=0.05, maximum=20.0):
    """Fit one NLL-optimal multiplicative std on a reserved CLEAN holdout.

    This is a Gaussian scale diagnostic, not conformal or joint calibration. The
    caller must reserve these labels before model selection; test never enters.
    """
    metadata = ordered_window_metadata(calibration_loader)
    if not 0 < minimum < maximum:
        raise ValueError("Temperature bounds must be positive and increasing")
    model.eval()
    squared_z, count, offset = 0.0, 0, 0
    for batch in calibration_loader:
        size = len(batch["x"]) if isinstance(batch, dict) else len(batch[0])
        x, y, x_mask, y_mask, window_start = _batch_fields(batch, metadata["time_start"][offset:offset + size])
        x_mask = x_mask.bool() & torch.isfinite(x)
        x = torch.where(x_mask, x, torch.zeros_like(x))
        output = model(x.to(device), mask=x_mask.to(device), cycle_index=window_start.to(device), return_diagnostics=False)
        target = y.cpu().numpy().astype(np.float64)
        observed = y_mask.cpu().numpy().astype(bool)
        mean, std = _checked_output(output, target, observed)
        residual = ((target[observed] - mean[observed]) / std[observed]) ** 2
        squared_z += float(residual.sum(dtype=np.float64))
        count += residual.size
        offset += size
    if count == 0:
        raise ValueError("Calibration holdout has no observed targets")
    unconstrained = math.sqrt(squared_z / count)
    return {"std_temperature": float(np.clip(unconstrained, minimum, maximum)),
            "unconstrained_temperature": unconstrained, "observations": int(count),
            "windows": len(calibration_loader.dataset), "condition": "clean",
            "objective": "masked_gaussian_nll", "bounds": [minimum, maximum],
            "clipped": bool(unconstrained < minimum or unconstrained > maximum),
            "claim": "constant Gaussian scale correction; no joint/conformal coverage guarantee"}


@torch.no_grad()
def evaluate_model(model, loader, device, scaler=None, condition="clean", corruption_seed=73021,
                   std_temperature=1.0, diagnostics=True, output_dir=None,
                   save_predictions=False, levels=DEFAULT_LEVELS):
    """Return aggregate metrics and compact FULL ordered window arrays.

    Scores are observation-weighted globally. Per-window averages + target_count
    retain enough information for correctly weighted paired resampling. Windows
    with no observed targets remain present (NaN scores, zero count).
    """
    if not math.isfinite(std_temperature) or std_temperature <= 0:
        raise ValueError("std_temperature must be finite and positive")
    if any(not 0 < level < 1 for level in levels):
        raise ValueError("Coverage levels must lie in (0,1)")
    if save_predictions and output_dir is None:
        raise ValueError("save_predictions requires output_dir")
    _validate_condition(condition)
    metadata = ordered_window_metadata(loader)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        if save_predictions:
            (destination / "prediction_chunks").mkdir(exist_ok=True)
    model.eval()
    chunks = {}
    offset = 0
    examples = {}
    random_index = int(np.random.RandomState(1729).randint(max(1, len(loader.dataset))))
    # Fixed chronological indices avoid outcome-based cherry picking.
    example_indices = {0, len(loader.dataset) // 2, max(0, len(loader.dataset) - 1), random_index}
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        size = len(batch["x"]) if isinstance(batch, dict) else len(batch[0])
        x, y, x_mask, y_mask, window_start = _batch_fields(batch, metadata["time_start"][offset:offset + size])
        x, y = x.cpu(), y.cpu()
        x_mask = x_mask.cpu().bool() & torch.isfinite(x)
        y_mask = y_mask.cpu().bool()
        x = torch.where(x_mask, x, torch.zeros_like(x))
        count = len(x)
        ids = metadata["window_id"][offset:offset + count]
        corrupted, corrupted_mask = deterministic_corruption(x, condition, ids, corruption_seed)
        observed_x = x_mask & corrupted_mask
        corrupted = torch.where(observed_x, corrupted, torch.zeros_like(corrupted))
        output = model(corrupted.to(device), mask=observed_x.to(device), cycle_index=window_start.to(device), return_diagnostics=diagnostics)
        target_norm = y.numpy().astype(np.float64)
        mask = y_mask.numpy()
        mean_norm, std_norm = _checked_output(output, target_norm, mask)
        std_norm *= float(std_temperature)
        target = scaler.inverse(target_norm) if scaler is not None else target_norm
        mean = scaler.inverse(mean_norm) if scaler is not None else mean_norm
        std = scaler.inverse_std(std_norm) if scaler is not None else std_norm
        scores = _window_scores(target, mean, std, mask, levels)
        scores.update({"normalized_" + k: v for k, v in _window_scores(target_norm, mean_norm, std_norm, mask, levels).items()})
        scores["target_count"] = mask.reshape(count, -1).sum(axis=1).astype(np.int64)
        scores["target_hash"] = np.array([int.from_bytes(hashlib.blake2b(
            np.where(mask[j], target_norm[j], 0).astype("<f8").tobytes() + mask[j].tobytes(), digest_size=8).digest(), "little")
            for j in range(count)], dtype=np.uint64)
        scores["context_observed_fraction"] = observed_x.numpy().reshape(count, -1).mean(axis=1)
        scores["predictive_std_mean"] = _masked_average(std, mask)
        scores.update(_diagnostic_window_summaries(getattr(output, "diagnostics", None), count))
        for key, value in scores.items():
            chunks.setdefault(key, []).append(value)
        if save_predictions:
            # Independently readable shards avoid a huge in-memory concatenation.
            np.savez_compressed(str(destination / "prediction_chunks" / ("batch_%06d.npz" % batch_index)),
                                window_id=ids, target=target.astype(np.float32), mean=mean.astype(np.float32),
                                std=std.astype(np.float32), target_mask=mask)
        for global_index in example_indices:
            if offset <= global_index < offset + count:
                j = global_index - offset
                examples[global_index] = dict(window_id=ids[j], context=corrupted[j].numpy(),
                    context_mask=observed_x[j].numpy(), target=target_norm[j], mean=mean_norm[j],
                    std=std_norm[j], target_mask=mask[j])
                for key, tensor in (getattr(output, "diagnostics", None) or {}).items():
                    if tensor.ndim >= 2 and tensor.shape[0] == count:
                        examples[global_index]["diag_" + key] = tensor[j].detach().cpu().numpy()
        offset += count
    if offset != len(loader.dataset) or offset == 0:
        raise RuntimeError("Evaluation did not cover the complete nonempty dataset")
    arrays = dict(metadata)
    arrays.update({key: np.concatenate(value) for key, value in chunks.items()})
    counts = arrays["target_count"]
    if counts.sum() == 0:
        raise ValueError("Test dataset has no observed forecasting targets")
    valid = counts > 0
    metrics = {}
    for key in chunks:
        if key.startswith("diag_") or key in {"target_hash", "target_count"}:
            continue
        metrics[key] = float(np.sum(arrays[key][valid] * counts[valid]) / counts.sum())
    for prefix in ("", "normalized_"):
        metrics[prefix + "rmse"] = math.sqrt(metrics[prefix + "mse"])
        for level in levels:
            key = int(round(100 * level))
            metrics[prefix + "calibration_error_%d" % key] = abs(metrics[prefix + "picp_%d" % key] - level)
        for metric in ("mae", "crps"):
            values = arrays[prefix + metric][valid]
            threshold = float(np.quantile(values, 0.95))
            metrics[prefix + "sample_%s_q95" % metric] = threshold
            metrics[prefix + "sample_%s_cvar95" % metric] = float(values[values >= threshold].mean())
    metrics.update(condition=condition, corruption_seed=int(corruption_seed), std_temperature=float(std_temperature),
                   windows=int(offset), observed_targets=int(counts.sum()), zero_target_windows=int((~valid).sum()),
                   num_vars=int(target.shape[-1]), seq_len=int(x.shape[1]), pred_len=int(target.shape[1]),
                   seconds=float(time.perf_counter() - started), complete=True,
                   pairing="original-window-id, corruption-family, evaluation-seed; CPU RNG",
                   prediction_storage="full_shards" if save_predictions else "all_window_scores_plus_four_fixed_examples",
                   score_weighting="observed target cells; tails are equally weighted nonempty windows",
                   levels=list(levels))
    if destination is not None:
        np.savez_compressed(str(destination / "window_metrics.npz"), **arrays)
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
        if examples:
            example_order = sorted(examples)
            selected = {key: np.stack([examples[i][key] for i in example_order]) for key in next(iter(examples.values()))}
            selected["selection_index"] = np.asarray(example_order)
            np.savez_compressed(str(destination / "representative_cases.npz"), **selected)
    return metrics, arrays
