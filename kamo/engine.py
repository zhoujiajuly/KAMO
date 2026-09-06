from __future__ import annotations

import csv
import contextlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import Standardizer
from .metrics import gaussian_crps_values, regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gaussian_nll(
    y: Tensor, mean: Tensor, std: Tensor, mask: Optional[Tensor] = None
) -> Tensor:
    std = torch.clamp(std, min=1e-4, max=1e3)
    values = torch.log(std) + 0.5 * ((y - mean) / std).square()
    if mask is None:
        return values.mean()
    observed = mask.bool() & torch.isfinite(y)
    if not torch.any(observed):
        raise ValueError("A training batch contains no observed forecasting targets")
    return values[observed].mean()


def masked_mae(y: Tensor, mean: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    values = torch.abs(y - mean)
    if mask is None:
        return values.mean()
    observed = mask.bool() & torch.isfinite(y)
    if not torch.any(observed):
        raise ValueError("A training batch contains no observed forecasting targets")
    return values[observed].mean()


def unpack_batch(batch) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Accept legacy (x, y) and naturally-missing (x, y, x_mask, y_mask) batches."""
    if len(batch) == 2:
        x, y = batch
        return x, y, torch.isfinite(x), torch.isfinite(y)
    if len(batch) == 4:
        x, y, x_mask, y_mask = batch
        return x, y, x_mask.bool(), y_mask.bool()
    raise ValueError(f"Expected a 2- or 4-tensor batch, received {len(batch)} items")


def augment_context(
    x: Tensor,
    noise_std: float = 0.0,
    missing_rate: float = 0.0,
    impulse_rate: float = 0.0,
    block_rate: float = 0.0,
    block_probability: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    mask = torch.ones_like(x, dtype=torch.bool)
    if missing_rate > 0:
        mask &= torch.rand_like(x) >= missing_rate
    if block_rate > 0 and block_probability > 0:
        length = max(1, int(round(x.shape[1] * block_rate)))
        maximum_start = max(1, x.shape[1] - length + 1)
        starts = torch.randint(maximum_start, (x.shape[0], x.shape[2]), device=x.device)
        selected = torch.rand(x.shape[0], x.shape[2], device=x.device) < block_probability
        time = torch.arange(x.shape[1], device=x.device)[None, :, None]
        in_block = (time >= starts[:, None, :]) & (time < starts[:, None, :] + length)
        mask &= ~(in_block & selected[:, None, :])
    corrupted = torch.where(mask, x, torch.zeros_like(x))
    if impulse_rate > 0:
        hits = (torch.rand_like(x) < impulse_rate) & mask
        signs = torch.where(torch.rand_like(x) < 0.5, -1.0, 1.0)
        corrupted = corrupted + hits * signs * 5.0
    if noise_std > 0:
        corrupted = corrupted + noise_std * torch.randn_like(corrupted) * mask
    return corrupted, mask


def corrupt_context(x: Tensor, condition: str, seed: int = 0) -> Tuple[Tensor, Tensor]:
    mask = torch.ones_like(x, dtype=torch.bool)
    if condition == "clean":
        return x, mask
    parts = condition.split(":")
    if len(parts) < 2:
        raise ValueError(f"Malformed corruption condition: {condition}")
    name = parts[0]
    value = float(parts[1])
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    if name == "gaussian":
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + value * noise, mask
    if name == "impulse":
        # Backward compatible: impulse:<rate> keeps the historical amplitude of
        # five standardized units.  impulse:<rate>:<amplitude> is used by the
        # paper sensitivity sweep to test the bounded-influence prediction.
        amplitude = float(parts[2]) if len(parts) >= 3 else 5.0
        if amplitude < 0:
            raise ValueError("Impulse amplitude must be non-negative")
        hits = torch.rand(x.shape, generator=generator, device=x.device) < value
        signs = torch.where(
            torch.rand(x.shape, generator=generator, device=x.device) < 0.5,
            -torch.ones_like(x),
            torch.ones_like(x),
        )
        return x + hits * signs * amplitude, mask
    if name == "random_missing":
        mask = torch.rand(x.shape, generator=generator, device=x.device) >= value
        return torch.where(mask, x, torch.zeros_like(x)), mask
    if name == "block_missing":
        length = max(1, int(round(x.shape[1] * value)))
        max_start = max(1, x.shape[1] - length + 1)
        starts = torch.randint(
            0, max_start, (x.shape[0], x.shape[2]), generator=generator, device=x.device
        )
        time = torch.arange(x.shape[1], device=x.device)[None, :, None]
        mask = ~((time >= starts[:, None, :]) & (time < starts[:, None, :] + length))
        return torch.where(mask, x, torch.zeros_like(x)), mask
    if name == "level_shift":
        if value < 0:
            raise ValueError("Level-shift magnitude must be non-negative")
        start = max(1, (2 * x.shape[1]) // 3)
        signs = torch.where(
            torch.rand((x.shape[0], 1, x.shape[2]), generator=generator, device=x.device) < 0.5,
            -torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
            torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
        )
        shifted = x.clone()
        shifted[:, start:, :] = shifted[:, start:, :] + value * signs
        return shifted, mask
    if name == "gradual_drift":
        if value < 0:
            raise ValueError("Drift magnitude must be non-negative")
        signs = torch.where(
            torch.rand((x.shape[0], 1, x.shape[2]), generator=generator, device=x.device) < 0.5,
            -torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
            torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
        )
        ramp = torch.linspace(0.0, value, x.shape[1], device=x.device, dtype=x.dtype)[None, :, None]
        return x + ramp * signs, mask
    if name == "sensor_freeze":
        # sensor_freeze:<variable_fraction>:<duration_fraction>
        duration_fraction = float(parts[2]) if len(parts) >= 3 else 0.25
        if not 0.0 <= value <= 1.0 or not 0.0 < duration_fraction <= 1.0:
            raise ValueError(
                "sensor_freeze requires variable_fraction in [0,1] and "
                "duration_fraction in (0,1]"
            )
        duration = max(1, int(round(x.shape[1] * duration_fraction)))
        maximum_start = max(2, x.shape[1] - duration + 1)
        starts = torch.randint(
            1,
            maximum_start,
            (x.shape[0], x.shape[2]),
            generator=generator,
            device=x.device,
        )
        selected = (
            torch.rand(
                (x.shape[0], x.shape[2]), generator=generator, device=x.device
            )
            < value
        )
        previous = torch.gather(x, 1, (starts - 1)[:, None, :]).squeeze(1)
        time_index = torch.arange(x.shape[1], device=x.device)[None, :, None]
        frozen_region = (
            selected[:, None, :]
            & (time_index >= starts[:, None, :])
            & (time_index < starts[:, None, :] + duration)
        )
        return torch.where(frozen_region, previous[:, None, :], x), mask
    if name == "heavy_tail":
        # Deterministic, clipped Cauchy contamination.  Unlike training impulses,
        # this produces a continuum of rare magnitudes and is reserved for testing.
        clip = float(parts[2]) if len(parts) >= 3 else 20.0
        if value < 0 or clip <= 0:
            raise ValueError("heavy_tail requires non-negative scale and positive clip")
        uniform = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        uniform = uniform.clamp(1e-5, 1.0 - 1e-5)
        noise = value * torch.tan(math.pi * (uniform - 0.5))
        return x + noise.clamp(-clip, clip), mask
    if name == "burst_impulse":
        amplitude = float(parts[2]) if len(parts) >= 3 else 5.0
        if not 0.0 <= value <= 1.0 or amplitude < 0:
            raise ValueError("burst_impulse requires rate in [0,1] and non-negative amplitude")
        length = max(1, int(round(x.shape[1] * value)))
        max_start = max(1, x.shape[1] - length + 1)
        starts = torch.randint(
            0, max_start, (x.shape[0], x.shape[2]), generator=generator, device=x.device
        )
        time = torch.arange(x.shape[1], device=x.device)[None, :, None]
        hits = (time >= starts[:, None, :]) & (time < starts[:, None, :] + length)
        signs = torch.where(
            torch.rand((x.shape[0], 1, x.shape[2]), generator=generator, device=x.device) < 0.5,
            -torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
            torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
        )
        return x + hits.to(x.dtype) * signs * amplitude, mask
    if name == "correlated_impulse":
        amplitude = float(parts[2]) if len(parts) >= 3 else 5.0
        if not 0.0 <= value <= 1.0 or amplitude < 0:
            raise ValueError("correlated_impulse requires rate in [0,1] and non-negative amplitude")
        # Hits are shared over variables, mimicking a multichannel sensor event.
        hits = torch.rand((x.shape[0], x.shape[1], 1), generator=generator, device=x.device) < value
        signs = torch.where(
            torch.rand((x.shape[0], 1, x.shape[2]), generator=generator, device=x.device) < 0.5,
            -torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
            torch.ones((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype),
        )
        return x + hits.to(x.dtype) * signs * amplitude, mask
    if name == "variable_block_missing":
        maximum = float(parts[2]) if len(parts) >= 3 else value
        if not 0.0 <= value <= maximum <= 1.0:
            raise ValueError("variable_block_missing requires 0 <= min_rate <= max_rate <= 1")
        minimum_length = max(1, int(round(x.shape[1] * value)))
        maximum_length = max(minimum_length, int(round(x.shape[1] * maximum)))
        lengths = torch.randint(
            minimum_length,
            maximum_length + 1,
            (x.shape[0], x.shape[2]),
            generator=generator,
            device=x.device,
        )
        start_unit = torch.rand((x.shape[0], x.shape[2]), generator=generator, device=x.device)
        starts = torch.floor(start_unit * (x.shape[1] - lengths + 1).to(x.dtype)).long()
        time = torch.arange(x.shape[1], device=x.device)[None, :, None]
        mask = ~((time >= starts[:, None, :]) & (time < starts[:, None, :] + lengths[:, None, :]))
        return torch.where(mask, x, torch.zeros_like(x)), mask
    if name == "impulse_missing":
        missing_rate = float(parts[2]) if len(parts) >= 3 else value
        amplitude = float(parts[3]) if len(parts) >= 4 else 5.0
        if not 0.0 <= value <= 1.0 or not 0.0 <= missing_rate <= 1.0 or amplitude < 0:
            raise ValueError("impulse_missing requires valid rates and non-negative amplitude")
        mask = torch.rand(x.shape, generator=generator, device=x.device) >= missing_rate
        hits = (torch.rand(x.shape, generator=generator, device=x.device) < value) & mask
        signs = torch.where(
            torch.rand(x.shape, generator=generator, device=x.device) < 0.5,
            -torch.ones_like(x),
            torch.ones_like(x),
        )
        corrupted = torch.where(mask, x, torch.zeros_like(x)) + hits.to(x.dtype) * signs * amplitude
        return corrupted, mask
    raise ValueError(f"Unknown corruption condition: {condition}")


def _inverse_batch(
    scaler: Standardizer, y: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return scaler.inverse(y), scaler.inverse(mean), scaler.inverse_std(std)


def _sample_masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    mask = np.asarray(mask, dtype=bool).reshape(mask.shape[0], -1)
    counts = mask.sum(axis=1)
    valid = counts > 0
    sums = np.where(mask, values, 0.0).sum(axis=1)
    return sums[valid] / counts[valid]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: Standardizer,
    device: torch.device,
    condition: str = "clean",
    corruption_seed: int = 123,
    max_batches: Optional[int] = None,
    save_batches: int = 4,
    diagnostics: bool = False,
    std_temperature: float = 1.0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    if not math.isfinite(std_temperature) or std_temperature <= 0:
        raise ValueError("std_temperature must be finite and positive")
    model.eval()
    totals: Dict[str, float] = {}
    normalized_totals: Dict[str, float] = {}
    count = 0
    saved = {
        "target": [], "mean": [], "std": [], "context": [],
        "target_mask": [], "context_mask": [],
    }
    diagnostic_out: Dict[str, np.ndarray] = {}
    sample_mae_values = []
    sample_crps_values = []
    start_time = time.perf_counter()
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y, x_mask, y_mask = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_mask, y_mask = x_mask.to(device), y_mask.to(device)
        x_corrupt, corruption_mask = corrupt_context(
            x, condition, seed=corruption_seed + batch_index
        )
        mask = x_mask & corruption_mask
        x_corrupt = torch.where(mask, x_corrupt, torch.zeros_like(x_corrupt))
        output = model(
            x_corrupt,
            mask=mask,
            return_diagnostics=diagnostics and batch_index == 0,
        )
        target_norm = y.detach().cpu().numpy()
        mean_norm = output.mean.detach().cpu().numpy()
        std_norm = output.std.detach().cpu().numpy() * float(std_temperature)
        target_mask = y_mask.detach().cpu().numpy().astype(bool)
        normalized = regression_metrics(
            target_norm, mean_norm, std_norm, target_mask
        )
        target, mean, std = _inverse_batch(scaler, target_norm, mean_norm, std_norm)
        metrics = regression_metrics(target, mean, std, target_mask)
        elements = int(target_mask.sum())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * elements
        for key, value in normalized.items():
            normalized_totals[key] = normalized_totals.get(key, 0.0) + value * elements
        count += elements
        batches += 1
        absolute_error = np.abs(mean - target)
        crps_values = gaussian_crps_values(target, mean, std)
        sample_mae_values.append(_sample_masked_mean(absolute_error, target_mask))
        sample_crps_values.append(_sample_masked_mean(crps_values, target_mask))
        if batch_index < save_batches:
            saved["target"].append(target.astype(np.float32))
            saved["mean"].append(mean.astype(np.float32))
            saved["std"].append(std.astype(np.float32))
            saved["context"].append(
                scaler.inverse(x.detach().cpu().numpy()).astype(np.float32)
            )
            saved["target_mask"].append(target_mask)
            saved["context_mask"].append(x_mask.detach().cpu().numpy().astype(bool))
        if output.diagnostics is not None:
            for key, value in output.diagnostics.items():
                diagnostic_out[key] = value.detach().cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start_time
    if count == 0:
        raise RuntimeError("Evaluation loader produced no batches")
    result = {key: value / count for key, value in totals.items()}
    normalized_result = {key: value / count for key, value in normalized_totals.items()}
    # Nonlinear derived metrics must be computed after global aggregation.
    result["rmse"] = math.sqrt(result["mse"])
    normalized_result["rmse"] = math.sqrt(normalized_result["mse"])
    for level in (80, 90, 95, 99):
        result[f"calibration_error_{level}"] = abs(result[f"picp_{level}"] - level / 100.0)
        normalized_result[f"calibration_error_{level}"] = abs(normalized_result[f"picp_{level}"] - level / 100.0)
    for label, chunks in (
        ("sample_mae", sample_mae_values),
        ("sample_crps", sample_crps_values),
    ):
        values = np.concatenate(chunks) if chunks else np.empty(0)
        if values.size:
            threshold = float(np.quantile(values, 0.95))
            result[f"{label}_q95"] = threshold
            result[f"{label}_cvar95"] = float(values[values >= threshold].mean())
    result.update({f"normalized_{key}": value for key, value in normalized_result.items()})
    result["seconds"] = elapsed
    result["batches"] = float(batches)
    result["std_temperature"] = float(std_temperature)
    arrays = {
        key: np.concatenate(value, axis=0) for key, value in saved.items() if value
    }
    arrays.update({f"diag_{key}": value for key, value in diagnostic_out.items()})
    return result, arrays


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: Standardizer,
    device: torch.device,
    output_dir: Path,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    grad_clip: float = 1.0,
    train_noise: float = 0.02,
    train_missing: float = 0.05,
    train_impulse: float = 0.01,
    train_block: float = 0.10,
    train_block_prob: float = 0.25,
    mae_weight: float = 0.1,
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = model.optimizer_groups(learning_rate) if hasattr(model, "optimizer_groups") else model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    use_amp = device.type == "cuda"
    amp = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    best_crps = math.inf
    best_epoch = 0
    stale = 0
    history = []
    checkpoint = output_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch_index, batch in enumerate(progress):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            x, y, x_mask, y_mask = unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            x_mask, y_mask = x_mask.to(device), y_mask.to(device)
            x_aug, augmentation_mask = augment_context(
                x,
                noise_std=train_noise,
                missing_rate=train_missing,
                impulse_rate=train_impulse,
                block_rate=train_block,
                block_probability=train_block_prob,
            )
            mask = x_mask & augmentation_mask
            x_aug = torch.where(mask, x_aug, torch.zeros_like(x_aug))
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
            with amp_context:
                output = model(x_aug, mask=mask)
                loss = gaussian_nll(y, output.mean, output.std, y_mask)
                loss = loss + mae_weight * masked_mae(y, output.mean, y_mask)
                if output.auxiliary_loss is not None:
                    loss = loss + output.auxiliary_loss
            if use_amp:
                amp.scale(loss).backward()
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        val_metrics, _ = evaluate(
            model,
            val_loader,
            scaler,
            device,
            max_batches=max_val_batches,
            save_batches=0,
        )
        train_loss = float(np.mean(losses)) if losses else math.inf
        scheduler.step(val_metrics["crps"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_nll": val_metrics["nll"],
            "val_crps": val_metrics["crps"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={train_loss:.4f} "
            f"val_MAE={row['val_mae']:.4f} val_CRPS={row['val_crps']:.4f}"
        )
        if val_metrics["crps"] < best_crps:
            best_crps = val_metrics["crps"]
            best_epoch = epoch
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return {
        "best_epoch": best_epoch,
        "best_val_crps": best_crps,
        "history": history,
        "checkpoint": str(checkpoint),
    }


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
