from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch

from evaluate_robustness import file_sha256
from kamo.data import build_dataloaders
from kamo.engine import set_seed, unpack_batch
from train import build_model, choose_device


CALIBRATION_TYPE = "global_std_temperature"


def closed_form_temperature(
    squared_standardized_error_sum: float,
    observations: int,
    lower: float,
    upper: float,
) -> Tuple[float, float]:
    """Return raw and clipped Gaussian-NLL-optimal global scale.

    For predictive standard deviations ``sigma`` and a positive multiplier
    ``T``, differentiating validation Gaussian NLL gives
    ``T^2 = mean(((y - mu) / sigma)^2)``.  The broad bounds are numerical
    safeguards, not a searched hyperparameter grid.
    """
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0.0 < lower <= upper:
        raise ValueError("temperature bounds must satisfy 0 < lower <= upper")
    value = float(squared_standardized_error_sum) / int(observations)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("standardized squared-error sum must be finite and non-negative")
    raw = math.sqrt(value)
    return raw, min(max(raw, lower), upper)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=device)


@torch.no_grad()
def fit_clean_validation_temperature(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.eval()
    squared_ratio_sum = 0.0
    log_std_sum = 0.0
    observations = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y, x_mask, y_mask = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_mask, y_mask = x_mask.to(device), y_mask.to(device)
        clean = torch.where(x_mask, x, torch.zeros_like(x))
        output = model(clean, mask=x_mask, return_diagnostics=False)
        std = output.std.clamp_min(1e-6)
        observed = y_mask & torch.isfinite(y) & torch.isfinite(output.mean) & torch.isfinite(std)
        if not torch.any(observed):
            continue
        ratio = (y - output.mean) / std
        squared_ratio_sum += float(ratio[observed].double().square().sum().cpu())
        log_std_sum += float(std[observed].double().log().sum().cpu())
        observations += int(observed.sum().item())
        batches += 1
    if observations == 0 or batches == 0:
        raise RuntimeError("Clean validation loader produced no observed targets")
    return {
        "squared_standardized_error_sum": squared_ratio_sum,
        "log_std_sum": log_std_sum,
        "observations": observations,
        "batches": batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one checkpoint-bound global standard-deviation temperature by "
            "the closed-form Gaussian-NLL optimum on clean validation only."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--temperature-min", type=float, default=0.05)
    parser.add_argument("--temperature-max", type=float, default=100.0)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.temperature_min <= args.temperature_max:
        raise SystemExit("Invalid temperature bounds")
    if args.max_batches is not None and args.max_batches <= 0:
        raise SystemExit("--max-batches must be positive")

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    for path in (config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else run_dir / "std_temperature_calibration.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = file_sha256(checkpoint_path)
    if output_path.is_file() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        reusable = (
            existing.get("calibration_type") == CALIBRATION_TYPE
            and existing.get("checkpoint_sha256") == fingerprint
            and existing.get("selection_split") == "validation"
            and existing.get("test_accessed") is False
            and existing.get("full_validation") is (args.max_batches is None)
            and float(existing.get("temperature_bounds", {}).get("minimum", -1.0))
            == float(args.temperature_min)
            and float(existing.get("temperature_bounds", {}).get("maximum", -1.0))
            == float(args.temperature_max)
        )
        if reusable:
            print(f"[reuse checkpoint-bound calibration] {output_path}", flush=True)
            print(json.dumps(existing, indent=2), flush=True)
            return

    if config.get("smoke_test") and args.max_batches is None:
        raise SystemExit("A smoke-test checkpoint cannot produce formal calibration")
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.workers is not None:
        config["workers"] = args.workers
    namespace = SimpleNamespace(**config)
    device = choose_device(args.device)
    set_seed(int(namespace.seed))
    bundle = build_dataloaders(
        namespace.dataset,
        namespace.data_root,
        namespace.seq_len,
        namespace.pred_len,
        namespace.batch_size,
        getattr(namespace, "workers", 0),
        getattr(namespace, "cache_dir", None),
    )
    model = build_model(namespace, bundle).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])

    fitted = fit_clean_validation_temperature(
        model, bundle.val, device, args.max_batches
    )
    raw, selected = closed_form_temperature(
        fitted["squared_standardized_error_sum"],
        int(fitted["observations"]),
        args.temperature_min,
        args.temperature_max,
    )
    count = int(fitted["observations"])
    mean_squared_ratio = fitted["squared_standardized_error_sum"] / count
    mean_log_std = fitted["log_std_sum"] / count
    constant = 0.5 * math.log(2.0 * math.pi)
    nll_before = constant + mean_log_std + 0.5 * mean_squared_ratio
    nll_after = (
        constant
        + mean_log_std
        + math.log(selected)
        + 0.5 * mean_squared_ratio / (selected * selected)
    )
    artifact = {
        "schema_version": 2,
        "calibration_type": CALIBRATION_TYPE,
        "selection_split": "validation",
        "fit_condition": "clean",
        "fit_metric": "gaussian_nll",
        "temperature_formula": "sqrt(mean(((target-mean)/std)^2))",
        "test_accessed": False,
        "held_out_conditions_accessed": False,
        "full_validation": args.max_batches is None,
        "debug_max_batches": args.max_batches,
        "checkpoint_sha256": fingerprint,
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "temperature_bounds": {
            "minimum": float(args.temperature_min),
            "maximum": float(args.temperature_max),
        },
        "raw_std_temperature": raw,
        "selected": {"std_temperature": selected},
        "fit_statistics_normalized_scale": {
            "observations": count,
            "batches": int(fitted["batches"]),
            "nll_before": nll_before,
            "nll_after": nll_after,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2), flush=True)
    print(f"Saved validation-only calibration: {output_path}", flush=True)


if __name__ == "__main__":
    main()
