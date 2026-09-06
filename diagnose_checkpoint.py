from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from kamo.data import build_dataloaders
from kamo.engine import unpack_batch
from train import build_model, choose_device


LEVELS = {
    "50": 0.6744897501960817,
    "60": 0.8416212335729143,
    "70": 1.0364333894937896,
    "80": 1.2815515655446004,
    "90": 1.6448536269514722,
    "95": 1.959963984540054,
    "99": 2.5758293035489004,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-test scale, NLL-tail, and calibration diagnostics for a saved checkpoint"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None, help="Override the data_root saved in config.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def finite_stats(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    values = np.asarray(values, dtype=np.float64)
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    return values[np.isfinite(values)], nan_count, inf_count


@torch.no_grad()
def main() -> None:
    cli = parse_args()
    run_dir = Path(cli.run_dir)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is required. This script reuses the trained model and does not retrain it."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["data_root"] = cli.data_root or config["data_root"]
    config["workers"] = cli.workers
    args = Namespace(**config)
    device = choose_device(cli.device)
    bundle = build_dataloaders(
        args.dataset,
        args.data_root,
        args.seq_len,
        args.pred_len,
        args.batch_size,
        args.workers,
        getattr(args, "cache_dir", None),
    )
    model = build_model(args, bundle).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rng = np.random.default_rng(20260731)
    per_batch_sample = max(1, int(math.ceil(cli.sample_size / max(1, len(bundle.test)))))
    std_samples: list[np.ndarray] = []
    nll_samples: list[np.ndarray] = []
    total = 0
    nll_sum = 0.0
    std_sum = 0.0
    std_min = math.inf
    std_max = -math.inf
    nll_min = math.inf
    nll_max = -math.inf
    std_below_1e4 = 0
    std_below_1e3 = 0
    y_sum = 0.0
    y_square_sum = 0.0
    invalid = {
        "target_nan": 0,
        "target_inf": 0,
        "mean_nan": 0,
        "mean_inf": 0,
        "std_nan": 0,
        "std_inf": 0,
        "nll_nan": 0,
        "nll_inf": 0,
    }
    covered = {key: 0 for key in LEVELS}
    width_sum = {key: 0.0 for key in LEVELS}

    for batch_index, batch in enumerate(bundle.test):
        x, y, x_mask, y_mask = unpack_batch(batch)
        x, y, x_mask = x.to(device), y.to(device), x_mask.to(device)
        output = model(x, mask=x_mask)
        observed_target = y_mask.detach().cpu().numpy().astype(bool)
        target = bundle.scaler.inverse(y.detach().cpu().numpy()).astype(np.float64)[observed_target]
        mean = bundle.scaler.inverse(output.mean.detach().cpu().numpy()).astype(np.float64)[observed_target]
        raw_std = bundle.scaler.inverse_std(output.std.detach().cpu().numpy()).astype(np.float64)[observed_target]
        std = np.maximum(raw_std, 1e-6)
        elem_nll = 0.5 * np.log(2.0 * math.pi * std**2) + 0.5 * ((target - mean) / std) ** 2

        target_finite, target_nan, target_inf = finite_stats(target)
        mean_finite, mean_nan, mean_inf = finite_stats(mean)
        std_finite, std_nan, std_inf = finite_stats(raw_std)
        nll_finite, nll_nan, nll_inf = finite_stats(elem_nll)
        invalid["target_nan"] += target_nan
        invalid["target_inf"] += target_inf
        invalid["mean_nan"] += mean_nan
        invalid["mean_inf"] += mean_inf
        invalid["std_nan"] += std_nan
        invalid["std_inf"] += std_inf
        invalid["nll_nan"] += nll_nan
        invalid["nll_inf"] += nll_inf
        if not (
            target_finite.size == target.size
            and mean_finite.size == mean.size
            and std_finite.size == raw_std.size
            and nll_finite.size == elem_nll.size
        ):
            raise FloatingPointError(
                f"Non-finite value detected in batch {batch_index}; counts are recorded in diagnostics."
            )

        elements = target.size
        total += elements
        nll_sum += float(elem_nll.sum())
        std_sum += float(raw_std.sum())
        y_sum += float(target.sum())
        y_square_sum += float(np.square(target).sum())
        std_min = min(std_min, float(raw_std.min()))
        std_max = max(std_max, float(raw_std.max()))
        nll_min = min(nll_min, float(elem_nll.min()))
        nll_max = max(nll_max, float(elem_nll.max()))
        std_below_1e4 += int((raw_std < 1e-4).sum())
        std_below_1e3 += int((raw_std < 1e-3).sum())
        for key, z in LEVELS.items():
            covered[key] += int(((target >= mean - z * std) & (target <= mean + z * std)).sum())
            width_sum[key] += float((2.0 * z * std).sum())

        flat_std = raw_std.ravel()
        flat_nll = elem_nll.ravel()
        take = min(per_batch_sample, elements)
        indices = rng.choice(elements, size=take, replace=False)
        std_samples.append(flat_std[indices].astype(np.float32))
        nll_samples.append(flat_nll[indices].astype(np.float32))
        if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(bundle.test):
            print(f"evaluated {batch_index + 1}/{len(bundle.test)} test batches", flush=True)

    sampled_std = np.concatenate(std_samples)
    sampled_nll = np.concatenate(nll_samples)
    if sampled_std.size > cli.sample_size:
        indices = rng.choice(sampled_std.size, size=cli.sample_size, replace=False)
        sampled_std = sampled_std[indices]
        sampled_nll = sampled_nll[indices]
    target_mean = y_sum / total
    target_sd = math.sqrt(max(y_square_sum / total - target_mean**2, 0.0))
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "test_batches": len(bundle.test),
        "n_elements": total,
        "quantile_sample_size": int(sampled_std.size),
        "quantile_sampling": "deterministic stratified random sample across all test batches",
        "std_min_exact": std_min,
        "std_p01_sampled": float(np.percentile(sampled_std, 1)),
        "std_p50_sampled": float(np.percentile(sampled_std, 50)),
        "std_p99_sampled": float(np.percentile(sampled_std, 99)),
        "std_max_exact": std_max,
        "std_mean_exact": std_sum / total,
        "std_below_1e-4_exact": std_below_1e4,
        "std_below_1e-3_exact": std_below_1e3,
        "nll_mean_exact": nll_sum / total,
        "nll_min_exact": nll_min,
        "nll_p50_sampled": float(np.percentile(sampled_nll, 50)),
        "nll_p95_sampled": float(np.percentile(sampled_nll, 95)),
        "nll_p99_sampled": float(np.percentile(sampled_nll, 99)),
        "nll_max_exact": nll_max,
        "target_sd_exact": target_sd,
        **invalid,
    }
    for key in LEVELS:
        nominal = int(key) / 100.0
        picp = covered[key] / total
        mpiw = width_sum[key] / total
        result[f"picp_{key}_exact"] = picp
        result[f"calibration_error_{key}_exact"] = abs(picp - nominal)
        result[f"mpiw_{key}_exact"] = mpiw
        result[f"standardized_mpiw_{key}_exact"] = mpiw / max(target_sd, 1e-12)

    output = Path(cli.output) if cli.output else run_dir / "diagnostics_full_test.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved full-test diagnostics to {output.resolve()}")


if __name__ == "__main__":
    main()
