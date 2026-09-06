from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from evaluate_robustness import file_sha256
from kamo.data import build_dataloaders
from kamo.engine import set_seed, unpack_batch
from train import build_model, choose_device


CALIBRATION_TYPE = "global_std_temperature_multimetric"
DEFAULT_TEMPERATURES = (
    0.50,
    0.67,
    0.80,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
    4.00,
    6.00,
    8.00,
    12.00,
    16.00,
    24.00,
    32.00,
    48.00,
    64.00,
)


def parse_float_list(raw: str) -> Tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one temperature is required")
    return values


def normalized_temperatures(values: Iterable[float]) -> Tuple[float, ...]:
    temperatures = tuple(sorted({round(float(value), 12) for value in values}))
    if not temperatures or temperatures[0] <= 0.0:
        raise ValueError("All temperatures must be positive")
    if 1.0 not in temperatures:
        raise ValueError("The identity temperature T=1.0 must be included")
    return temperatures


def select_temperature(
    scores: Sequence[Dict[str, float]],
    clean_tolerance: float = 0.01,
    coverage_error_tolerance: float = 0.01,
) -> Tuple[Dict[str, float], Tuple[Dict[str, float], ...]]:
    """Minimize NLL subject to clean probabilistic non-inferiority.

    T=1 is the fixed reference and is required to be present.  A candidate is
    feasible only when clean CRPS and IS95 are within ``clean_tolerance`` of
    the uncalibrated distribution and its PICP95 error is no more than one
    absolute tolerance worse.  Therefore the selector can always retain the
    native predictive distribution instead of forcing calibration.
    """
    if not 0.0 <= clean_tolerance <= 0.20:
        raise ValueError("clean_tolerance must be in [0, 0.20]")
    if not 0.0 <= coverage_error_tolerance <= 0.20:
        raise ValueError("coverage_error_tolerance must be in [0, 0.20]")
    references = [row for row in scores if math.isclose(row["temperature"], 1.0)]
    if len(references) != 1:
        raise ValueError("Exactly one T=1 reference row is required")
    reference = references[0]
    feasible = []
    for row in scores:
        values = tuple(float(row[key]) for key in ("nll", "crps", "interval_score_95", "calibration_error_95"))
        if not all(math.isfinite(value) for value in values):
            continue
        if row["crps"] > reference["crps"] * (1.0 + clean_tolerance):
            continue
        if row["interval_score_95"] > reference["interval_score_95"] * (1.0 + clean_tolerance):
            continue
        if row["calibration_error_95"] > reference["calibration_error_95"] + coverage_error_tolerance:
            continue
        feasible.append(row)
    if not feasible:
        raise RuntimeError("T=1 should make the constrained temperature set non-empty")
    selected = min(
        feasible,
        key=lambda row: (
            float(row["nll"]),
            float(row["crps"]),
            abs(float(row["temperature"]) - 1.0),
        ),
    )
    return selected, tuple(feasible)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=device)


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr

        return ndtr(values)
    except ImportError:
        return 0.5 * (
            1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0))
        )


@torch.no_grad()
def score_clean_validation(
    model: torch.nn.Module,
    loader,
    scaler,
    device: torch.device,
    temperatures: Sequence[float],
    max_batches: Optional[int],
) -> Tuple[Tuple[Dict[str, float], ...], int]:
    model.eval()
    temperature_array = np.asarray(temperatures, dtype=np.float64)[:, None]
    totals = {
        "nll": np.zeros(len(temperatures), dtype=np.float64),
        "crps": np.zeros(len(temperatures), dtype=np.float64),
        "picp_95": np.zeros(len(temperatures), dtype=np.float64),
        "mpiw_95": np.zeros(len(temperatures), dtype=np.float64),
        "interval_score_95": np.zeros(len(temperatures), dtype=np.float64),
    }
    observations = 0
    batches = 0
    z95 = 1.9599639845
    log_constant = 0.5 * math.log(2.0 * math.pi)
    pdf_constant = 1.0 / math.sqrt(2.0 * math.pi)
    crps_constant = 1.0 / math.sqrt(math.pi)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x, y, x_mask, y_mask = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_mask, y_mask = x_mask.to(device), y_mask.to(device)
        clean = torch.where(x_mask, x, torch.zeros_like(x))
        output = model(clean, mask=x_mask, return_diagnostics=False)
        target_norm = y.detach().cpu().numpy()
        mean_norm = output.mean.detach().cpu().numpy()
        std_norm = output.std.detach().cpu().numpy()
        observed = y_mask.detach().cpu().numpy().astype(bool)
        finite = observed & np.isfinite(target_norm) & np.isfinite(mean_norm) & np.isfinite(std_norm)
        if not np.any(finite):
            continue

        error = scaler.inverse(target_norm) - scaler.inverse(mean_norm)
        base_std = np.maximum(scaler.inverse_std(std_norm), 1e-6)
        error = np.asarray(error[finite], dtype=np.float64)[None, :]
        base_std = np.asarray(base_std[finite], dtype=np.float64)[None, :]
        std = np.maximum(temperature_array * base_std, 1e-6)
        standardized = error / std
        absolute_error = np.abs(error)
        width = 2.0 * z95 * std
        exceedance = np.maximum(absolute_error - z95 * std, 0.0)
        cdf = _normal_cdf(standardized)
        pdf = np.exp(-0.5 * standardized**2) * pdf_constant
        crps = std * (
            standardized * (2.0 * cdf - 1.0)
            + 2.0 * pdf
            - crps_constant
        )
        totals["nll"] += (
            log_constant + np.log(std) + 0.5 * standardized**2
        ).sum(axis=1)
        totals["crps"] += crps.sum(axis=1)
        totals["picp_95"] += (absolute_error <= z95 * std).sum(axis=1)
        totals["mpiw_95"] += width.sum(axis=1)
        totals["interval_score_95"] += (width + 40.0 * exceedance).sum(axis=1)
        observations += int(error.shape[1])
        batches += 1

    if observations == 0 or batches == 0:
        raise RuntimeError("Clean validation loader produced no observed targets")
    rows = []
    for index, temperature in enumerate(temperatures):
        row = {
            "temperature": float(temperature),
            **{key: float(value[index] / observations) for key, value in totals.items()},
        }
        row["calibration_error_95"] = abs(row["picp_95"] - 0.95)
        rows.append(row)
    return tuple(rows), batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a checkpoint-bound std temperature on clean validation by "
            "minimizing NLL subject to CRPS, IS95 and PICP95 non-inferiority."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--temperatures", type=parse_float_list, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--clean-tolerance", type=float, default=0.01)
    parser.add_argument("--coverage-error-tolerance", type=float, default=0.01)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = normalized_temperatures(args.temperatures)
    if not 0.0 <= args.clean_tolerance <= 0.20:
        raise SystemExit("--clean-tolerance must be in [0,0.20]")
    if not 0.0 <= args.coverage_error_tolerance <= 0.20:
        raise SystemExit("--coverage-error-tolerance must be in [0,0.20]")
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
        else run_dir / "std_multimetric_calibration.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = file_sha256(checkpoint_path)
    protocol = {
        "temperatures": list(temperatures),
        "clean_tolerance": float(args.clean_tolerance),
        "coverage_error_tolerance": float(args.coverage_error_tolerance),
    }
    if output_path.is_file() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        reusable = (
            existing.get("calibration_type") == CALIBRATION_TYPE
            and existing.get("checkpoint_sha256") == fingerprint
            and existing.get("selection_split") == "validation"
            and existing.get("test_accessed") is False
            and existing.get("full_validation") is (args.max_batches is None)
            and existing.get("protocol") == protocol
        )
        if reusable:
            print(f"[reuse checkpoint-bound multimetric calibration] {output_path}", flush=True)
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
    scores, batches = score_clean_validation(
        model, bundle.val, bundle.scaler, device, temperatures, args.max_batches
    )
    selected, feasible = select_temperature(
        scores, args.clean_tolerance, args.coverage_error_tolerance
    )
    reference = next(row for row in scores if math.isclose(row["temperature"], 1.0))
    artifact = {
        "schema_version": 2,
        "calibration_type": CALIBRATION_TYPE,
        "selection_split": "validation",
        "fit_condition": "clean",
        "selection_rule": (
            "minimum clean NLL subject to clean CRPS and IS95 relative "
            "non-inferiority and PICP95 absolute-error non-inferiority"
        ),
        "test_accessed": False,
        "held_out_conditions_accessed": False,
        "full_validation": args.max_batches is None,
        "debug_max_batches": args.max_batches,
        "checkpoint_sha256": fingerprint,
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "protocol": protocol,
        "validation_batches": batches,
        "reference_t1": reference,
        "feasible_temperatures": [row["temperature"] for row in feasible],
        "selected": {"std_temperature": selected["temperature"]},
        "selected_validation_metrics": selected,
        "candidate_validation_metrics": list(scores),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2), flush=True)
    print(f"Saved multimetric validation calibration: {output_path}", flush=True)


if __name__ == "__main__":
    main()
