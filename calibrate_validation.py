from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from kamo.data import build_dataloaders
from kamo.engine import corrupt_context, set_seed, unpack_batch
from kamo.metrics import regression_metrics
from kamo.model import KAMOV5
from train import build_model, choose_device


# These are deliberately limited to corruptions already represented by the
# training augmentation.  Distribution-shift, freezing and heavy-tail tests are
# held out and must never be used to choose the controls below.
DEFAULT_VALIDATION_CONDITIONS = (
    "clean",
    "gaussian:0.1",
    "impulse:0.05:5.0",
    "random_missing:0.1",
    "block_missing:0.2",
)
HELD_OUT_TEST_CONDITIONS = (
    "level_shift:2.0",
    "gradual_drift:2.0",
    "sensor_freeze:0.30:0.25",
    "heavy_tail:0.20:20.0",
)
METRIC_KEYS = (
    "mae",
    "nll",
    "crps",
    "picp_95",
    "mpiw_95",
    "interval_score_95",
)


@dataclass(frozen=True)
class Candidate:
    reliability_strength: float
    scale_adjustment_coefficient: float
    uncertainty_reliability_strength: float

    @property
    def name(self) -> str:
        return (
            f"r{self.reliability_strength:.3f}_"
            f"a{self.scale_adjustment_coefficient:.3f}_"
            f"u{self.uncertainty_reliability_strength:.3f}"
        )


@dataclass(frozen=True)
class ScoredCandidate:
    structural: Candidate
    temperature_mode: str
    std_temperature: float
    raw_nll_temperature: float

    @property
    def name(self) -> str:
        return f"{self.structural.name}_t{self.temperature_mode}"


def parse_float_list(raw: str) -> Tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select KAMO reliability strength and uncertainty calibration using "
            "validation data only; the test loader is never iterated or used for selection."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--conditions", nargs="+", default=list(DEFAULT_VALIDATION_CONDITIONS)
    )
    parser.add_argument(
        "--reliability-strengths", type=parse_float_list, default=(0.50, 0.75, 1.0)
    )
    parser.add_argument(
        "--uncertainty-strengths", type=parse_float_list, default=(0.0, 0.25, 0.5, 1.0)
    )
    parser.add_argument(
        "--scale-coefficients",
        type=parse_float_list,
        default=None,
        help="Default is 0, half the checkpoint coefficient, and the checkpoint coefficient.",
    )
    parser.add_argument("--clean-tolerance", type=float, default=0.01)
    parser.add_argument("--probability-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-crps-improvement", type=float, default=0.001)
    parser.add_argument("--temperature-fit-fraction", type=float, default=0.50)
    parser.add_argument("--temperature-min", type=float, default=0.25)
    parser.add_argument("--temperature-max", type=float, default=4.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debug only. Artifacts produced with this option are rejected by test evaluation.",
    )
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=device)


def candidate_std(
    base_std: np.ndarray,
    scale_signal: np.ndarray,
    corruption_fraction: np.ndarray,
    uncertainty_gain: float,
    candidate: Candidate,
    temperature: float = 1.0,
) -> np.ndarray:
    std = base_std * np.exp(
        candidate.scale_adjustment_coefficient * scale_signal
    )
    std = std * (
        1.0
        + uncertainty_gain
        * candidate.uncertainty_reliability_strength
        * corruption_fraction
    )
    return np.maximum(std * temperature, 1e-6)


@torch.no_grad()
def batches_for_condition(
    model: KAMOV5,
    loader,
    scaler,
    device: torch.device,
    condition: str,
    start_batch: int,
    stop_batch: Optional[int],
):
    model.eval()
    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue
        if stop_batch is not None and batch_index >= stop_batch:
            break
        x, y, x_mask, y_mask = unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_mask, y_mask = x_mask.to(device), y_mask.to(device)
        x_corrupt, corruption_mask = corrupt_context(
            x, condition, seed=123 + batch_index
        )
        observed = x_mask & corruption_mask
        x_corrupt = torch.where(observed, x_corrupt, torch.zeros_like(x_corrupt))
        output = model(x_corrupt, mask=observed, return_diagnostics=True)
        diagnostics = output.diagnostics
        if diagnostics is None:
            raise RuntimeError("KAMOV5 diagnostics are required for calibration")

        target_norm = y.detach().cpu().numpy()
        mean_norm = output.mean.detach().cpu().numpy()
        target = scaler.inverse(target_norm)
        mean = scaler.inverse(mean_norm)
        internal_scale = diagnostics["normalization_scale"].detach().cpu().numpy()
        base_std_norm = diagnostics["std_base_norm"].detach().cpu().numpy()
        base_std = scaler.inverse_std(base_std_norm * internal_scale)
        scale_signal = diagnostics["scale_log_signal"].detach().cpu().numpy()
        corruption_fraction = diagnostics["corruption_fraction"].detach().cpu().numpy()
        gain_tensor = diagnostics["uncertainty_gain"].detach().cpu().numpy()
        uncertainty_gain = float(np.asarray(gain_tensor).reshape(-1)[0])
        target_mask = y_mask.detach().cpu().numpy().astype(bool)
        yield (
            target,
            mean,
            base_std,
            scale_signal,
            corruption_fraction,
            uncertainty_gain,
            target_mask,
        )


def fit_clean_temperatures(
    model: KAMOV5,
    loader,
    scaler,
    device: torch.device,
    candidates: Sequence[Candidate],
    reliability_strength: float,
    temperature_bounds: Tuple[float, float],
    stop_batch: int,
) -> Dict[str, Dict[str, float]]:
    model.reliability_strength = reliability_strength
    sums = {candidate.name: 0.0 for candidate in candidates}
    counts = {candidate.name: 0 for candidate in candidates}
    for values in batches_for_condition(
        model, loader, scaler, device, "clean", 0, stop_batch
    ):
        target, mean, base_std, signal, fraction, gain, mask = values
        error = target - mean
        for candidate in candidates:
            std = candidate_std(base_std, signal, fraction, gain, candidate)
            observed = mask & np.isfinite(error) & np.isfinite(std)
            sums[candidate.name] += float(np.square(error[observed] / std[observed]).sum())
            counts[candidate.name] += int(observed.sum())

    fitted: Dict[str, Dict[str, float]] = {}
    lower, upper = temperature_bounds
    for candidate in candidates:
        count = counts[candidate.name]
        if count == 0:
            raise RuntimeError("Clean validation loader produced no observed targets")
        raw = math.sqrt(sums[candidate.name] / count)
        fitted[candidate.name] = {
            "raw": raw,
            "selected": min(max(raw, lower), upper),
            "observations": count,
        }
    return fitted


def score_candidates(
    model: KAMOV5,
    loader,
    scaler,
    device: torch.device,
    candidates: Sequence[ScoredCandidate],
    conditions: Sequence[str],
    reliability_strength: float,
    start_batch: int,
    stop_batch: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    model.reliability_strength = reliability_strength
    totals = {}
    for candidate in candidates:
        totals[candidate.name] = {}
        for condition in conditions:
            bucket = {key: 0.0 for key in METRIC_KEYS}
            bucket["count"] = 0.0
            totals[candidate.name][condition] = bucket
    for condition in conditions:
        batches = 0
        for values in batches_for_condition(
            model, loader, scaler, device, condition, start_batch, stop_batch
        ):
            batches += 1
            target, mean, base_std, signal, fraction, gain, mask = values
            count = int(mask.sum())
            for candidate in candidates:
                std = candidate_std(
                    base_std,
                    signal,
                    fraction,
                    gain,
                    candidate.structural,
                    candidate.std_temperature,
                )
                metrics = regression_metrics(target, mean, std, mask)
                bucket = totals[candidate.name][condition]
                for key in METRIC_KEYS:
                    bucket[key] += float(metrics[key]) * count
                bucket["count"] += count
        if batches == 0:
            raise RuntimeError(f"Validation loader produced no batches for {condition}")

    for candidate in candidates:
        for condition in conditions:
            bucket = totals[candidate.name][condition]
            count = bucket.pop("count")
            for key in METRIC_KEYS:
                bucket[key] /= count
            bucket["calibration_error_95"] = abs(bucket["picp_95"] - 0.95)
    return totals


def unique_sorted(values: Iterable[float]) -> Tuple[float, ...]:
    return tuple(sorted({round(float(value), 12) for value in values}))


def main() -> None:
    args = parse_args()
    if "clean" not in args.conditions:
        raise SystemExit("--conditions must include clean")
    forbidden = sorted(set(args.conditions) & set(HELD_OUT_TEST_CONDITIONS))
    if forbidden:
        raise SystemExit(
            "Held-out test corruptions cannot be used for calibration: "
            + ", ".join(forbidden)
        )
    if not 0.0 <= args.clean_tolerance <= 0.20:
        raise SystemExit("--clean-tolerance must be in [0,0.20]")
    if not 0.0 <= args.probability_tolerance <= 0.20:
        raise SystemExit("--probability-tolerance must be in [0,0.20]")
    if not 0.0 <= args.minimum_crps_improvement <= 0.20:
        raise SystemExit("--minimum-crps-improvement must be in [0,0.20]")
    if not 0.1 <= args.temperature_fit_fraction <= 0.9:
        raise SystemExit("--temperature-fit-fraction must be in [0.1,0.9]")
    if not 0 < args.temperature_min <= args.temperature_max:
        raise SystemExit("Invalid temperature bounds")

    run_dir = Path(args.run_dir).resolve()
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best.pt"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not str(config.get("model", "")).startswith("kamo_v5"):
        raise SystemExit("Structural calibration currently supports KAMO-v5 runs only")
    if config.get("smoke_test") and args.max_batches is None:
        raise SystemExit("A smoke-test checkpoint cannot produce a paper calibration")

    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.workers is not None:
        config["workers"] = args.workers
    namespace = SimpleNamespace(**config)
    device = choose_device(args.device)
    set_seed(int(namespace.seed))
    # build_dataloaders returns all split loaders, but the returned test loader is
    # intentionally never iterated or inspected in this program.
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
    if not isinstance(model, KAMOV5):
        raise RuntimeError("Expected build_model to return KAMOV5")

    default_coefficient = float(
        getattr(namespace, "scale_adjustment_coefficient", 0.2)
    )
    scale_coefficients = (
        unique_sorted((0.0, 0.5 * default_coefficient, default_coefficient))
        if args.scale_coefficients is None
        else unique_sorted(args.scale_coefficients)
    )
    reliability_strengths = unique_sorted(args.reliability_strengths)
    uncertainty_strengths = unique_sorted(args.uncertainty_strengths)
    for label, values in (
        ("reliability", reliability_strengths),
        ("uncertainty", uncertainty_strengths),
    ):
        if min(values) < 0 or max(values) > 1:
            raise SystemExit(f"{label} strengths must be in [0,1]")
    if min(scale_coefficients) < 0:
        raise SystemExit("Scale coefficients must be non-negative")

    total_batches = len(bundle.val)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    if total_batches < 2:
        raise SystemExit("Calibration requires at least two validation batches")
    fit_stop = max(1, min(total_batches - 1, int(total_batches * args.temperature_fit_fraction)))
    selection_start = fit_stop

    all_candidates: List[ScoredCandidate] = []
    all_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for reliability_strength in reliability_strengths:
        structural_candidates = [
            Candidate(reliability_strength, coefficient, uncertainty_strength)
            for coefficient in scale_coefficients
            for uncertainty_strength in uncertainty_strengths
        ]
        temperatures = fit_clean_temperatures(
            model,
            bundle.val,
            bundle.scaler,
            device,
            structural_candidates,
            reliability_strength,
            (args.temperature_min, args.temperature_max),
            fit_stop,
        )
        candidates = []
        for structural in structural_candidates:
            fitted = temperatures[structural.name]
            candidates.append(
                ScoredCandidate(structural, "identity", 1.0, fitted["raw"])
            )
            candidates.append(
                ScoredCandidate(
                    structural, "nll", fitted["selected"], fitted["raw"]
                )
            )
        scores = score_candidates(
            model,
            bundle.val,
            bundle.scaler,
            device,
            candidates,
            args.conditions,
            reliability_strength,
            selection_start,
            total_batches,
        )
        all_candidates.extend(candidates)
        all_scores.update(scores)

    reference = min(
        all_candidates,
        key=lambda candidate: (
            abs(candidate.structural.reliability_strength - 1.0)
            + abs(candidate.structural.scale_adjustment_coefficient - default_coefficient)
            + abs(candidate.structural.uncertainty_reliability_strength - 1.0)
            + (0.0 if candidate.temperature_mode == "identity" else 10.0)
        ),
    )
    corrupt_conditions = [value for value in args.conditions if value != "clean"]
    rows = []
    for candidate in all_candidates:
        condition_scores = all_scores[candidate.name]
        corrupt_crps = float(
            np.mean([condition_scores[value]["crps"] for value in corrupt_conditions])
        )
        corrupt_interval = float(
            np.mean(
                [condition_scores[value]["interval_score_95"] for value in corrupt_conditions]
            )
        )
        corrupt_calibration = float(
            np.mean(
                [condition_scores[value]["calibration_error_95"] for value in corrupt_conditions]
            )
        )
        corrupt_nll = float(
            np.mean([condition_scores[value]["nll"] for value in corrupt_conditions])
        )
        clean_crps = condition_scores["clean"]["crps"]
        clean_mae = condition_scores["clean"]["mae"]
        structural = candidate.structural
        row: Dict[str, object] = {
            "candidate": candidate.name,
            "reliability_strength": structural.reliability_strength,
            "scale_adjustment_coefficient": structural.scale_adjustment_coefficient,
            "uncertainty_reliability_strength": structural.uncertainty_reliability_strength,
            "temperature_mode": candidate.temperature_mode,
            "std_temperature": candidate.std_temperature,
            "raw_nll_temperature": candidate.raw_nll_temperature,
            "clean_crps": clean_crps,
            "clean_mae": clean_mae,
            "clean_picp_95": condition_scores["clean"]["picp_95"],
            "clean_interval_score_95": condition_scores["clean"]["interval_score_95"],
            "known_corruption_mean_crps": corrupt_crps,
            "known_corruption_mean_nll": corrupt_nll,
            "known_corruption_mean_interval_score_95": corrupt_interval,
            "known_corruption_mean_calibration_error_95": corrupt_calibration,
        }
        rows.append(row)

    row_by_name = {str(row["candidate"]): row for row in rows}
    reference_row = row_by_name[reference.name]
    eligible: List[Dict[str, object]] = []
    for row in rows:
        clean_ok = bool(
            row["clean_crps"] <= reference_row["clean_crps"] * (1.0 + args.clean_tolerance)
            and row["clean_mae"] <= reference_row["clean_mae"] * (1.0 + args.clean_tolerance)
        )
        probability_ok = bool(
            row["known_corruption_mean_nll"]
            <= reference_row["known_corruption_mean_nll"] * (1.0 + args.probability_tolerance)
            and row["known_corruption_mean_interval_score_95"]
            <= reference_row["known_corruption_mean_interval_score_95"]
            * (1.0 + args.probability_tolerance)
            and row["known_corruption_mean_calibration_error_95"]
            <= reference_row["known_corruption_mean_calibration_error_95"] + 0.002
        )
        row["clean_noninferiority"] = clean_ok
        row["probability_noninferiority"] = probability_ok
        if clean_ok and probability_ok:
            eligible.append(row)
    if not eligible:
        raise RuntimeError("Internal error: identity reference candidate is not eligible")
    best = min(
        eligible,
        key=lambda row: (
            row["known_corruption_mean_crps"],
            row["known_corruption_mean_interval_score_95"],
            row["known_corruption_mean_calibration_error_95"],
        ),
    )
    required_crps = reference_row["known_corruption_mean_crps"] * (
        1.0 - args.minimum_crps_improvement
    )
    selected = best if best["known_corruption_mean_crps"] <= required_crps else reference_row
    selected_is_reference = selected["candidate"] == reference.name
    selected["recommended_action"] = (
        "retain_uncalibrated" if selected_is_reference else "apply_validation_selected_controls"
    )

    output = Path(args.output).resolve() if args.output else run_dir / "calibration.json"
    artifact = {
        "schema_version": 2,
        "dataset": namespace.dataset,
        "model": namespace.model,
        "seed": int(namespace.seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "selection_split": "validation",
        "temperature_fit_metric": "early_validation_clean_gaussian_nll",
        "structural_selection_metric": "late_validation_known_corruption_mean_crps",
        "validation_partition": {
            "order": "chronological_batches",
            "total_batches": total_batches,
            "temperature_fit_batches": [0, fit_stop],
            "selection_batches": [selection_start, total_batches],
        },
        "clean_noninferiority_tolerance": args.clean_tolerance,
        "probability_noninferiority_tolerance": args.probability_tolerance,
        "minimum_crps_improvement": args.minimum_crps_improvement,
        "validation_conditions": list(args.conditions),
        "held_out_test_conditions": list(HELD_OUT_TEST_CONDITIONS),
        "test_accessed": False,
        "full_validation": args.max_batches is None,
        "reference_candidate": reference.name,
        "reference_clean_crps": reference_row["clean_crps"],
        "reference_clean_mae": reference_row["clean_mae"],
        "recommended_action": selected["recommended_action"],
        "selected": selected,
        "candidates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    pd.DataFrame(rows).sort_values(
        ["clean_noninferiority", "known_corruption_mean_crps"],
        ascending=[False, True],
    ).to_csv(csv_path, index=False)
    print(json.dumps(artifact["selected"], indent=2, ensure_ascii=False))
    print(f"Saved validation-only calibration: {output}")
    print(f"Saved candidate audit table: {csv_path}")


if __name__ == "__main__":
    main()
