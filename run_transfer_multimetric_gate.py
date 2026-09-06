from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from run_transfer_soft_gate import (
    KNOWN_VALIDATION_CONDITIONS,
    mean_metric,
    relative,
    require_checkpoint,
    run,
)
from run_transfer_temperature_gate import calibration_error


def load_temperature(run_dir: Path) -> float:
    artifact = json.loads(
        (run_dir / "std_multimetric_calibration.json").read_text(encoding="utf-8")
    )
    return float(artifact["selected"]["std_temperature"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only transfer gate using the same constrained multimetric "
            "standard-deviation calibration protocol for every model."
        )
    )
    parser.add_argument("--dataset", choices=("ETTh2", "weather"), required=True)
    parser.add_argument("--backbone", choices=("patchtst", "itransformer"), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    runs_root = Path(args.runs_root).resolve()
    results_root = Path(args.results_root).resolve()
    seed = int(args.seed)
    ordinary = args.backbone
    robust = f"{args.backbone}_robust"
    soft = f"{args.backbone}_reliability_soft"
    control_name = f"revision_transfer_s{seed}_h96"
    soft_name = f"revision_transfer_soft_gate_s{seed}_h96"
    run_dirs = {
        ordinary: runs_root / args.dataset / ordinary / control_name,
        robust: runs_root / args.dataset / robust / control_name,
        soft: runs_root / args.dataset / soft / soft_name,
    }
    for model, run_dir in run_dirs.items():
        require_checkpoint(run_dir, model)
        run(
            [
                args.python,
                str(code_root / "calibrate_std_multimetric.py"),
                "--run-dir", str(run_dir),
                "--data-root", str(data_root),
                "--device", args.device,
                "--workers", str(args.workers),
            ],
            args.dry_run,
        )
    if args.dry_run:
        return

    output_dir = results_root / "validation_transfer_multimetric_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.dataset}_{args.backbone}_s{seed}.csv"
    command = [args.python, str(code_root / "evaluate_robustness.py")]
    for run_dir in run_dirs.values():
        command.extend(["--run-dir", str(run_dir)])
    command.extend(
        [
            "--data-root", str(data_root),
            "--device", args.device,
            "--calibration", "std-multimetric-auto",
            "--split", "validation",
            "--conditions", *KNOWN_VALIDATION_CONDITIONS,
            "--output", str(csv_path),
        ]
    )
    run(command, args.dry_run)

    frame = pd.read_csv(csv_path)
    if set(frame.get("evaluation_split", [])) != {"validation"}:
        raise RuntimeError("Gate output is not validation-only")
    if set(frame.get("calibration_type", [])) != {
        "global_std_temperature_multimetric"
    }:
        raise RuntimeError("The common multimetric calibration was not applied")
    known = KNOWN_VALIDATION_CONDITIONS[1:]
    clean_crps_plain = mean_metric(frame, ordinary, ("clean",), "crps")
    clean_crps_soft = mean_metric(frame, soft, ("clean",), "crps")
    known_crps_plain = mean_metric(frame, ordinary, known, "crps")
    known_crps_robust = mean_metric(frame, robust, known, "crps")
    known_crps_soft = mean_metric(frame, soft, known, "crps")
    known_mae_plain = mean_metric(frame, ordinary, known, "mae")
    known_mae_soft = mean_metric(frame, soft, known, "mae")
    clean_nll_plain = mean_metric(frame, ordinary, ("clean",), "nll")
    clean_nll_soft = mean_metric(frame, soft, ("clean",), "nll")
    known_nll_plain = mean_metric(frame, ordinary, known, "nll")
    known_nll_soft = mean_metric(frame, soft, known, "nll")
    clean_is_plain = mean_metric(frame, ordinary, ("clean",), "interval_score_95")
    clean_is_soft = mean_metric(frame, soft, ("clean",), "interval_score_95")
    known_is_plain = mean_metric(frame, ordinary, known, "interval_score_95")
    known_is_soft = mean_metric(frame, soft, known, "interval_score_95")
    clean_cal_plain = calibration_error(frame, ordinary, ("clean",))
    clean_cal_soft = calibration_error(frame, soft, ("clean",))
    known_cal_plain = calibration_error(frame, ordinary, known)
    known_cal_soft = calibration_error(frame, soft, known)

    clean_crps_delta = relative(clean_crps_soft, clean_crps_plain)
    known_crps_delta = relative(known_crps_soft, known_crps_plain)
    known_mae_delta = relative(known_mae_soft, known_mae_plain)
    criteria = {
        "clean_crps_noninferior_within_1pct": clean_crps_delta <= 0.01,
        "known_crps_improves_over_plain": known_crps_delta < 0.0,
        "known_mae_noninferior_within_1pct": known_mae_delta <= 0.01,
        "clean_nll_no_instability": clean_nll_soft
        <= clean_nll_plain + max(0.10, 0.10 * abs(clean_nll_plain)),
        "known_nll_no_instability": known_nll_soft
        <= known_nll_plain + max(0.10, 0.10 * abs(known_nll_plain)),
        "clean_picp95_error_noninferior_within_1point": clean_cal_soft
        <= clean_cal_plain + 0.01,
        "known_picp95_error_noninferior_within_1point": known_cal_soft
        <= known_cal_plain + 0.01,
        "clean_is95_noninferior_within_1pct": relative(clean_is_soft, clean_is_plain)
        <= 0.01,
        "known_is95_noninferior_within_1pct": relative(known_is_soft, known_is_plain)
        <= 0.01,
    }
    metrics = {
        "std_temperatures": {
            model: load_temperature(run_dir) for model, run_dir in run_dirs.items()
        },
        "clean_crps_plain": clean_crps_plain,
        "clean_crps_soft": clean_crps_soft,
        "clean_crps_soft_relative_delta": clean_crps_delta,
        "known_crps_plain": known_crps_plain,
        "known_crps_hard_robust": known_crps_robust,
        "known_crps_soft": known_crps_soft,
        "known_crps_soft_relative_delta_vs_plain": known_crps_delta,
        "known_crps_soft_relative_delta_vs_hard_robust": relative(
            known_crps_soft, known_crps_robust
        ),
        "known_mae_soft_relative_delta_vs_plain": known_mae_delta,
        "clean_nll_plain": clean_nll_plain,
        "clean_nll_soft": clean_nll_soft,
        "known_nll_plain": known_nll_plain,
        "known_nll_soft": known_nll_soft,
        "clean_picp95_error_plain": clean_cal_plain,
        "clean_picp95_error_soft": clean_cal_soft,
        "known_picp95_error_plain": known_cal_plain,
        "known_picp95_error_soft": known_cal_soft,
        "clean_interval_score_95_plain": clean_is_plain,
        "clean_interval_score_95_soft": clean_is_soft,
        "known_interval_score_95_plain": known_is_plain,
        "known_interval_score_95_soft": known_is_soft,
    }
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("Non-finite gate metric")
    report = {
        "schema_version": 3,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": seed,
        "selection_split": "validation",
        "calibration_fit_split": "validation_clean",
        "calibration_protocol": (
            "same constrained clean-validation multimetric std-temperature "
            "selection for every model"
        ),
        "test_accessed_by_gate": False,
        "held_out_conditions_accessed": False,
        "known_validation_conditions": list(KNOWN_VALIDATION_CONDITIONS),
        "models": {"plain": ordinary, "hard_robust": robust, "soft_reliability": soft},
        "metrics": metrics,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "warning": (
            "This is a validation engineering gate, not inferential evidence. "
            "The candidate grid and selected temperatures are frozen before test."
        ),
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved multimetric gate decision: {json_path}", flush=True)


if __name__ == "__main__":
    main()
