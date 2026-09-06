from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_revision_suite import TrainJob, model_defaults


KNOWN_VALIDATION_CONDITIONS = (
    "clean",
    "gaussian:0.1",
    "gaussian:0.3",
    "impulse:0.05:5.0",
    "random_missing:0.1",
    "random_missing:0.3",
    "block_missing:0.2",
)


def run(command: list[str], dry_run: bool = False) -> None:
    display = (
        shlex.join(command)
        if hasattr(shlex, "join")
        else " ".join(shlex.quote(value) for value in command)
    )
    print(display, flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def require_checkpoint(path: Path, label: str) -> None:
    required = ("best.pt", "config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise SystemExit(
            f"Missing {label} checkpoint artifacts in {path}: {', '.join(missing)}"
        )


def mean_metric(frame: pd.DataFrame, model: str, conditions, metric: str) -> float:
    selected = frame[
        (frame["model"] == model) & frame["condition"].isin(tuple(conditions))
    ]
    if len(selected) != len(tuple(conditions)):
        raise RuntimeError(
            f"Incomplete validation rows for {model}/{metric}: "
            f"expected {len(tuple(conditions))}, found {len(selected)}"
        )
    value = float(selected[metric].mean())
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {metric} for {model}")
    return value


def relative(candidate: float, reference: float) -> float:
    return candidate / max(abs(reference), 1e-12) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only gate for the parameter-free portable reliability "
            "adapter. Test and held-out corruption conditions are never accessed."
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

    ordinary_dir = runs_root / args.dataset / ordinary / control_name
    robust_dir = runs_root / args.dataset / robust / control_name
    soft_dir = runs_root / args.dataset / soft / soft_name
    require_checkpoint(ordinary_dir, "ordinary control")
    require_checkpoint(robust_dir, "hard-robust control")

    job = TrainJob("transfer_soft_gate", args.dataset, soft, seed)
    settings = model_defaults(job)
    selection = soft_dir / "selection.json"
    if selection.is_file() and (soft_dir / "best.pt").is_file():
        artifact = json.loads(selection.read_text(encoding="utf-8"))
        if artifact.get("test_accessed") is not False:
            raise SystemExit(f"Unsafe selection artifact claims test access: {selection}")
        print(f"[reuse validation-only] {soft_dir}", flush=True)
    else:
        command = [
            args.python,
            str(code_root / "train.py"),
            "--data-root", str(data_root),
            "--output-root", str(runs_root),
            "--dataset", args.dataset,
            "--model", soft,
            "--seq-len", "96",
            "--pred-len", "96",
            "--seed", str(seed),
            "--run-name", soft_name,
            "--device", args.device,
            "--workers", str(args.workers),
            "--skip-test",
        ]
        for key, value in settings.items():
            command.extend([f"--{key}", value])
        run(command, args.dry_run)
        if args.dry_run:
            return
        require_checkpoint(soft_dir, "soft reliability")
        if not selection.is_file():
            raise RuntimeError(f"Validation selection artifact was not created: {selection}")

    output_dir = results_root / "validation_transfer_soft_gate"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.dataset}_{args.backbone}_s{seed}.csv"
    command = [
        args.python,
        str(code_root / "evaluate_robustness.py"),
        "--run-dir", str(ordinary_dir),
        "--run-dir", str(robust_dir),
        "--run-dir", str(soft_dir),
        "--data-root", str(data_root),
        "--device", args.device,
        "--calibration", "none",
        "--split", "validation",
        "--conditions", *KNOWN_VALIDATION_CONDITIONS,
        "--output", str(csv_path),
    ]
    run(command, args.dry_run)
    if args.dry_run:
        return

    frame = pd.read_csv(csv_path)
    if set(frame.get("evaluation_split", [])) != {"validation"}:
        raise RuntimeError("Gate output is not validation-only")
    known = KNOWN_VALIDATION_CONDITIONS[1:]
    clean_plain = mean_metric(frame, ordinary, ("clean",), "crps")
    clean_soft = mean_metric(frame, soft, ("clean",), "crps")
    known_plain = mean_metric(frame, ordinary, known, "crps")
    known_robust = mean_metric(frame, robust, known, "crps")
    known_soft = mean_metric(frame, soft, known, "crps")
    known_mae_plain = mean_metric(frame, ordinary, known, "mae")
    known_mae_soft = mean_metric(frame, soft, known, "mae")
    clean_nll_plain = mean_metric(frame, ordinary, ("clean",), "nll")
    clean_nll_soft = mean_metric(frame, soft, ("clean",), "nll")
    known_nll_plain = mean_metric(frame, ordinary, known, "nll")
    known_nll_soft = mean_metric(frame, soft, known, "nll")

    clean_crps_delta = relative(clean_soft, clean_plain)
    known_crps_delta = relative(known_soft, known_plain)
    known_mae_delta = relative(known_mae_soft, known_mae_plain)
    clean_nll_tolerance = max(0.10, 0.10 * abs(clean_nll_plain))
    known_nll_tolerance = max(0.10, 0.10 * abs(known_nll_plain))
    criteria = {
        "clean_crps_noninferior_within_1pct": clean_crps_delta <= 0.01,
        "known_crps_improves_over_plain": known_crps_delta < 0.0,
        "known_mae_noninferior_within_1pct": known_mae_delta <= 0.01,
        "clean_nll_no_instability": clean_nll_soft <= clean_nll_plain + clean_nll_tolerance,
        "known_nll_no_instability": known_nll_soft <= known_nll_plain + known_nll_tolerance,
    }
    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": seed,
        "selection_split": "validation",
        "test_accessed_by_gate": False,
        "held_out_conditions_accessed": False,
        "known_validation_conditions": list(KNOWN_VALIDATION_CONDITIONS),
        "models": {"plain": ordinary, "hard_robust": robust, "soft_reliability": soft},
        "metrics": {
            "clean_crps_plain": clean_plain,
            "clean_crps_soft": clean_soft,
            "clean_crps_soft_relative_delta": clean_crps_delta,
            "known_crps_plain": known_plain,
            "known_crps_hard_robust": known_robust,
            "known_crps_soft": known_soft,
            "known_crps_soft_relative_delta_vs_plain": known_crps_delta,
            "known_crps_soft_relative_delta_vs_hard_robust": relative(known_soft, known_robust),
            "known_mae_soft_relative_delta_vs_plain": known_mae_delta,
            "clean_nll_plain": clean_nll_plain,
            "clean_nll_soft": clean_nll_soft,
            "known_nll_plain": known_nll_plain,
            "known_nll_soft": known_nll_soft,
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved gate decision: {json_path}", flush=True)


if __name__ == "__main__":
    main()
