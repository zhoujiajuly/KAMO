from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed audit of revision experiment artifacts")
    parser.add_argument("--runs-root", default="revision_runs")
    parser.add_argument("--results-root", default="revision_results")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    results_root = Path(args.results_root).resolve()
    problems: List[str] = []
    warnings: List[str] = []

    configs = list(runs_root.rglob("config.json"))
    if not configs:
        problems.append(f"No configs found below {runs_root}")
    run_records = []
    for config_path in configs:
        run_dir = config_path.parent
        try:
            config = load_json(config_path)
        except (OSError, ValueError) as error:
            problems.append(f"Unreadable config {config_path}: {error}")
            continue
        for required in ("best.pt", "history.csv", "metrics.json"):
            if not (run_dir / required).exists():
                problems.append(f"Missing {required}: {run_dir}")
        try:
            metrics = load_json(run_dir / "metrics.json")
        except (OSError, ValueError) as error:
            problems.append(f"Unreadable metrics {run_dir}: {error}")
            continue
        if config.get("smoke_test"):
            problems.append(f"Smoke-test run cannot support a paper table: {run_dir}")
        if config.get("max_train_batches") is not None:
            problems.append(f"Truncated training detected: {run_dir}")
        if config.get("max_val_batches") is not None:
            problems.append(f"Truncated validation detected: {run_dir}")
        if config.get("max_test_batches") is not None:
            problems.append(f"Truncated test detected: {run_dir}")
        run_records.append(
            {
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "seed": int(config.get("seed", -1)),
                "parameters": int(metrics.get("parameters", -1)),
                "epochs": int(config.get("epochs", -1)),
                "patience": int(config.get("patience", -1)),
                "batch_size": int(config.get("batch_size", -1)),
                "run_dir": str(run_dir),
            }
        )

    all_results = results_root / "all_seed_level_results.csv"
    if not all_results.exists():
        problems.append("Run collect_revision_results.py before the audit")
        results = pd.DataFrame()
    else:
        results = pd.read_csv(all_results)
        if "full_test" not in results or not results["full_test"].astype(bool).all():
            problems.append("At least one reported robustness row is not a full-test evaluation")
        counts = results.groupby(["family", "dataset", "model", "condition"])["seed"].nunique()
        incomplete = counts[counts < 3]
        for index, count in incomplete.items():
            problems.append(f"Only {count} seeds for result cell {index}")

        gate = results[results["family"] == "gate"]
        matched = gate[gate["model"].isin(
            ["kamo_v5_shared", "kamo_v5_separate", "kamo_v5_mask_only"]
        )]
        for dataset, group in matched.groupby("dataset"):
            parameter_counts = group.groupby("model")["parameters"].first()
            if len(parameter_counts) != 3 or parameter_counts.nunique() != 1:
                problems.append(
                    f"Reliability gate is not parameter matched on {dataset}: "
                    f"{parameter_counts.to_dict()}"
                )

    manifests = list(results_root.glob("manifest_*.json"))
    if not manifests:
        warnings.append("No suite manifest found; provenance is incomplete")
    else:
        for path in manifests:
            manifest = load_json(path)
            if not manifest.get("full_test_required", False):
                problems.append(f"Debug max-batch setting recorded in {path}")

    decision_path = results_root / "gate_decision.json"
    if decision_path.exists():
        decision = load_json(decision_path)
        if not decision.get("complete", False):
            problems.append("Reliability gate is incomplete")
        if not decision.get("pass", False):
            warnings.append("Reliability gate did not pass; do not scale or claim shared-signal superiority")
    else:
        warnings.append("No gate_decision.json found")

    report = {
        "pass": not problems,
        "problems": problems,
        "warnings": warnings,
        "runs_checked": len(run_records),
        "result_rows_checked": len(results),
    }
    output = Path(args.output).resolve() if args.output else results_root / "audit_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
