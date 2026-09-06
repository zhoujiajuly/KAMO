from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


IDENTIFIERS = ["family", "dataset", "model", "condition"]
METRICS = [
    "mae", "rmse", "nll", "crps",
    "picp_80", "mpiw_80", "interval_score_80",
    "picp_90", "mpiw_90", "interval_score_90",
    "picp_95", "mpiw_95", "interval_score_95",
    "picp_99", "mpiw_99", "interval_score_99",
    "sample_mae_q95", "sample_mae_cvar95",
    "sample_crps_q95", "sample_crps_cvar95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and audit KAMO revision results")
    parser.add_argument("--results-root", default="revision_results")
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    robustness = root / "robustness"
    for path in sorted(robustness.rglob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        try:
            family = path.relative_to(robustness).parts[0]
        except (ValueError, IndexError):
            family = "unknown"
        frame["family"] = family
        frame["source_file"] = str(path.resolve())
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No robustness CSV files found below {robustness}")
    frame = pd.concat(rows, ignore_index=True)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def mean_std_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [metric for metric in METRICS if metric in frame.columns]
    grouped = frame.groupby(IDENTIFIERS, dropna=False)
    mean = grouped[metrics].mean().add_suffix("_mean")
    std = grouped[metrics].std(ddof=1).add_suffix("_std")
    count = grouped["seed"].nunique().rename("n_seeds")
    parameters = grouped["parameters"].first().rename("parameters")
    full_test = grouped["full_test"].all().rename("full_test")
    return pd.concat([count, parameters, full_test, mean, std], axis=1).reset_index()


def tradeoff_table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "dataset", "model", "seed"]
    clean = frame[frame["condition"] == "clean"][keys + ["mae", "crps"]].rename(
        columns={"mae": "clean_mae", "crps": "clean_crps"}
    )
    corrupted = frame[frame["condition"] != "clean"].merge(clean, on=keys, how="left")
    corrupted["mae_degradation_pct"] = 100.0 * (
        corrupted["mae"] / corrupted["clean_mae"] - 1.0
    )
    corrupted["crps_degradation_pct"] = 100.0 * (
        corrupted["crps"] / corrupted["clean_crps"] - 1.0
    )
    return (
        corrupted.groupby(["family", "dataset", "model", "condition"])[
            ["mae_degradation_pct", "crps_degradation_pct"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )


def gate_decision(frame: pd.DataFrame) -> Dict[str, object]:
    gate = frame[frame["family"] == "gate"].copy()
    required = {"kamo_v5_shared", "kamo_v5_separate", "kamo_v5_mask_only"}
    available = set(gate["model"].unique())
    missing = sorted(required - available)
    if missing:
        return {
            "pass": False,
            "complete": False,
            "reason": f"Missing gate models: {missing}",
        }
    seed_counts = (
        gate.groupby(["dataset", "model", "condition"])["seed"].nunique()
    )
    if (seed_counts < 3).any() or not gate["full_test"].astype(bool).all():
        return {
            "pass": False,
            "complete": False,
            "reason": "Gate requires three seeds and full-test evaluation for every cell.",
            "minimum_seed_count": int(seed_counts.min()),
        }

    corrupt = gate[gate["condition"] != "clean"]
    per_seed = (
        corrupt.groupby(["dataset", "model", "seed"], as_index=False)["crps"].mean()
    )
    mean_corrupt = per_seed.groupby(["dataset", "model"])["crps"].mean().unstack("model")
    clean = (
        gate[gate["condition"] == "clean"]
        .groupby(["dataset", "model"])["crps"]
        .mean()
        .unstack("model")
    )
    details = []
    for dataset in sorted(mean_corrupt.index):
        shared = float(mean_corrupt.loc[dataset, "kamo_v5_shared"])
        separate = float(mean_corrupt.loc[dataset, "kamo_v5_separate"])
        mask_only = float(mean_corrupt.loc[dataset, "kamo_v5_mask_only"])
        details.append(
            {
                "dataset": dataset,
                "shared_crps": shared,
                "separate_crps": separate,
                "mask_only_crps": mask_only,
                "shared_vs_separate_pct": 100.0 * (shared / separate - 1.0),
                "shared_improvement_over_mask_pct": 100.0 * (1.0 - shared / mask_only),
                "shared_clean_vs_mask_pct": 100.0
                * (
                    float(clean.loc[dataset, "kamo_v5_shared"])
                    / float(clean.loc[dataset, "kamo_v5_mask_only"])
                    - 1.0
                ),
            }
        )
    # This is a predeclared engineering gate, not a significance test.  Sharing
    # must be non-inferior to separate gates on each dataset, improve on the
    # mask-only multiscale control on average, and avoid a material clean penalty.
    noninferior = all(row["shared_vs_separate_pct"] <= 1.0 for row in details)
    robust_gain = float(np.mean([row["shared_improvement_over_mask_pct"] for row in details])) >= 0.5
    clean_safe = all(row["shared_clean_vs_mask_pct"] <= 1.0 for row in details)
    passed = noninferior and robust_gain and clean_safe
    return {
        "pass": passed,
        "complete": True,
        "screening_rule": {
            "shared_vs_separate_noninferiority_margin_pct": 1.0,
            "mean_shared_improvement_over_mask_required_pct": 0.5,
            "maximum_clean_penalty_vs_mask_pct": 1.0,
        },
        "checks": {
            "shared_noninferior_to_separate": noninferior,
            "shared_improves_mask_only": robust_gain,
            "clean_tradeoff_acceptable": clean_safe,
        },
        "details": details,
        "warning": (
            "This gate prevents wasteful scaling. It is not inferential evidence; "
            "the paper must report all seed-level results and uncertainty."
        ),
    }


def completeness_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(IDENTIFIERS)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda values: ",".join(map(str, sorted(set(values))))),
            full_test=("full_test", "all"),
            rows=("seed", "size"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frame = load_results(root)
    frame.to_csv(root / "all_seed_level_results.csv", index=False)
    mean_std_table(frame).to_csv(root / "summary_mean_std.csv", index=False)
    tradeoff_table(frame).to_csv(root / "clean_robustness_tradeoff.csv", index=False)
    completeness_table(frame).to_csv(root / "completeness.csv", index=False)
    probability_columns = [
        column
        for column in [*IDENTIFIERS, "seed", "nll", "crps", "picp_90", "mpiw_90",
                       "picp_95", "mpiw_95", "picp_99", "mpiw_99",
                       "interval_score_90", "interval_score_95", "interval_score_99"]
        if column in frame.columns
    ]
    frame[probability_columns].to_csv(root / "probabilistic_metrics.csv", index=False)
    decision = gate_decision(frame)
    (root / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print(f"Collected {len(frame)} seed-level condition rows below {root}")


if __name__ == "__main__":
    main()
