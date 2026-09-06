from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KNOWN = {
    "gaussian:0.1",
    "gaussian:0.3",
    "impulse:0.05:5.0",
    "random_missing:0.1",
    "random_missing:0.3",
    "block_missing:0.2",
}
UNSEEN = {
    "level_shift:2.0",
    "gradual_drift:2.0",
    "sensor_freeze:0.30:0.25",
    "heavy_tail:0.20:20.0",
}
FULL = "kamo_v5_shared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired summary for the shared-learned role ablation"
    )
    parser.add_argument("--results-root", default="revision_results")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    source = root / "robustness" / "role_ablation"
    files = sorted(source.glob("ETTh2_kamo_v5_shared*.csv"))
    if not files:
        raise SystemExit(f"No shared-learned role CSV files found below {source}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame = frame[frame["model"].astype(str).str.startswith("kamo_v5_shared")].copy()
    keys = ["dataset", "seed", "condition"]
    if frame.duplicated(keys + ["model"]).any():
        raise SystemExit("Duplicate shared-role result cells detected")
    full = frame[frame["model"] == FULL]
    if full.empty:
        raise SystemExit("The full kamo_v5_shared row is missing")
    ablated = frame[frame["model"] != FULL]
    paired = ablated.merge(
        full[keys + ["mae", "crps", "nll", "interval_score_95", "picp_95"]],
        on=keys,
        suffixes=("_ablation", "_full"),
        validate="many_to_one",
    )
    if len(paired) != len(ablated):
        raise SystemExit("Some ablation cells lack an exact full-model pair")
    for metric in ("mae", "crps", "nll", "interval_score_95"):
        paired[f"{metric}_change_percent"] = 100.0 * (
            paired[f"{metric}_ablation"] - paired[f"{metric}_full"]
        ) / paired[f"{metric}_full"].abs().clip(lower=1e-12)
    paired["picp_95_error_change"] = (
        (paired["picp_95_ablation"] - 0.95).abs()
        - (paired["picp_95_full"] - 0.95).abs()
    )
    paired["condition_group"] = "other"
    paired.loc[paired["condition"].eq("clean"), "condition_group"] = "clean"
    paired.loc[paired["condition"].isin(KNOWN), "condition_group"] = "known"
    paired.loc[paired["condition"].isin(UNSEEN), "condition_group"] = "unseen"

    summaries = []
    for model, group in paired.groupby("model"):
        row = {
            "model": model,
            "seeds": sorted(int(value) for value in group["seed"].unique()),
            "paired_cells": int(len(group)),
            "complete_three_seed": bool(group["seed"].nunique() == 3),
        }
        for label in ("clean", "known", "unseen"):
            subset = group[group["condition_group"] == label]
            if subset.empty:
                continue
            row[f"{label}_mae_change_percent"] = float(
                subset["mae_change_percent"].mean()
            )
            row[f"{label}_crps_change_percent"] = float(
                subset["crps_change_percent"].mean()
            )
            row[f"{label}_nll_change_percent"] = float(
                subset["nll_change_percent"].mean()
            )
            row[f"{label}_interval_score_95_change_percent"] = float(
                subset["interval_score_95_change_percent"].mean()
            )
            row[f"{label}_picp_95_error_change"] = float(
                subset["picp_95_error_change"].mean()
            )
            row[f"{label}_full_better_crps_pairs"] = int(
                (subset["crps_change_percent"] > 0).sum()
            )
            row[f"{label}_total_pairs"] = int(len(subset))
        summaries.append(row)

    output = (
        Path(args.output).resolve()
        if args.output
        else root / "shared_role_paired_rows.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output, index=False)
    summary_path = output.with_suffix(".json")
    payload = {
        "interpretation": (
            "Positive ablation-minus-full percentages mean the full shared-learned "
            "model is better. Seed-2026-only output is a feasibility diagnostic, "
            "not paper evidence."
        ),
        "all_models_complete_three_seed": bool(
            summaries and all(row["complete_three_seed"] for row in summaries)
        ),
        "models": summaries,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved paired rows: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
