from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
LOWER_IS_BETTER = ("mae", "nll", "crps", "interval_score_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired audit of uncalibrated and validation-calibrated results"
    )
    parser.add_argument("--uncalibrated", required=True)
    parser.add_argument("--calibrated", required=True)
    parser.add_argument("--output", default="calibration_comparison.csv")
    return parser.parse_args()


def mean_change(frame: pd.DataFrame, metric: str) -> float:
    return float(frame[f"{metric}_change_percent"].mean())


def main() -> None:
    args = parse_args()
    old = pd.read_csv(args.uncalibrated)
    new = pd.read_csv(args.calibrated)
    keys = ["dataset", "model", "seed", "condition"]
    metrics = [
        "mae", "nll", "crps", "picp_95", "mpiw_95", "interval_score_95"
    ]
    required = set(keys + metrics)
    for label, frame in (("uncalibrated", old), ("calibrated", new)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SystemExit(f"{label} file is missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise SystemExit(f"{label} file contains duplicate comparison keys")
    merged = old[keys + metrics].merge(
        new[keys + metrics], on=keys, suffixes=("_uncalibrated", "_calibrated"),
        validate="one_to_one",
    )
    if len(merged) != len(old) or len(merged) != len(new):
        raise SystemExit("The two files do not contain identical paired result cells")
    for metric in LOWER_IS_BETTER + ("mpiw_95",):
        merged[f"{metric}_change_percent"] = 100.0 * (
            merged[f"{metric}_calibrated"] - merged[f"{metric}_uncalibrated"]
        ) / merged[f"{metric}_uncalibrated"].abs().clip(lower=1e-12)
    merged["picp_95_error_uncalibrated"] = (
        merged["picp_95_uncalibrated"] - 0.95
    ).abs()
    merged["picp_95_error_calibrated"] = (
        merged["picp_95_calibrated"] - 0.95
    ).abs()
    merged["picp_95_error_change"] = (
        merged["picp_95_error_calibrated"]
        - merged["picp_95_error_uncalibrated"]
    )
    merged["condition_group"] = np.select(
        [
            merged["condition"].eq("clean"),
            merged["condition"].isin(KNOWN),
            merged["condition"].isin(UNSEEN),
        ],
        ["clean", "known", "unseen"],
        default="other",
    )

    clean = merged[merged["condition_group"] == "clean"]
    known = merged[merged["condition_group"] == "known"]
    if clean.empty or known.empty:
        raise SystemExit("Comparison requires clean and known-corruption rows")
    summary = {
        "clean_mae_change_percent": mean_change(clean, "mae"),
        "clean_crps_change_percent": mean_change(clean, "crps"),
        "known_crps_change_percent": mean_change(known, "crps"),
        "known_nll_change_percent": mean_change(known, "nll"),
        "known_interval_score_95_change_percent": mean_change(
            known, "interval_score_95"
        ),
        "known_picp_95_error_change": float(known["picp_95_error_change"].mean()),
    }
    improves = bool(
        summary["clean_mae_change_percent"] <= 1.0
        and summary["clean_crps_change_percent"] <= 1.0
        and summary["known_crps_change_percent"] <= 1.0
        and summary["known_nll_change_percent"] < 0.0
        and summary["known_interval_score_95_change_percent"] < 0.0
        and summary["known_picp_95_error_change"] < 0.0
    )
    identity_fallback = bool(
        abs(summary["clean_mae_change_percent"]) < 1e-9
        and abs(summary["clean_crps_change_percent"]) < 1e-9
        and abs(summary["known_crps_change_percent"]) < 1e-9
        and abs(summary["known_nll_change_percent"]) < 1e-9
        and abs(summary["known_interval_score_95_change_percent"]) < 1e-9
        and abs(summary["known_picp_95_error_change"]) < 1e-12
    )
    summary["status"] = (
        "adopt_validation_calibration"
        if improves
        else "retain_uncalibrated"
        if identity_fallback
        else "reject_calibration"
    )
    summary["adopt_calibration"] = improves
    summary["pass"] = bool(improves or identity_fallback)
    summary["rule"] = (
        "clean MAE/CRPS and known-corruption CRPS no more than 1% worse; known-corruption "
        "NLL, IS95 and absolute PICP95 error must all improve; an exactly identical "
        "identity fallback is valid but is reported as retain_uncalibrated"
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    decision_path = output.with_suffix(".json")
    decision_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved paired rows: {output}")
    print(f"Saved gate decision: {decision_path}")


if __name__ == "__main__":
    main()
