from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_robustness import load_one
from kamo.engine import set_seed
from train import choose_device


def _float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated number")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate impulse-rate and impulse-amplitude sensitivity without retraining"
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--rates", type=_float_list, default=_float_list("0.01,0.03,0.05,0.10"))
    parser.add_argument("--amplitudes", type=_float_list, default=_float_list("1,3,5,10"))
    parser.add_argument("--output", default="runs/impulse_sensitivity.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    if any(rate <= 0 or rate > 1 for rate in args.rates):
        parser.error("Every impulse rate must be in (0, 1]")
    if any(amplitude < 0 for amplitude in args.amplitudes):
        parser.error("Impulse amplitudes must be non-negative")

    conditions = ["clean"] + [
        f"impulse:{rate:g}:{amplitude:g}"
        for rate in args.rates
        for amplitude in args.amplitudes
    ]
    device = choose_device(args.device)
    all_rows = []
    for raw in args.run_dir:
        run_dir = Path(raw)
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        set_seed(int(config["seed"]))
        rows = load_one(
            run_dir,
            device,
            conditions,
            max_batches=args.max_batches,
        )
        for row in rows:
            condition = str(row["condition"])
            if condition == "clean":
                row["impulse_rate"] = 0.0
                row["impulse_amplitude"] = 0.0
            else:
                _, rate, amplitude = condition.split(":")
                row["impulse_rate"] = float(rate)
                row["impulse_amplitude"] = float(amplitude)
            row["run_dir"] = str(run_dir)
        all_rows.extend(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output, index=False)
    print(f"Saved sensitivity rows={len(frame)}: {output.resolve()}")


if __name__ == "__main__":
    main()
