from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from kamo.data import build_dataloaders
from kamo.engine import evaluate, set_seed
from train import build_model, choose_device


DEFAULT_CONDITIONS = [
    "clean",
    "gaussian:0.1",
    "gaussian:0.3",
    "impulse:0.05",
    "random_missing:0.1",
    "random_missing:0.3",
    "block_missing:0.2",
    "gradual_drift:2.0",
    "sensor_freeze:0.30:0.25",
    "heavy_tail:0.20:20.0",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration(
    run_dir: Path,
    raw: str,
    checkpoint_path: Path,
    config,
    allow_debug: bool = False,
):
    if raw == "none":
        return None
    if raw == "auto":
        path = run_dir / "calibration.json"
    elif raw == "std-auto":
        path = run_dir / "std_temperature_calibration.json"
    elif raw == "std-multimetric-auto":
        path = run_dir / "std_multimetric_calibration.json"
    else:
        path = Path(raw).resolve()
    if not path.exists():
        if raw == "auto":
            return None
        raise FileNotFoundError(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if int(artifact.get("schema_version", 0)) < 2:
        raise RuntimeError(
            f"Obsolete calibration protocol: {path}. Rerun calibrate_validation.py."
        )
    if artifact.get("selection_split") != "validation":
        raise RuntimeError(f"Calibration was not selected on validation data: {path}")
    if artifact.get("test_accessed") is not False:
        raise RuntimeError(f"Calibration may have accessed test data: {path}")
    if artifact.get("full_validation") is not True and not allow_debug:
        raise RuntimeError(f"Debug/truncated calibration cannot support test results: {path}")
    if artifact.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise RuntimeError(f"Calibration/checkpoint fingerprint mismatch: {path}")
    for key in ("dataset", "model", "seed"):
        if str(artifact.get(key)) != str(config.get(key)):
            raise RuntimeError(f"Calibration {key} mismatch: {path}")
    if "selected" not in artifact:
        raise RuntimeError(f"Calibration has no selected candidate: {path}")
    return artifact


def load_one(
    run_dir: Path,
    device: torch.device,
    conditions,
    max_batches=None,
    data_root=None,
    calibration="auto",
    split="test",
):
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    args = SimpleNamespace(**config)
    if data_root is not None:
        args.data_root = data_root
    bundle = build_dataloaders(
        args.dataset,
        args.data_root,
        args.seq_len,
        args.pred_len,
        args.batch_size,
        getattr(args, "workers", 0),
        getattr(args, "cache_dir", None),
    )
    model = build_model(args, bundle).to(device)
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    checkpoint_path = run_dir / "best.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    calibration_artifact = load_calibration(
        run_dir,
        calibration,
        checkpoint_path,
        config,
        allow_debug=max_batches is not None,
    )
    std_temperature = 1.0
    selected = {}
    if calibration_artifact is not None:
        selected = calibration_artifact["selected"]
        if hasattr(model, "reliability_strength"):
            model.reliability_strength = float(
                selected.get("reliability_strength", model.reliability_strength)
            )
        if hasattr(model, "scale_adjustment_coefficient"):
            model.scale_adjustment_coefficient = float(
                selected.get(
                    "scale_adjustment_coefficient",
                    model.scale_adjustment_coefficient,
                )
            )
        if hasattr(model, "uncertainty_reliability_strength"):
            model.uncertainty_reliability_strength = float(
                selected.get(
                    "uncertainty_reliability_strength",
                    model.uncertainty_reliability_strength,
                )
            )
        std_temperature = float(selected["std_temperature"])
    rows = []
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    loader = bundle.val if split == "validation" else bundle.test
    for condition in conditions:
        metrics, _ = evaluate(
            model,
            loader,
            bundle.scaler,
            device,
            condition=condition,
            max_batches=max_batches,
            save_batches=0,
            std_temperature=std_temperature,
        )
        rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "seed": args.seed,
                "condition": condition,
                "parameters": parameters,
                "evaluation_split": split,
                "full_evaluation": max_batches is None,
                "full_test": split == "test" and max_batches is None,
                "calibration_applied": calibration_artifact is not None,
                "calibration_selection_split": (
                    calibration_artifact.get("selection_split")
                    if calibration_artifact is not None
                    else "none"
                ),
                "calibration_type": (
                    calibration_artifact.get("calibration_type", "structural")
                    if calibration_artifact is not None
                    else "none"
                ),
                "reliability_strength": float(
                    selected.get("reliability_strength", getattr(model, "reliability_strength", 1.0))
                ),
                "scale_adjustment_coefficient": float(
                    selected.get(
                        "scale_adjustment_coefficient",
                        getattr(model, "scale_adjustment_coefficient", 0.0),
                    )
                ),
                "uncertainty_reliability_strength": float(
                    selected.get(
                        "uncertainty_reliability_strength",
                        getattr(model, "uncertainty_reliability_strength", 0.0),
                    )
                ),
                **metrics,
            }
        )
        print(
            f"{args.dataset:10s} {args.model:18s} {condition:22s} "
            f"MAE={metrics['mae']:.4f} CRPS={metrics['crps']:.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained runs under corruptions")
    parser.add_argument("--run-dir", action="append", required=True, help="Repeat for model comparisons")
    parser.add_argument("--output", default="robustness.csv")
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None, help="Override data_root stored in old checkpoints")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Evaluate validation for architecture selection or test for final reporting.",
    )
    parser.add_argument(
        "--calibration",
        default="auto",
        help=(
            "auto: optionally use run_dir/calibration.json; std-auto: require "
            "run_dir/std_temperature_calibration.json; std-multimetric-auto: "
            "require run_dir/std_multimetric_calibration.json; none: disable; "
            "or provide an explicit JSON path."
        ),
    )
    args = parser.parse_args()
    device = choose_device(args.device)
    all_rows = []
    for raw in args.run_dir:
        run_dir = Path(raw)
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            seed = json.load(handle)["seed"]
        set_seed(seed)
        all_rows.extend(
            load_one(
                run_dir,
                device,
                args.conditions,
                args.max_batches,
                args.data_root,
                args.calibration,
                args.split,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output, index=False)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
