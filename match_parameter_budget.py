from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from train import build_model


KNOWN_VARIABLES = {
    "ETTh1": 7,
    "ETTh2": 7,
    "weather": 21,
    "exchange": 8,
    "electricity": 321,
    "traffic": 862,
    "physionet2012": 37,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find width-matched external baselines")
    parser.add_argument("--dataset", default="ETTh2", choices=sorted(KNOWN_VARIABLES))
    parser.add_argument("--num-vars", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--target-model", default="kamo_v5_shared")
    parser.add_argument("--target-d-model", type=int, default=64)
    parser.add_argument("--baselines", nargs="+", default=["patchtst_robust", "timemixer"])
    parser.add_argument(
        "--widths",
        default="16,24,32,48,64,80,96,128,160,192,256,320,384,448,512,576",
    )
    parser.add_argument("--output", default="revision_results/parameter_matches.json")
    return parser.parse_args()


def namespace(model: str, d_model: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        d_model=d_model,
        num_scales=3,
        mixer_rank=16,
        mixer_type="low_rank",
        robust_window=7,
        huber_delta=3.0,
        patch_len=16,
        patch_stride=8,
        n_layers=2,
        n_heads=4,
        ff_mult=4,
        dropout=0.1,
        kamo_patch_scales="8,16,32",
        kamo_patch_layers=1,
        reliability_hidden=16,
        recent_window=12,
        residual_trust_floor=0.1,
        scale_adjustment_coefficient=0.2,
        vp_middle_size=256,
        vp_kernel_vars=7,
        vp_kernel_patches=3,
        vp_reconstruction_weight=0.1,
    )


def count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    args = parse_args()
    num_vars = args.num_vars or KNOWN_VARIABLES[args.dataset]
    bundle = SimpleNamespace(
        num_vars=num_vars,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        adjacency=None,
    )
    target = build_model(namespace(args.target_model, args.target_d_model), bundle)
    target_parameters = count(target)
    widths = [int(value) for value in args.widths.split(",") if value.strip()]
    matches = {}
    for baseline in args.baselines:
        candidates = []
        for width in widths:
            parameters = count(build_model(namespace(baseline, width), bundle))
            candidates.append(
                {
                    "d_model": width,
                    "parameters": parameters,
                    "absolute_gap": abs(parameters - target_parameters),
                    "relative_gap_pct": 100.0 * abs(parameters - target_parameters) / target_parameters,
                }
            )
        matches[baseline] = min(candidates, key=lambda row: row["absolute_gap"])
    result = {
        "dataset": args.dataset,
        "num_vars": num_vars,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "target_model": args.target_model,
        "target_d_model": args.target_d_model,
        "target_parameters": target_parameters,
        "matches": matches,
        "rule": "closest trainable parameter count from a predeclared width grid",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
