from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


# Same six-candidate budget for every method.  Candidates are declared before
# any test evaluation and selected only by validation CRPS.
SEARCH_GRID = (
    {"d-model": "48", "lr": "0.001", "dropout": "0.1"},
    {"d-model": "64", "lr": "0.001", "dropout": "0.1"},
    {"d-model": "64", "lr": "0.0005", "dropout": "0.1"},
    {"d-model": "96", "lr": "0.0005", "dropout": "0.1"},
    {"d-model": "64", "lr": "0.0005", "dropout": "0.2"},
    {"d-model": "96", "lr": "0.0002", "dropout": "0.1"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equal-budget validation-only tuning")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--search-root", default="tuning_runs")
    parser.add_argument("--final-root", default="revision_runs")
    parser.add_argument("--output", default="revision_results/best_validation_configs.json")
    parser.add_argument("--datasets", nargs="+", default=["ETTh2", "weather"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["kamo_v5", "patchtst", "itransformer", "timemixer", "vpnet"],
    )
    parser.add_argument("--stage", choices=["search", "final", "all"], default="search")
    parser.add_argument("--seed", type=int, default=2026, help="Fixed search seed")
    parser.add_argument("--final-seeds", default="2026,2027,2028")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: Sequence[str], log: Path, dry_run: bool) -> None:
    print(" ".join(shlex.quote(str(value)) for value in command), flush=True)
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def common_command(args: argparse.Namespace, dataset: str, model: str) -> List[str]:
    return [
        args.python,
        "train.py",
        "--data-root", str(Path(args.data_root).resolve()),
        "--dataset", dataset,
        "--model", model,
        "--seq-len", "96",
        "--pred-len", "96",
        "--epochs", "40",
        "--patience", "7",
        "--batch-size", "16",
        "--device", args.device,
        "--workers", str(args.workers),
    ]


def search(args: argparse.Namespace) -> Dict[str, object]:
    root = Path(args.search_root).resolve()
    selections: Dict[str, object] = {
        "selection_split": "validation",
        "selection_metric": "crps",
        "test_accessed_during_search": False,
        "candidates_per_model": len(SEARCH_GRID),
        "grid": SEARCH_GRID,
        "best": {},
    }
    for dataset, model in itertools.product(args.datasets, args.models):
        candidates = []
        for index, settings in enumerate(SEARCH_GRID):
            run_name = f"validation_search_c{index:02d}_seed{args.seed}"
            run_dir = root / dataset / model / run_name
            selection_path = run_dir / "selection.json"
            command = common_command(args, dataset, model) + [
                "--output-root", str(root),
                "--run-name", run_name,
                "--seed", str(args.seed),
                "--skip-test",
            ]
            for key, value in settings.items():
                command.extend([f"--{key}", value])
            if not selection_path.exists():
                run(
                    command,
                    Path(args.search_root) / "logs" / f"{dataset}_{model}_c{index:02d}.log",
                    args.dry_run,
                )
            if args.dry_run:
                continue
            result = json.loads(selection_path.read_text(encoding="utf-8"))
            if result.get("test_accessed") is not False:
                raise RuntimeError(f"Unsafe tuning artifact: {selection_path}")
            candidates.append(
                {
                    "candidate": index,
                    "settings": settings,
                    "best_val_crps": float(result["best_val_crps"]),
                    "best_epoch": int(result["best_epoch"]),
                    "run_dir": str(run_dir),
                }
            )
        if not args.dry_run:
            best = min(candidates, key=lambda row: row["best_val_crps"])
            selections["best"][f"{dataset}|{model}"] = best
    return selections


def final(args: argparse.Namespace, selections: Dict[str, object]) -> None:
    seeds = tuple(int(value) for value in args.final_seeds.split(",") if value.strip())
    root = Path(args.final_root).resolve()
    for dataset, model in itertools.product(args.datasets, args.models):
        key = f"{dataset}|{model}"
        if key not in selections.get("best", {}):
            raise KeyError(f"No predeclared validation selection for {key}")
        settings = selections["best"][key]["settings"]
        for seed in seeds:
            run_name = f"tuned_validation_only_seed{seed}_h96"
            run_dir = root / dataset / model / run_name
            if (run_dir / "metrics.json").exists() and (run_dir / "best.pt").exists():
                print(f"[reuse] {run_dir}")
                continue
            command = common_command(args, dataset, model) + [
                "--output-root", str(root),
                "--run-name", run_name,
                "--seed", str(seed),
            ]
            for option, value in settings.items():
                command.extend([f"--{option}", value])
            run(
                command,
                Path(args.final_root) / "logs" / f"{dataset}_{model}_seed{seed}.log",
                args.dry_run,
            )


def main() -> None:
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)
    output = Path(args.output).resolve()
    if args.stage in {"search", "all"}:
        selections = search(args)
        if not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(selections, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Validation selections saved to {output}")
    else:
        selections = json.loads(output.read_text(encoding="utf-8"))
    if args.stage in {"final", "all"}:
        if args.dry_run and not output.exists():
            print("Dry-run final commands require an existing selection file; search commands shown only.")
            return
        final(args, selections)


if __name__ == "__main__":
    main()
