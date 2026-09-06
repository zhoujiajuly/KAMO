from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SEEDS = (2026, 2027, 2028)
CORE_DATASETS = ("ETTh2", "weather")
STANDARD_AND_UNSEEN_CONDITIONS = (
    "clean",
    "gaussian:0.1",
    "gaussian:0.3",
    "impulse:0.05:5.0",
    "random_missing:0.1",
    "random_missing:0.3",
    "block_missing:0.2",
    "level_shift:2.0",
    "gradual_drift:2.0",
    "sensor_freeze:0.30:0.25",
    "heavy_tail:0.20:20.0",
)


@dataclass(frozen=True)
class TrainJob:
    family: str
    dataset: str
    model: str
    seed: int
    seq_len: int = 96
    pred_len: int = 96
    overrides: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def run_name(self) -> str:
        suffix = "_".join(f"{key}-{value}" for key, value in self.overrides)
        suffix = suffix.replace(",", "-").replace(".", "p")
        core = f"revision_{self.family}_s{self.seed}_h{self.pred_len}"
        return f"{core}_{suffix}" if suffix else core


def gate_jobs() -> List[TrainJob]:
    models = (
        "kamo_v5",
        "kamo_v5_shared",
        "kamo_v5_separate",
        "kamo_v5_mask_only",
    )
    return [
        TrainJob("gate", dataset, model, seed)
        for dataset in CORE_DATASETS
        for model in models
        for seed in SEEDS
    ]


def role_ablation_jobs() -> List[TrainJob]:
    """Role-specific reliability ablations requested in Reviewer 1 Comment 7.

    ETTh2 is the predeclared representative dataset.  Every row uses the
    shared-learned reliability controller that passed the core gate.  This keeps
    the role evidence aligned with the final main model.
    """
    models = (
        "kamo_v5_shared",
        "kamo_v5_shared_no_rel_norm",
        "kamo_v5_shared_no_rel_routing",
        "kamo_v5_shared_no_rel_gate",
        "kamo_v5_shared_no_unrel_scale",
    )
    return [
        TrainJob("role_ablation", "ETTh2", model, seed)
        for model in models
        for seed in SEEDS
    ]


def long_horizon_jobs() -> List[TrainJob]:
    models = ("kamo_v5_shared", "patchtst", "itransformer", "timemixer", "vpnet")
    return [
        TrainJob("long", dataset, model, seed, pred_len=horizon)
        for dataset in CORE_DATASETS
        for horizon in (192, 336, 720)
        for model in models
        for seed in SEEDS
    ]


def transfer_jobs() -> List[TrainJob]:
    # The *_reliability_soft variants transfer only the shared reliability
    # signal through a parameter-free soft input blend.  The legacy
    # *_reliability checkpoints are retained for the pilot audit but are not
    # part of the formal transfer table because their extra normalization and
    # output-scale transform modify the native backbones.
    models = (
        "patchtst", "patchtst_robust", "patchtst_reliability_soft",
        "itransformer", "itransformer_robust", "itransformer_reliability_soft",
    )
    return [
        TrainJob("transfer", dataset, model, seed)
        for dataset in CORE_DATASETS
        for model in models
        for seed in SEEDS
    ]


def capacity_jobs() -> List[TrainJob]:
    # Widths are the closest matches found by match_parameter_budget.py on the
    # predeclared grid (about 4% parameter gap at d_model=64 for shared KAMO).
    specifications = (
        ("kamo_v5_shared", "64"),
        ("patchtst_robust", "128"),
        ("timemixer", "512"),
    )
    return [
        TrainJob(
            "capacity",
            dataset,
            model,
            seed,
            overrides=(("d-model", width),),
        )
        for dataset in CORE_DATASETS
        for model, width in specifications
        for seed in SEEDS
    ]


def high_dimensional_jobs(include_traffic: bool) -> List[TrainJob]:
    datasets = ["electricity"] + (["traffic"] if include_traffic else [])
    models = ("kamo_v5_shared", "patchtst", "itransformer")
    return [
        TrainJob("highdim", dataset, model, seed)
        for dataset in datasets
        for model in models
        for seed in SEEDS
    ]


def real_missing_jobs() -> List[TrainJob]:
    models = (
        "kamo_v5_shared",
        "patchtst",
        "patchtst_reliability",
        "itransformer",
        "itransformer_reliability",
    )
    return [
        TrainJob("real_missing", "physionet2012", model, seed, 24, 12)
        for model in models
        for seed in SEEDS
    ]


def mixer_jobs() -> List[TrainJob]:
    jobs: List[TrainJob] = []
    for dataset in CORE_DATASETS:
        for seed in SEEDS:
            jobs.append(TrainJob("mixer", dataset, "kamo_v5_shared_no_mixer", seed))
            jobs.append(TrainJob("mixer", dataset, "kamo_v5_shared_dense_mixer", seed))
            for rank in (4, 8, 16, 32):
                jobs.append(
                    TrainJob(
                        "mixer",
                        dataset,
                        "kamo_v5_shared",
                        seed,
                        overrides=(("mixer-rank", str(rank)),),
                    )
                )
    return jobs


def sensitivity_jobs() -> List[TrainJob]:
    # One-factor-at-a-time around the predeclared default.  Three seeds are kept
    # because single-seed sensitivity curves can be misleading on Weather.
    sweeps = {
        "robust-window": ("3", "15"),
        "huber-delta": ("1.5", "6.0"),
        "kamo-patch-scales": ("8,16", "16,32,48"),
        "recent-window": ("6", "24"),
        "scale-adjustment-coefficient": ("0.0", "0.4"),
    }
    jobs = [
        TrainJob("sensitivity", "ETTh2", "kamo_v5_shared", seed)
        for seed in SEEDS
    ]
    for key, values in sweeps.items():
        for value in values:
            for seed in SEEDS:
                jobs.append(
                    TrainJob(
                        "sensitivity",
                        "ETTh2",
                        "kamo_v5_shared",
                        seed,
                        overrides=((key, value),),
                    )
                )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable, audit-ready KAMO reviewer revision experiments"
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--runs-root", default="revision_runs")
    parser.add_argument(
        "--legacy-runs-root",
        action="append",
        default=[],
        help="Optional prior checkpoint roots; only exact config matches are reused.",
    )
    parser.add_argument("--results-root", default="revision_results")
    parser.add_argument(
        "--stage",
        choices=(
            "gate", "role_ablation", "long", "transfer", "capacity", "highdim",
            "real_missing", "mixer", "sensitivity", "postgate", "all",
        ),
        default="gate",
    )
    parser.add_argument("--include-traffic", action="store_true")
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028",
        help="Comma-separated subset of the predeclared seeds; useful for a seed-2026 feasibility gate.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated exact dataset filter for safe multi-GPU sharding.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated exact model filter for safe multi-GPU sharding.",
    )
    parser.add_argument("--override-gate", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def model_defaults(job: TrainJob) -> Dict[str, str]:
    values = {
        "epochs": "40",
        "patience": "7",
        "batch-size": "16",
        "d-model": "64",
        "lr": "0.0005",
        "weight-decay": "0.0001",
        "num-scales": "3",
        "mixer-rank": "16",
        "robust-window": "7",
        "huber-delta": "3.0",
        "n-layers": "2",
        "n-heads": "4",
        "ff-mult": "4",
        "dropout": "0.1",
        "patch-len": "16",
        "patch-stride": "8",
        "kamo-patch-scales": "8,16,32",
        "kamo-patch-layers": "1",
        "vp-middle-size": "256",
        "vp-kernel-vars": "7",
        "vp-kernel-patches": "3",
        "vp-reconstruction-weight": "0.1",
        "train-noise": "0.02",
        "train-missing": "0.05",
        "train-impulse": "0.01",
        "train-block": "0.1",
        "train-block-prob": "0.25",
    }
    base = job.model
    for suffix in ("_reliability_soft", "_reliability", "_robust"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base == "timemixer":
        values["lr"] = "0.001"
    if base == "vpnet":
        values.update({"d-model": "96", "patch-len": "8", "lr": "0.0001"})
        if job.dataset == "weather":
            values.update(
                {
                    "d-model": "128", "patch-len": "16", "n-layers": "3",
                    "vp-kernel-vars": "17", "lr": "0.001",
                }
            )
    if job.dataset in {"electricity", "traffic"}:
        values["batch-size"] = "8"
    if job.dataset == "physionet2012":
        values.update({"batch-size": "32", "epochs": "50"})
    values.update(dict(job.overrides))
    return values


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in command)


def record(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload},
                ensure_ascii=False,
            )
            + "\n"
        )


def execute(
    command: Sequence[str], log: Path, events: Path, job_id: str, dry_run: bool
) -> None:
    print(f"[{job_id}] {command_text(command)}", flush=True)
    record(events, {"job": job_id, "status": "dry_run" if dry_run else "started", "command": list(command)})
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
            handle.flush()
        code = process.wait()
    record(events, {"job": job_id, "status": "completed" if code == 0 else "failed", "returncode": code})
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_directory(root: Path, job: TrainJob) -> Path:
    return root / job.dataset / job.model / job.run_name


def training_evidence_complete(
    job: TrainJob,
    metrics: Optional[Dict[str, object]] = None,
    selection: Optional[Dict[str, object]] = None,
) -> bool:
    """Accept either a completed test run or a validation-only selected checkpoint.

    Validation-only checkpoints are deliberately produced by the transfer gate with
    ``--skip-test``.  They are safe to reuse for formal evaluation because all
    hyperparameters and the checkpoint were selected without accessing the test set.
    """
    try:
        if metrics is not None:
            return (
                int(metrics.get("seed", -1)) == job.seed
                and float(metrics.get("batches", 0)) > 0
            )
        if selection is not None:
            return (
                selection.get("dataset") == job.dataset
                and selection.get("model") == job.model
                and int(selection.get("seed", -1)) == job.seed
                and selection.get("selection_split") == "validation"
                and selection.get("selection_metric") == "crps"
                and not bool(selection.get("test_accessed", True))
                and int(selection.get("best_epoch", 0)) > 0
                and math.isfinite(float(selection.get("best_val_crps", float("nan"))))
            )
    except (TypeError, ValueError):
        return False
    return False


def run_training_evidence_complete(run_dir: Path, job: TrainJob) -> bool:
    for filename, key in (("metrics.json", "metrics"), ("selection.json", "selection")):
        try:
            payload = json.loads((run_dir / filename).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if training_evidence_complete(job, **{key: payload}):
            return True
    return False


def completed(run_dir: Path, job: TrainJob) -> bool:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = model_defaults(job)
    return (
        (run_dir / "best.pt").exists()
        and config.get("dataset") == job.dataset
        and config.get("model") == job.model
        and int(config.get("seed", -1)) == job.seed
        and int(config.get("seq_len", -1)) == job.seq_len
        and int(config.get("pred_len", -1)) == job.pred_len
        and all(
            str(config.get(key.replace("-", "_"))) == value
            for key, value in expected.items()
        )
        and run_training_evidence_complete(run_dir, job)
        and not bool(config.get("smoke_test", False))
        and config.get("max_train_batches") is None
        and config.get("max_val_batches") is None
        and config.get("max_test_batches") is None
    )


def find_legacy(job: TrainJob, args: argparse.Namespace) -> Optional[Path]:
    if (job.overrides and job.family != "capacity") or job.family not in {
        "long", "transfer", "role_ablation", "capacity"
    }:
        return None
    expected = model_defaults(job)
    for raw_root in args.legacy_runs_root:
        model_root = Path(raw_root).resolve() / job.dataset / job.model
        if not model_root.exists():
            continue
        for config_path in sorted(model_root.rglob("config.json")):
            run_dir = config_path.parent
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            exact = (
                config.get("dataset") == job.dataset
                and config.get("model") == job.model
                and int(config.get("seed", -1)) == job.seed
                and int(config.get("seq_len", -1)) == job.seq_len
                and int(config.get("pred_len", -1)) == job.pred_len
                and all(
                    str(config.get(key.replace("-", "_"))) == value
                    for key, value in expected.items()
                )
                and not bool(config.get("smoke_test", False))
                and config.get("max_train_batches") is None
                and config.get("max_val_batches") is None
                and config.get("max_test_batches") is None
            )
            if exact and (run_dir / "best.pt").exists() and run_training_evidence_complete(run_dir, job):
                return run_dir
    return None


def train(job: TrainJob, args: argparse.Namespace, events: Path) -> Path:
    runs_root = Path(args.runs_root).resolve()
    run_dir = run_directory(runs_root, job)
    if completed(run_dir, job):
        print(f"[reuse] {run_dir}")
        return run_dir
    legacy = find_legacy(job, args)
    if legacy is not None:
        print(f"[reuse exact legacy] {legacy}")
        return legacy
    settings = model_defaults(job)
    command = [
        args.python,
        "train.py",
        "--data-root", str(Path(args.data_root).resolve()),
        "--output-root", str(runs_root),
        "--dataset", job.dataset,
        "--model", job.model,
        "--seq-len", str(job.seq_len),
        "--pred-len", str(job.pred_len),
        "--seed", str(job.seed),
        "--run-name", job.run_name,
        "--device", args.device,
        "--workers", str(args.workers),
    ]
    for key, value in settings.items():
        command.extend([f"--{key}", value])
    job_id = f"train:{job.family}:{job.dataset}:{job.model}:{job.seed}:{job.run_name}"
    log = Path(args.results_root) / "logs" / f"{job_id.replace(':', '_')}.log"
    execute(command, log, events, job_id, args.dry_run)
    if not args.dry_run and not completed(run_dir, job):
        raise RuntimeError(f"Training finished but run validation failed: {run_dir}")
    return run_dir


def evaluate_run(
    job: TrainJob,
    run_dir: Path,
    args: argparse.Namespace,
    events: Path,
) -> None:
    result_family, calibration_mode = evaluation_protocol(job)
    output = (
        Path(args.results_root)
        / "robustness"
        / result_family
        / f"{job.dataset}_{job.model}_{job.run_name}.csv"
    )
    if evaluation_output_complete(output, job, calibration_mode):
        print(f"[reuse] {output}")
        return
    if job.family == "transfer":
        calibration_command = [
            args.python,
            "calibrate_std_multimetric.py",
            "--run-dir", str(run_dir),
            "--data-root", str(Path(args.data_root).resolve()),
            "--device", args.device,
            "--workers", str(args.workers),
        ]
        if args.max_eval_batches is not None:
            calibration_command.extend(
                ["--max-batches", str(args.max_eval_batches)]
            )
        calibration_id = (
            f"calibrate:{job.family}:{job.dataset}:{job.model}:"
            f"{job.seed}:{job.run_name}"
        )
        calibration_log = (
            Path(args.results_root)
            / "logs"
            / f"{calibration_id.replace(':', '_')}.log"
        )
        execute(
            calibration_command,
            calibration_log,
            events,
            calibration_id,
            args.dry_run,
        )
    command = [
        args.python,
        "evaluate_robustness.py",
        "--run-dir", str(run_dir),
        "--data-root", str(Path(args.data_root).resolve()),
        "--output", str(output.resolve()),
        "--device", args.device,
        "--calibration", calibration_mode,
        "--conditions", *STANDARD_AND_UNSEEN_CONDITIONS,
    ]
    if args.max_eval_batches is not None:
        command.extend(["--max-batches", str(args.max_eval_batches)])
    job_id = f"evaluate:{job.family}:{job.dataset}:{job.model}:{job.seed}:{job.run_name}"
    log = Path(args.results_root) / "logs" / f"{job_id.replace(':', '_')}.log"
    execute(command, log, events, job_id, args.dry_run)


def evaluation_protocol(job: TrainJob) -> Tuple[str, str]:
    if job.family == "transfer":
        return "transfer_multimetric", "std-multimetric-auto"
    return job.family, "none"


def evaluation_output_complete(
    output: Path, job: TrainJob, calibration_mode: str
) -> bool:
    if not output.is_file() or output.stat().st_size <= 500:
        return False
    try:
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    if len(rows) != len(STANDARD_AND_UNSEEN_CONDITIONS):
        return False
    expected_calibration = (
        "global_std_temperature_multimetric"
        if calibration_mode == "std-multimetric-auto"
        else "none"
    )
    return (
        {row.get("condition") for row in rows}
        == set(STANDARD_AND_UNSEEN_CONDITIONS)
        and all(row.get("dataset") == job.dataset for row in rows)
        and all(row.get("model") == job.model for row in rows)
        and all(int(row.get("seed", -1)) == job.seed for row in rows)
        and all(row.get("full_test", "").lower() == "true" for row in rows)
        and all(
            row.get("calibration_type", "none") == expected_calibration
            for row in rows
        )
    )


def gate_has_passed(results_root: Path) -> bool:
    try:
        decision = json.loads(
            (results_root / "gate_decision.json").read_text(encoding="utf-8")
        )
        return bool(decision.get("pass", False))
    except (OSError, ValueError):
        return False


def jobs_for_stage(stage: str, include_traffic: bool) -> List[TrainJob]:
    if stage == "gate":
        return gate_jobs()
    if stage == "role_ablation":
        return role_ablation_jobs()
    if stage == "long":
        return long_horizon_jobs()
    if stage == "transfer":
        return transfer_jobs()
    if stage == "capacity":
        return capacity_jobs()
    if stage == "highdim":
        return high_dimensional_jobs(include_traffic)
    if stage == "real_missing":
        return real_missing_jobs()
    if stage == "mixer":
        return mixer_jobs()
    if stage == "sensitivity":
        return sensitivity_jobs()
    if stage == "postgate":
        return (
            role_ablation_jobs()
            + long_horizon_jobs()
            + transfer_jobs()
            + capacity_jobs()
            + high_dimensional_jobs(include_traffic)
            + real_missing_jobs()
            + mixer_jobs()
            + sensitivity_jobs()
        )
    raise ValueError(stage)


def run_jobs(jobs: Iterable[TrainJob], args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    events = results_root / "job_events.jsonl"
    selected_seeds = tuple(
        int(value.strip()) for value in str(args.seeds).split(",") if value.strip()
    )
    if not selected_seeds or not set(selected_seeds).issubset(set(SEEDS)):
        raise SystemExit(f"--seeds must be a non-empty subset of {SEEDS}")
    selected_datasets = (
        tuple(value.strip() for value in args.datasets.split(",") if value.strip())
        if args.datasets
        else None
    )
    selected_models = (
        tuple(value.strip() for value in args.models.split(",") if value.strip())
        if args.models
        else None
    )
    jobs = [
        job
        for job in jobs
        if job.seed in selected_seeds
        and (selected_datasets is None or job.dataset in selected_datasets)
        and (selected_models is None or job.model in selected_models)
    ]
    if not jobs:
        raise SystemExit("The requested stage contains no jobs for the selected seeds")
    seed_tag = "-".join(str(value) for value in selected_seeds)
    dataset_tag = "-".join(selected_datasets or ("all",))
    model_tag = "-".join(selected_models or ("all",))
    manifest = results_root / (
        f"manifest_{args.stage}_datasets-{dataset_tag}_models-{model_tag}_"
        f"seeds-{seed_tag}.json"
    )
    manifest.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "data_root": str(Path(args.data_root).resolve()),
                "legacy_runs_roots": [str(Path(value).resolve()) for value in args.legacy_runs_root],
                "selected_datasets": selected_datasets,
                "selected_models": selected_models,
                "full_test_required": args.max_eval_batches is None,
                "transfer_calibration": "global_std_temperature_multimetric",
                "conditions": STANDARD_AND_UNSEEN_CONDITIONS,
                "jobs": [dict(asdict(job), run_name=job.run_name) for job in jobs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for job in jobs:
        run_dir = train(job, args, events)
        if not args.train_only:
            evaluate_run(job, run_dir, args, events)


def main() -> None:
    args = parse_args()
    os.chdir(Path(__file__).resolve().parent)
    args.runs_root = str(Path(args.runs_root).resolve())
    args.results_root = str(Path(args.results_root).resolve())
    if args.stage == "all":
        args.stage = "gate"
        run_jobs(gate_jobs(), args)
        if args.dry_run:
            return
        collect = [args.python, "collect_revision_results.py", "--results-root", args.results_root]
        execute(
            collect,
            Path(args.results_root) / "logs" / "collect_gate.log",
            Path(args.results_root) / "job_events.jsonl",
            "collect:gate",
            False,
        )
        if not gate_has_passed(Path(args.results_root)) and not args.override_gate:
            raise SystemExit(
                "Core reliability gate did not pass. Post-gate training was stopped. "
                "Inspect gate_decision.json; use --override-gate only with an explicit scientific rationale."
            )
        args.stage = "postgate"
        run_jobs(jobs_for_stage("postgate", args.include_traffic), args)
        return
    if args.stage == "postgate" and not (
        args.override_gate or gate_has_passed(Path(args.results_root))
    ):
        raise SystemExit(
            "Post-gate experiments are locked until gate_decision.json reports pass. "
            "Run --stage gate and collect_revision_results.py first."
        )
    run_jobs(jobs_for_stage(args.stage, args.include_traffic), args)


if __name__ == "__main__":
    main()
