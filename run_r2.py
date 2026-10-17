#!/usr/bin/env python
"""Main entry: python run_r2.py --help. Run from the extracted package directory."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import os
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from kamo.engine import set_seed
from kamo_r2.data import build_bundle
from kamo_r2.factory import build_model
from kamo_r2.protocol import (PROTOCOL_VERSION, TUNE_CONDITIONS, SEEDS, recipe_cases,
                              candidates, case_name, case_shard, reference_config)
from kamo_r2.storage import (digest, source_digest, read_json, write_json, load_checkpoint)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", default=os.environ.get("KAMO_DATA_ROOT"))
    p.add_argument("--out", default="r2_runs")
    p.add_argument("--recipe", choices=["pilot", "core", "controls", "highdim", "physionet", "long", "dimension"], default="core")
    p.add_argument("--stage", choices=["plan", "check", "train", "evaluate", "all", "report", "profile", "profile-scaling"], default="plan")
    p.add_argument("--datasets", help="Comma separated filter; does not add unplanned cases")
    p.add_argument("--models", help="Comma separated filter; does not add unplanned cases")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--seq-len", type=int, help="Explicit override, normally leave unset")
    p.add_argument("--pred-len", type=int, help="Explicit override, normally leave unset")
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--search-epochs", type=int, default=10)
    p.add_argument("--search-patience", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="Opt-in mixed precision; apply identically to every compared method")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--eval-set", choices=["minimal", "core", "severity", "all"], default="core")
    p.add_argument("--save-predictions", action="store_true", help="Save ALL full-size raw arrays; can use many GB")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--profile-batch-size", type=int, default=4)
    p.add_argument("--profile-repeats", type=int, default=30)
    p.add_argument("--keep-going", action="store_true", help="Record failed cases and continue; final exit remains nonzero")
    args = p.parse_args(argv)
    for key in ["candidates", "search_epochs", "epochs", "batch_size", "threads", "shards"]:
        if getattr(args, key) < 1:
            p.error(key + " must be positive")
    if not 0 <= args.shard_index < args.shards:
        p.error("shard-index must be in [0, shards)")
    args.seed_list = [int(x) for x in args.seeds.split(",")]
    if len(set(args.seed_list)) != len(args.seed_list) or not args.seed_list:
        p.error("Seeds must be nonempty and unique")
    return args


def cases_for(args):
    cases = recipe_cases(args.recipe)
    if args.datasets:
        allowed = set(x.lower() for x in args.datasets.split(","))
        cases = [c for c in cases if c["dataset"].lower() in allowed]
    if args.models:
        allowed = set(args.models.split(","))
        cases = [c for c in cases if c["model"] in allowed]
    for case in cases:
        if args.seq_len is not None: case["seq_len"] = args.seq_len
        if args.pred_len is not None: case["pred_len"] = args.pred_len
    if not cases:
        raise ValueError("Filters selected no cases; check recipe/model/dataset spelling")
    return cases


def make_plan(args, cases):
    searched = sum(not c["inherited"] for c in cases)
    return dict(protocol_version=PROTOCOL_VERSION, principal_model="kamo_s",
                principal_reason="shared learned observation reliability is the stated mechanism; chosen before new test results",
                recipe=args.recipe, cases=cases, seeds=args.seed_list,
                hpo_seed=2025, effective_trials_per_searched_case=args.candidates,
                search_runs=searched * args.candidates, final_runs=len(cases) * len(args.seed_list),
                search_max_epochs=args.search_epochs, final_max_epochs=args.epochs,
                search_patience=args.search_patience, final_patience=args.patience,
                batch_size=args.batch_size, amp=args.amp, split="70/5/5/20 continuous; patient-disjoint for PhysioNet",
                tuning_conditions=TUNE_CONDITIONS,
                selection="checkpoint=min clean tuning normalized CRPS; candidate=min mean of 5 tuning-condition normalized CRPS",
                calibration="clean, disjoint calibration split; scalar Gaussian NLL-optimal temperature; frozen for all test corruptions",
                control_rule="inherit principal selected hyperparameters; retrain each ablation from scratch; NOT independently optimal controls",
                caveats=["Existing test sets were viewed in R1; new selections use tuning only.",
                         "Not an official-recipe leaderboard replication. See BASELINE_SOURCES.md.",
                         "Trial budget equality does not imply equal compute or optimal tuning.",
                         "Synthetic corruption does not establish real-fault robustness."])


@lru_cache(maxsize=1)
def cached_bundle(dataset, data_root, seq_len, pred_len, batch_size, workers, cache, dimensions):
    return build_bundle(dataset, data_root, seq_len, pred_len, batch_size, workers, cache, dimensions)


def get_bundle(args, case):
    return cached_bundle(case["dataset"], str(Path(args.data_root).resolve()), case["seq_len"], case["pred_len"],
                         args.batch_size, args.workers, str(Path(args.out) / "data_cache"), case["dimensions"])


def run_config(args, case, candidate, seed, search=False):
    cfg = dict(candidate)
    cfg.update({k: v for k, v in case.items() if k != "inherited"})
    cfg.update(seed=seed, batch_size=args.batch_size, epochs=args.search_epochs if search else args.epochs,
               patience=args.search_patience if search else args.patience, amp=args.amp,
               train_noise=.02, train_missing=.05, train_impulse=.01, train_block=.10,
               train_block_prob=.25, mae_weight=.1, grad_clip=1.)
    return cfg


def case_protocol(args, case, bundle, source):
    result = make_plan(args, [case])
    # Recipe and shard affect scheduling, never the scientific identity of a case.
    result.pop("recipe", None)
    result.update(data=bundle.audit, source_sha256=source)
    return result


def loaded_model(config, bundle, checkpoint, device):
    model = build_model(config, bundle).to(device)
    state = load_checkpoint(checkpoint, device)
    model.load_state_dict(state["model"], strict=True)
    return model


def choose_candidate(args, case, bundle, root, protocol, device):
    from kamo_r2.training import train_model
    from kamo_r2.evaluation import evaluate_model
    identity = digest(protocol)
    selection_path = root / "selection.json"
    if selection_path.exists():
        selected = read_json(selection_path)
        if selected["protocol_fingerprint"] != identity:
            raise RuntimeError("Protocol/data/source changed: choose a NEW --out directory, do not reuse " + str(root))
        return selected["config"]
    if case["inherited"]:
        parent_case = dict(case, model="kamo_s", inherited=False)
        parent = Path(args.out) / case_name(parent_case) / "selection.json"
        if not parent.exists():
            raise FileNotFoundError("Run matching core kamo_s first: " + str(parent))
        previous = read_json(parent)
        parent_protocol = case_protocol(args, parent_case, bundle, protocol["source_sha256"])
        if previous["protocol_fingerprint"] != digest(parent_protocol):
            raise RuntimeError("Principal selection protocol mismatch; use identical training flags for controls")
        chosen = dict(previous["config"], model=case["model"])
        write_json(selection_path, dict(protocol_fingerprint=identity, config=chosen,
                   selection_kind="inherited_component_control", parent_selection=str(parent), protocol=protocol))
        return chosen
    trial_candidates = candidates(case["model"], case["dataset"], args.candidates, num_vars=bundle.num_vars)
    write_json(root / "candidates.json", dict(protocol=protocol, candidates=trial_candidates))
    records = []
    for index, candidate in enumerate(trial_candidates):
        cfg = run_config(args, case, candidate, 2025, search=True)
        trial = root / "search" / ("trial_%02d" % index)
        fp = digest(dict(protocol=identity, config=cfg))
        print("SEARCH {} trial {}/{}".format(case_name(case), index + 1, len(trial_candidates)), flush=True)
        trained = train_model(cfg, bundle, trial, device=device, fingerprint=fp)
        score_file = trial / "selection_score.json"
        if score_file.exists():
            record = read_json(score_file)
            if record["fingerprint"] != fp: raise RuntimeError("Stale validation score")
        else:
            model = loaded_model(cfg, bundle, trial / "best.pt", device)
            scores = {}
            for condition in TUNE_CONDITIONS:
                metrics, _ = evaluate_model(model, bundle.val, device, scaler=bundle.scaler,
                                            condition=condition, diagnostics=False)
                scores[condition] = float(metrics["normalized_crps"])
            record = dict(trial=index, score=float(np.mean(list(scores.values()))), conditions=scores,
                          fingerprint=fp, config=candidate, training=trained)
            write_json(score_file, record)
            del model
        records.append(record)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    # Failures are never silently removed from the declared candidate budget.
    chosen = min(records, key=lambda r: (r["score"], r["trial"]))
    write_json(selection_path, dict(protocol_fingerprint=identity, config=chosen["config"],
               trial=chosen["trial"], score=chosen["score"], completed_candidates=len(records),
               protocol=protocol, records=records))
    return chosen["config"]


def train_case(args, case, bundle, root, protocol, device):
    from kamo_r2.training import train_model
    candidate = choose_candidate(args, case, bundle, root, protocol, device)
    identity = digest(protocol)
    for seed in args.seed_list:
        cfg = run_config(args, case, candidate, seed)
        final_dir = root / "final" / ("seed_%d" % seed)
        fp = digest(dict(protocol=identity, config=cfg))
        print("FINAL {} seed {}".format(case_name(case), seed), flush=True)
        write_json(final_dir / "run_config.json", dict(config=cfg, fingerprint=fp, protocol=protocol))
        train_model(cfg, bundle, final_dir, device=device, fingerprint=fp)


def conditions_for(which):
    from kamo_r2.evaluation import CORE_CONDITIONS, SEVERITY_CONDITIONS
    if which == "minimal": raw = ["clean", "gaussian:0.10", "level_shift:1.0"]
    elif which == "core": raw = CORE_CONDITIONS
    elif which == "severity": raw = SEVERITY_CONDITIONS
    else: raw = CORE_CONDITIONS + SEVERITY_CONDITIONS
    # Numeric spelling aliases (1 vs 1.0) are one condition, not extra evidence.
    normalized = []
    for condition in raw:
        parts = condition.split(":")
        normalized.append(":".join([parts[0]] + [format(float(v), ".10g") for v in parts[1:]]))
    return list(dict.fromkeys(normalized))


def evaluate_case(args, case, bundle, root, protocol, device):
    from kamo_r2.evaluation import evaluate_model, fit_std_temperature, condition_slug
    selection = read_json(root / "selection.json")
    if selection["protocol_fingerprint"] != digest(protocol):
        raise RuntimeError("Evaluation protocol changed; keep training flags identical or use new --out")
    for seed in args.seed_list:
        final_dir = root / "final" / ("seed_%d" % seed)
        run = read_json(final_dir / "run_config.json")
        complete = read_json(final_dir / "complete.json")
        if complete["fingerprint"] != run["fingerprint"]:
            raise RuntimeError("Incomplete or mismatched final training")
        model = loaded_model(run["config"], bundle, final_dir / "best.pt", device)
        calibration_file = final_dir / "calibration.json"
        if calibration_file.exists():
            calibration = read_json(calibration_file)
            if calibration["fingerprint"] != run["fingerprint"]: raise RuntimeError("Stale calibration")
        else:
            calibration = fit_std_temperature(model, bundle.calibration, device)
            calibration.update(fingerprint=run["fingerprint"], source_split="independent_clean_calibration")
            write_json(calibration_file, calibration)
        temperatures = dict(raw=1., calibrated=calibration["std_temperature"])
        for variant, temperature in temperatures.items():
            for condition in conditions_for(args.eval_set):
                result_dir = final_dir / "test" / variant / condition_slug(condition)
                path = result_dir / "metrics.json"
                if path.exists():
                    old = read_json(path)
                    full_required = args.save_predictions and variant == "raw"
                    full_present = (old.get("prediction_storage") == "full_shards" and
                                    len(list((result_dir / "prediction_chunks").glob("batch_*.npz"))) == len(bundle.test))
                    if (old.get("fingerprint") == run["fingerprint"] and
                            (result_dir / "window_metrics.npz").exists() and
                            (result_dir / "representative_cases.npz").exists() and
                            (not full_required or full_present)):
                        continue
                print("EVAL {} seed={} {} {}".format(case_name(case), seed, variant, condition), flush=True)
                metrics, _ = evaluate_model(model, bundle.test, device, scaler=bundle.scaler,
                                            condition=condition, std_temperature=temperature,
                                            diagnostics=variant == "raw", output_dir=result_dir,
                                            save_predictions=args.save_predictions and variant == "raw")
                metrics.update(dataset=case["dataset"], model=case["model"], seed=seed,
                               seq_len=case["seq_len"], pred_len=case["pred_len"],
                               dimensions=bundle.num_vars, num_vars=bundle.num_vars,
                               variant=variant, condition=condition, fingerprint=run["fingerprint"],
                               data_sha256=bundle.audit["data_sha256"], protocol_version=PROTOCOL_VERSION)
                write_json(path, metrics)
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def profile_case(args, case, bundle, root, device):
    from kamo_r2.profiling import profile_model
    run_dir = root / "final" / ("seed_%d" % args.seed_list[0])
    run = read_json(run_dir / "run_config.json")
    model = loaded_model(run["config"], bundle, run_dir / "best.pt", device)
    batch = next(iter(bundle.test))
    n = args.profile_batch_size
    if len(batch["x"]) < n: raise ValueError("Raise --batch-size to at least --profile-batch-size")
    result = profile_model(model, batch["x"][:n], batch["x_mask"][:n], device,
                           repeats=args.profile_repeats, example_y=batch["y"][:n],
                           example_y_mask=batch["y_mask"][:n], learning_rate=run["config"]["lr"],
                           weight_decay=run["config"]["weight_decay"],
                           forward_kwargs=dict(cycle_index=batch["window_start"][:n]))
    result.update(case=case, fingerprint=run["fingerprint"], training=read_json(run_dir / "complete.json"),
                  weights="fitted", note="FLOPs are operator-covered partial counts, NOT whole-model total")
    write_json(root / "profile.json", result)


def profile_scaling(args, device):
    from types import SimpleNamespace
    from kamo_r2.profiling import profile_model
    # One axis at a time, common fixed batch and FP32; synthetic inputs measure COST only.
    cases = [(d, 96, 96) for d in [8, 32, 64, 128, 321, 512]]
    cases += [(64, l, 96) for l in [48, 192, 336, 720]]
    cases += [(64, 96, h) for h in [192, 336, 720]]
    models = args.models.split(",") if args.models else ["kamo_s", "dlinear", "cyclenet", "tqnet", "timemixer_robust"]
    rows = []
    for name in models:
        for dimensions, length, horizon in cases:
            descriptor = dict(model=name, dimensions=dimensions, seq_len=length, pred_len=horizon)
            print("SCALING " + str(descriptor), flush=True)
            model = None
            try:
                set_seed(73021)
                bundle = SimpleNamespace(num_vars=dimensions, seq_len=length, pred_len=horizon, adjacency=None)
                cfg = dict(reference_config(name, "electricity"), seq_len=length, pred_len=horizon)
                model = build_model(cfg, bundle)
                x = torch.randn(args.profile_batch_size, length, dimensions)
                result = profile_model(model, x, device=device, repeats=args.profile_repeats,
                                       forward_kwargs=dict(cycle_index=torch.arange(len(x))))
                rows.append(dict(case=descriptor, result=result, weights="random; compute scaling ONLY"))
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower(): raise
                rows.append(dict(case=descriptor, result=dict(status="oom", error=str(exc))))
            finally:
                del model
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            write_json(Path(args.out) / "profile_scaling.json", rows)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        from kamo_r2.analysis import summarize_results
        print(summarize_results(out, out / "reports", bootstrap_replicates=args.bootstrap)); return
    if args.stage == "profile-scaling":
        profile_scaling(args, device); return
    cases = cases_for(args)
    plan = make_plan(args, cases)
    write_json(out / ("plan_{}_shard{}.json".format(args.recipe, args.shard_index)), plan)
    print(json.dumps({k: plan[k] for k in ["recipe", "search_runs", "final_runs", "search_max_epochs", "final_max_epochs"]}, indent=2), flush=True)
    if args.stage == "plan": return
    if not args.data_root:
        raise ValueError("Pass --data-root /path/to/datasets or set KAMO_DATA_ROOT")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch cannot see a GPU. Check your existing server environment.")
    source = source_digest()
    environment = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                       numpy=np.__version__, device=str(device), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       source_sha256=source, args=vars(args))
    write_json(out / ("environment_{}_{}_shard{}.json".format(args.recipe, args.stage, args.shard_index)), environment)
    failures = []
    for case in cases:
        if case_shard(case, args.shards) != args.shard_index: continue
        root = out / case_name(case)
        try:
            bundle = get_bundle(args, case)
            protocol = case_protocol(args, case, bundle, source)
            write_json(root / "data_audit.json", bundle.audit)
            if args.stage == "check":
                cfg = dict(reference_config(case["model"], case["dataset"]), seq_len=case["seq_len"], pred_len=case["pred_len"])
                model = build_model(cfg, bundle).to(device)
                sample = next(iter(bundle.val))
                with torch.no_grad():
                    forecast = model(sample["x"][:2].to(device), mask=sample["x_mask"][:2].to(device), cycle_index=sample["window_start"][:2].to(device))
                if not torch.isfinite(forecast.mean).all() or not torch.isfinite(forecast.std).all():
                    raise FloatingPointError("Preflight nonfinite prediction")
                print("CHECK OK " + case_name(case) + " " + str(bundle.audit["split_windows"]), flush=True)
                del model
            if args.stage in {"train", "all"}: train_case(args, case, bundle, root, protocol, device)
            if args.stage in {"evaluate", "all"}: evaluate_case(args, case, bundle, root, protocol, device)
            if args.stage == "profile": profile_case(args, case, bundle, root, device)
        except Exception as exc:
            failure = dict(case=case, error=repr(exc))
            failures.append(failure)
            write_json(out / ("failures_{}_{}_shard{}.json".format(args.recipe, args.stage, args.shard_index)), failures)
            print("FAILED " + json.dumps(failure), file=sys.stderr, flush=True)
            if not args.keep_going: raise
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    if failures:
        raise RuntimeError("{} cases failed; see failures JSON. Incomplete experiments must not be reported as complete.".format(len(failures)))
    print("DONE stage={} recipe={} shard={}/{}".format(args.stage, args.recipe, args.shard_index, args.shards), flush=True)


if __name__ == "__main__":
    main()
