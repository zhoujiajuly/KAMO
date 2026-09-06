from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

DATASETS = ("ETTh2", "weather")
SEEDS = (2026, 2027, 2028)
VALIDATION = (
    "clean", "gaussian:0.1", "gaussian:0.3", "impulse:0.05:5.0",
    "random_missing:0.1", "random_missing:0.3", "block_missing:0.2",
)
UNSEEN = ("level_shift:2.0", "gradual_drift:2.0", "sensor_freeze:0.30:0.25", "heavy_tail:0.20:20.0")
TEST = VALIDATION + UNSEEN
# Different comparison architectures are different specifications.  In
# particular, TimeMixer d=64 must not stand in for matched TimeMixer d=512.
SPECS = {
    "kamo_core": ("kamo_v5_shared", 64, 5e-4, "core and capacity anchor"),
    "patchtst_core": ("patchtst", 64, 5e-4, "core"),
    "itransformer_core": ("itransformer", 64, 5e-4, "core"),
    "timemixer_core": ("timemixer", 64, 1e-3, "core"),
    "vpnet_core": ("vpnet", 96, 1e-4, "core"),
    "robust_patchtst_capacity": ("patchtst_robust", 128, 5e-4, "capacity"),
    "timemixer_capacity": ("timemixer", 512, 1e-3, "capacity"),
}
LANES = {
    0: tuple(("ETTh2", s) for s in SPECS) + (
        ("weather", "patchtst_core"), ("weather", "itransformer_core"),
        ("weather", "timemixer_core"), ("weather", "vpnet_core"), ("weather", "timemixer_capacity")),
    1: (("weather", "kamo_core"), ("weather", "robust_patchtst_capacity")),
}
SOURCE_FILES = ("train.py", "evaluate_robustness.py", "kamo/data.py", "kamo/engine.py",
                "kamo/metrics.py", "kamo/model.py", "kamo/baselines.py", "kamo/robustifier.py")
FIXED = dict(
    seq_len=96, pred_len=96, workers=4, epochs=40, patience=7, batch_size=16,
    weight_decay=1e-4, num_scales=3, mixer_rank=16, mixer_type="low_rank",
    robust_window=7, huber_delta=3.0, n_heads=4, ff_mult=4, patch_stride=8,
    kamo_patch_scales="8,16,32", kamo_patch_layers=1, reliability_hidden=16,
    recent_window=12, residual_trust_floor=0.1, scale_adjustment_coefficient=0.2,
    reliability_strength=1.0, uncertainty_reliability_strength=1.0,
    vp_middle_size=256, vp_kernel_patches=3, vp_reconstruction_weight=0.1,
    train_noise=0.02, train_missing=0.05, train_impulse=0.01, train_block=0.1, train_block_prob=0.25,
)
METRICS = ("mae", "crps", "nll", "picp_95", "mpiw_95", "interval_score_95")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def candidates(dataset, spec):
    model, width, lr, _ = SPECS[spec]
    base = dict(d_model=width, lr=lr, dropout=0.1, patch_len=16, n_layers=2, vp_kernel_vars=7)
    if model == "vpnet":
        base["patch_len"] = 8
        if dataset == "weather":
            base.update(d_model=128, lr=1e-3, patch_len=16, n_layers=3, vp_kernel_vars=17)
    return [
        dict(base, candidate="c0_current"),
        dict(base, lr=base["lr"] * 0.5, candidate="c1_lr_half"),
        dict(base, lr=base["lr"] * 2, candidate="c2_lr_double"),
        dict(base, dropout=0.2, candidate="c3_dropout_0p2"),
    ]


def protocol(args):
    code = Path(args.code_root).resolve()
    data = Path(args.data_root).resolve()
    return dict(
        schema=2, scope="CEE EiC fair-tuning and Reviewer-1 Comment-5 audit",
        datasets=list(DATASETS), specifications=SPECS, search_seed=2026, final_seeds=list(SEEDS),
        candidates_per_case=4, fixed_training=FIXED,
        selection_split="validation", selection_conditions=list(VALIDATION),
        selection_metric="unweighted mean CRPS over clean and six predefined known-corruption validation conditions",
        tie_breakers=["clean validation CRPS", "clean validation MAE", "candidate id"],
        test_conditions=list(TEST), test_access_during_search=False,
        unseen_used_for_selection=False,
        candidate_grids={d + "|" + s: candidates(d, s) for d in DATASETS for s in SPECS},
        lanes={str(k): list(v) for k, v in LANES.items()},
        source_sha256={p: sha256(code / p) for p in SOURCE_FILES},
        orchestrator_sha256=sha256(Path(__file__).resolve()),
        data_root=str(data), dataset_sha256={d: sha256(data / (d + ".csv")) for d in DATASETS},
    )


def write_exact(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        # Covers both ordinary reuse and a second GPU writing the same small
        # protocol file.  Never replace an existing, different frozen artifact.
        for _ in range(100):
            try:
                current = path.read_text(encoding="utf-8")
                if current == text:
                    return
                try:
                    json.loads(current)
                except ValueError:
                    pass
                else:
                    raise RuntimeError("Frozen artifact differs: " + str(path))
            except (OSError, UnicodeDecodeError):
                pass
            time.sleep(0.05)
        raise RuntimeError("Frozen artifact differs or is incomplete: " + str(path))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def protocol_hash(args):
    return sha256(Path(args.results_root) / "FROZEN_PROTOCOL.json")


def assert_code_frozen(args):
    frozen = read_json(Path(args.results_root) / "FROZEN_PROTOCOL.json")
    current = {p: sha256(Path(args.code_root).resolve() / p) for p in SOURCE_FILES}
    if current != frozen["source_sha256"] or sha256(Path(__file__).resolve()) != frozen["orchestrator_sha256"]:
        raise RuntimeError("Source code changed after protocol freeze; refusing to mix implementations")


def expected_config(dataset, spec, settings, seed):
    return dict(
        FIXED, **{k: v for k, v in settings.items() if k != "candidate"},
        dataset=dataset, model=SPECS[spec][0], seed=seed,
        skip_test=True, smoke_test=False,
        max_train_batches=None, max_val_batches=None, max_test_batches=None,
    )


def value_matches(actual, expected):
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def train_command(args, dataset, spec, settings, seed, run):
    cfg = expected_config(dataset, spec, settings, seed)
    root = Path(args.search_root if seed == 2026 else args.final_root).resolve()
    cmd = [args.python, "-u", "train.py", "--data-root", str(Path(args.data_root).resolve()),
           "--output-root", str(root), "--run-name", run.name, "--device", args.device]
    for key, value in cfg.items():
        if key in ("skip_test", "smoke_test", "max_train_batches", "max_val_batches", "max_test_batches"):
            continue
        cmd += ["--" + key.replace("_", "-"), str(value)]
    return cmd + ["--skip-test"]


def stream(cmd, log, dry=False):
    text = " ".join(shlex.quote(str(x)) for x in cmd)
    print("COMMAND:", text, flush=True)
    if dry:
        return
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    p = None
    start = time.monotonic()
    with log.open("a", encoding="utf-8") as f:
        f.write("\nSTART: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\nCOMMAND: " + text + "\n")
        f.flush()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = p.wait()
            elapsed = time.monotonic() - start
            f.write("\nFINISHED: elapsed_seconds=%.3f returncode=%d\n" % (elapsed, code))
            print("FINISHED: %.1f minutes, returncode=%d" % (elapsed / 60, code), flush=True)
        except BaseException:
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            raise
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def validate_run(run, dataset, spec, seed, settings):
    if not all((run / name).is_file() for name in ("config.json", "best.pt", "history.csv", "selection.json", "scaler.npz")):
        return False
    c, sel = read_json(run / "config.json"), read_json(run / "selection.json")
    if not all(value_matches(c.get(k), v) for k, v in expected_config(dataset, spec, settings, seed).items()):
        return False
    expected = dict(dataset=dataset, model=SPECS[spec][0], seed=seed,
                    selection_split="validation", selection_metric="crps", test_accessed=False)
    if not all(sel.get(k) == v for k, v in expected.items()):
        return False
    if not math.isfinite(float(sel.get("best_val_crps", float("nan")))) or not 1 <= int(sel.get("best_epoch", 0)) <= 40:
        return False
    if (run / "metrics.json").exists():
        raise RuntimeError("Validation-only run unexpectedly contains test metrics: " + str(run))
    return True


def validate_csv(path, dataset, model, seed, conditions, split):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if [r["condition"] for r in rows] != list(conditions):
        raise RuntimeError("Ordered condition grid mismatch: " + str(path))
    for r in rows:
        if r["dataset"] != dataset or r["model"] != model or int(r["seed"]) != seed:
            raise RuntimeError("CSV identity mismatch: " + str(path))
        if r["evaluation_split"] != split or r["full_evaluation"].lower() != "true":
            raise RuntimeError("CSV is truncated or has wrong split: " + str(path))
        if r["full_test"].lower() != str(split == "test").lower():
            raise RuntimeError("Wrong full_test flag: " + str(path))
        if r["calibration_type"] != "none" or r["calibration_applied"].lower() != "false":
            raise RuntimeError("Unexpected calibration: " + str(path))
        vals = {k: float(r[k]) for k in METRICS}
        if not all(math.isfinite(v) for v in vals.values()) or int(r["parameters"]) <= 0:
            raise RuntimeError("Invalid/nonfinite metrics: " + str(path))
        if min(vals[k] for k in ("mae", "crps", "mpiw_95", "interval_score_95")) < 0 or not 0 <= vals["picp_95"] <= 1:
            raise RuntimeError("Impossible metric range: " + str(path))
    if len({r["parameters"] for r in rows}) != 1:
        raise RuntimeError("Parameter count changes across conditions: " + str(path))
    return rows


def paths(args, dataset, spec, settings, seed, search):
    tag = settings["candidate"] + "_s2026" if search else "selected_s" + str(seed)
    name = "c5_" + spec + "_" + tag + "_h96"
    run = Path(args.search_root if search else args.final_root) / dataset / SPECS[spec][0] / name
    result = Path(args.results_root) / ("validation" if search else "test") / (dataset + "_" + spec + "_" + tag + ".csv")
    return run, result


def receipt(args, run, output, conditions, split):
    return dict(
        protocol_sha256=protocol_hash(args), run_dir=str(run.resolve()),
        checkpoint_sha256=sha256(run / "best.pt"), config_sha256=sha256(run / "config.json"),
        scaler_sha256=sha256(run / "scaler.npz"),
        csv_sha256=sha256(output), output_csv=str(output.resolve()),
        conditions=list(conditions), evaluation_split=split, calibration="none",
    )


def evaluate(args, run, output, conditions, split, log):
    assert_code_frozen(args)
    cfg = read_json(run / "config.json")
    identity = (cfg["dataset"], cfg["model"], int(cfg["seed"]), conditions, split)
    seal = output.with_suffix(".receipt.json")
    if output.exists():
        validate_csv(output, *identity)
        if not seal.exists() or read_json(seal) != receipt(args, run, output, conditions, split):
            raise RuntimeError("Existing evaluation lacks a matching receipt; inspect rather than overwrite: " + str(output))
        print("REUSE:", output, flush=True)
        return
    if seal.exists():
        raise RuntimeError("Receipt exists but CSV is absent: " + str(seal))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.csv")
    cmd = [args.python, "-u", "evaluate_robustness.py", "--run-dir", str(run.resolve()),
           "--data-root", str(Path(args.data_root).resolve()), "--device", args.device,
           "--calibration", "none", "--split", split, "--conditions", *conditions,
           "--output", str(partial.resolve())]
    stream(cmd, log, args.dry_run)
    if args.dry_run:
        return
    validate_csv(partial, *identity)
    # Only this audit's unfinished partial evaluation may be replaced on resume.
    os.replace(partial, output)
    write_exact(seal, receipt(args, run, output, conditions, split))


def selection_path(args, dataset, spec):
    return Path(args.results_root) / "selections" / (dataset + "_" + spec + ".json")


def scores(rows):
    clean = next(r for r in rows if r["condition"] == "clean")
    return dict(validation_crps_mean=sum(float(r["crps"]) for r in rows) / len(rows),
                clean_validation_crps=float(clean["crps"]), clean_validation_mae=float(clean["mae"]))


def order(entry):
    return (entry["validation_crps_mean"], entry["clean_validation_crps"],
            entry["clean_validation_mae"], entry["candidate"])


def ensure_training(args, dataset, spec, settings, seed, run):
    assert_code_frozen(args)
    if validate_run(run, dataset, spec, seed, settings):
        print("REUSE TRAINING:", run, flush=True)
        return
    if run.exists() and any(run.iterdir()):
        raise RuntimeError("Incomplete/mismatched training directory; refusing overwrite: " + str(run))
    log = Path(args.results_root) / "logs" / (run.name + "_" + dataset + "_train.log")
    stream(train_command(args, dataset, spec, settings, seed, run), log, args.dry_run)
    if not args.dry_run and not validate_run(run, dataset, spec, seed, settings):
        raise RuntimeError("Training did not produce valid completion evidence: " + str(run))


def search_case(args, dataset, spec):
    entries = []
    for i, settings in enumerate(candidates(dataset, spec), 1):
        print("\n[%s %s] SEARCH %d/4 %s" % (dataset, spec, i, settings["candidate"]), flush=True)
        run, out = paths(args, dataset, spec, settings, 2026, True)
        ensure_training(args, dataset, spec, settings, 2026, run)
        if args.dry_run:
            continue
        evaluate(args, run, out, VALIDATION, "validation", Path(args.results_root) / "logs" / (out.stem + ".log"))
        rows = validate_csv(out, dataset, SPECS[spec][0], 2026, VALIDATION, "validation")
        entries.append(dict(candidate=settings["candidate"], settings=settings, **scores(rows),
                            run_dir=str(run.resolve()), checkpoint_sha256=sha256(run / "best.pt"),
                            config_sha256=sha256(run / "config.json"),
                            validation_csv=str(out.resolve()), validation_csv_sha256=sha256(out)))
    if args.dry_run:
        return
    entries.sort(key=order)
    obj = dict(schema=2, dataset=dataset, specification=spec, model=SPECS[spec][0],
               protocol_sha256=protocol_hash(args), selection_split="validation",
               test_used_for_selection=False, unseen_used_for_selection=False,
               candidates=entries, selected=entries[0])
    write_exact(selection_path(args, dataset, spec), obj)
    print("SELECTED:", dataset, spec, entries[0]["candidate"], entries[0]["validation_crps_mean"], flush=True)


def validate_selection(args, dataset, spec):
    p = selection_path(args, dataset, spec)
    if not p.is_file():
        raise RuntimeError("Both search lanes must finish before test; missing: " + str(p))
    obj = read_json(p)
    identity = dict(dataset=dataset, specification=spec, model=SPECS[spec][0],
                    protocol_sha256=protocol_hash(args), selection_split="validation",
                    test_used_for_selection=False, unseen_used_for_selection=False)
    if any(obj.get(k) != v for k, v in identity.items()):
        raise RuntimeError("Selection identity/protocol mismatch: " + str(p))
    expected = {s["candidate"]: s for s in candidates(dataset, spec)}
    entries = obj.get("candidates", [])
    if len(entries) != 4 or {e["candidate"] for e in entries} != set(expected):
        raise RuntimeError("Incomplete/duplicate selection candidate grid: " + str(p))
    for e in entries:
        if e["settings"] != expected[e["candidate"]]:
            raise RuntimeError("Candidate settings changed: " + str(p))
        run, out = Path(e["run_dir"]), Path(e["validation_csv"])
        canonical_run, canonical_out = paths(args, dataset, spec, e["settings"], 2026, True)
        if run.resolve() != canonical_run.resolve() or out.resolve() != canonical_out.resolve():
            raise RuntimeError("Candidate points to a different run/result: " + str(p))
        if not validate_run(run, dataset, spec, 2026, e["settings"]):
            raise RuntimeError("Candidate run no longer validates: " + str(run))
        if sha256(run / "best.pt") != e["checkpoint_sha256"] or sha256(run / "config.json") != e["config_sha256"]:
            raise RuntimeError("Candidate checkpoint/config changed: " + str(run))
        rows = validate_csv(out, dataset, SPECS[spec][0], 2026, VALIDATION, "validation")
        if sha256(out) != e["validation_csv_sha256"]:
            raise RuntimeError("Candidate validation CSV changed: " + str(out))
        if read_json(out.with_suffix(".receipt.json")) != receipt(args, run, out, VALIDATION, "validation"):
            raise RuntimeError("Candidate receipt mismatch: " + str(out))
        if any(not value_matches(e[k], v) for k, v in scores(rows).items()):
            raise RuntimeError("Stored candidate score differs from CSV: " + str(out))
    if obj["selected"] != min(entries, key=order):
        raise RuntimeError("Selected candidate is not the frozen-rule winner: " + str(p))
    return obj


def all_search_complete(args):
    return {(d, s): validate_selection(args, d, s) for d in DATASETS for s in SPECS}


def final_case(args, dataset, spec):
    obj = validate_selection(args, dataset, spec)
    settings = obj["selected"]["settings"]
    records = []
    for seed in SEEDS:
        run, out = paths(args, dataset, spec, settings, seed, False)
        if seed == 2026:
            run = Path(obj["selected"]["run_dir"])
        else:
            ensure_training(args, dataset, spec, settings, seed, run)
        if args.dry_run:
            continue
        evaluate(args, run, out, TEST, "test", Path(args.results_root) / "logs" / (out.stem + ".log"))
        records.append(dict(seed=seed, **receipt(args, run, out, TEST, "test")))
    if not args.dry_run:
        write_exact(Path(args.results_root) / "final_manifests" / (dataset + "_" + spec + ".json"),
                    dict(dataset=dataset, specification=spec, model=SPECS[spec][0],
                         selected_settings=settings, selection_sha256=sha256(selection_path(args, dataset, spec)),
                         protocol_sha256=protocol_hash(args), runs=records))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("Refusing an empty summary: " + str(path))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(args):
    selections = all_search_complete(args)
    all_rows, tuning_rows, selected_rows = [], [], []
    for (d, s), sel in selections.items():
        selected = sel["selected"]
        for e in sel["candidates"]:
            row = dict(dataset=d, specification=s, model=SPECS[s][0], role=SPECS[s][3],
                       **e["settings"], **{k: e[k] for k in scores_keys()})
            row["selected"] = e["candidate"] == selected["candidate"]
            tuning_rows.append(row)
        selected_rows.append(dict(dataset=d, specification=s, model=SPECS[s][0],
                                  **selected["settings"], **{k: selected[k] for k in scores_keys()}))
        mf = Path(args.results_root) / "final_manifests" / (d + "_" + s + ".json")
        manifest = read_json(mf)
        expected = dict(dataset=d, specification=s, model=SPECS[s][0], selected_settings=selected["settings"],
                        selection_sha256=sha256(selection_path(args, d, s)), protocol_sha256=protocol_hash(args))
        if any(manifest.get(k) != v for k, v in expected.items()) or [r.get("seed") for r in manifest["runs"]] != list(SEEDS):
            raise RuntimeError("Invalid final manifest: " + str(mf))
        for record in manifest["runs"]:
            seed = record["seed"]
            run, out = paths(args, d, s, selected["settings"], seed, False)
            if seed == 2026:
                run = Path(selected["run_dir"])
            if not validate_run(run, d, s, seed, selected["settings"]):
                raise RuntimeError("Invalid final training run: " + str(run))
            if record != dict(seed=seed, **receipt(args, run, out, TEST, "test")):
                raise RuntimeError("Final output/checkpoint differs from manifest: " + str(out))
            if read_json(out.with_suffix(".receipt.json")) != receipt(args, run, out, TEST, "test"):
                raise RuntimeError("Final evaluation receipt mismatch: " + str(out))
            rows = validate_csv(out, d, SPECS[s][0], seed, TEST, "test")
            all_rows.extend(dict(r, specification=s) for r in rows)
    seed_rows = []
    for d in DATASETS:
        for s in SPECS:
            for seed in SEEDS:
                part = [r for r in all_rows if r["dataset"] == d and r["specification"] == s and int(r["seed"]) == seed]
                for group, conditions in (("clean", ("clean",)), ("known", VALIDATION[1:]), ("unseen", UNSEEN)):
                    selected = [r for r in part if r["condition"] in conditions]
                    seed_rows.append(dict(dataset=d, specification=s, model=SPECS[s][0], seed=seed,
                                          group=group, parameters=int(part[0]["parameters"]),
                                          **{k: sum(float(r[k]) for r in selected) / len(selected) for k in METRICS}))
    summary = []
    for d in DATASETS:
        for s in SPECS:
            for group in ("clean", "known", "unseen"):
                part = [r for r in seed_rows if r["dataset"] == d and r["specification"] == s and r["group"] == group]
                if len({r["parameters"] for r in part}) != 1:
                    raise RuntimeError("Parameter count differs across seeds: " + d + "/" + s)
                out = dict(dataset=d, specification=s, model=SPECS[s][0], group=group,
                           n_seeds=len(part), parameters=part[0]["parameters"])
                for k in METRICS:
                    vals = [r[k] for r in part]
                    mean = sum(vals) / len(vals)
                    out[k + "_mean"] = mean
                    out[k + "_std"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                summary.append(out)
    root = Path(args.results_root)
    for name, rows in (("tuning_candidates.csv", tuning_rows), ("selected_hyperparameters.csv", selected_rows),
                       ("all_condition_rows.csv", all_rows), ("seed_group_means.csv", seed_rows),
                       ("three_seed_summary.csv", summary)):
        write_csv(root / name, rows)
    audit = dict(complete=len(tuning_rows) == 56 and len(all_rows) == 462,
                 search_candidates=len(tuning_rows), selected_configurations=len(selected_rows),
                 selected_test_files=len(all_rows) // len(TEST), condition_rows=len(all_rows),
                 protocol_sha256=protocol_hash(args))
    write_exact(root / "FINAL_AUDIT.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def scores_keys():
    return ("validation_crps_mean", "clean_validation_crps", "clean_validation_mae")


def status(args):
    for lane, cases in LANES.items():
        search = selections = final = 0
        for d, s in cases:
            for settings in candidates(d, s):
                run, out = paths(args, d, s, settings, 2026, True)
                search += int((run / "selection.json").is_file() and out.is_file())
            selections += int(selection_path(args, d, s).is_file())
            for seed in SEEDS:
                _, out = paths(args, d, s, candidates(d, s)[0], seed, False)
                final += int(out.is_file())
        print("GPU lane %d: search outputs %d/%d; selections %d/%d; test files %d/%d" %
              (lane, search, len(cases) * 4, selections, len(cases), final, len(cases) * 3), flush=True)
    print("Counts are progress indicators; only summarize performs the final integrity audit.", flush=True)


class LaneLock:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise RuntimeError("Lane is already locked; do not launch twice: " + str(self.path)) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *unused):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def parse():
    p = argparse.ArgumentParser(description="CEE equal-budget tuning audit; search must finish on BOTH GPUs before final")
    p.add_argument("--stage", choices=("plan", "search", "final", "status", "summarize"), default="plan")
    p.add_argument("--lane", type=int, choices=(0, 1), default=0)
    p.add_argument("--code-root", default=".")
    p.add_argument("--data-root", default="datasets")
    p.add_argument("--search-root", default="runs/c5_tuning_search")
    p.add_argument("--final-root", default="runs/c5_tuned_final")
    p.add_argument("--results-root", default="results/c5_tuning")
    p.add_argument("--device", default="cuda")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    os.chdir(Path(args.code_root).resolve())
    if args.stage == "status":
        status(args)
        return
    write_exact(Path(args.results_root) / "FROZEN_PROTOCOL.json", protocol(args))
    if args.stage == "plan":
        print(json.dumps(dict(lane=args.lane, cases=LANES[args.lane],
                              lane_search_runs=len(LANES[args.lane]) * 4,
                              lane_additional_final_training=len(LANES[args.lane]) * 2,
                              total_search_runs=56, total_additional_final_training=28,
                              total_training_runs=84, total_test_files=42), indent=2))
        return
    if args.stage == "summarize":
        summarize(args)
        return
    if args.stage == "final":
        all_search_complete(args)
    with LaneLock(Path(args.results_root) / "locks" / ("lane%d.lock" % args.lane)):
        for d, s in LANES[args.lane]:
            if args.stage == "search":
                search_case(args, d, s)
            else:
                final_case(args, d, s)
    print("LANE %d %s COMPLETE" % (args.lane, args.stage.upper()), flush=True)


if __name__ == "__main__":
    main()
