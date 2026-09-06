#!/usr/bin/env python3
"""Read-only, configuration-aware audit for the KAMO CEE revision.

Mixer ranks, sensitivity settings and horizons remain distinct configurations.
The script never trains a model and never changes an existing result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REQUIRED_METRICS = (
    "mae", "crps", "nll", "picp_95", "mpiw_95", "interval_score_95",
    "picp_99", "mpiw_99", "interval_score_99",
)
STAGES = (
    "gate", "role_ablation", "long", "transfer", "capacity", "highdim",
    "real_missing", "mixer", "sensitivity",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit KAMO revision artifacts without training")
    p.add_argument("--code-root", default=".")
    p.add_argument("--results-root", required=True)
    p.add_argument("--runs-root", action="append", required=True)
    p.add_argument("--extra-results-root", action="append", default=[])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--hash-checkpoints", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError) as error:
        return [], ["unreadable CSV: %s" % error]
    issues: List[str] = []
    if not rows:
        return rows, ["empty CSV"]
    for metric in REQUIRED_METRICS:
        if metric not in rows[0]:
            issues.append("missing column %s" % metric)
        elif not all(finite(row.get(metric)) for row in rows):
            issues.append("non-finite %s" % metric)
    return rows, issues


def exact_config(config: Dict[str, object], job: object, defaults: Dict[str, str]) -> bool:
    try:
        fixed = (
            config.get("dataset") == job.dataset
            and config.get("model") == job.model
            and int(config.get("seed", -1)) == int(job.seed)
            and int(config.get("seq_len", -1)) == int(job.seq_len)
            and int(config.get("pred_len", -1)) == int(job.pred_len)
            and not bool(config.get("smoke_test", False))
            and config.get("max_train_batches") is None
            and config.get("max_val_batches") is None
            and config.get("max_test_batches") is None
        )
        values = all(
            str(config.get(key.replace("-", "_"))) == value
            for key, value in defaults.items()
        )
        return fixed and values
    except (TypeError, ValueError):
        return False


def training_ok(run_dir: Path, job: object, suite: object) -> Tuple[bool, str]:
    if not (run_dir / "best.pt").is_file():
        return False, "best.pt missing"
    metrics = load_json(run_dir / "metrics.json")
    selection = load_json(run_dir / "selection.json")
    if metrics is not None and suite.training_evidence_complete(job, metrics=metrics):
        return True, "test metrics"
    if selection is not None and suite.training_evidence_complete(job, selection=selection):
        return True, "validation-only selection"
    return False, "missing valid metrics/selection evidence"


def clean_metrics_ok(run_dir: Path, job: object) -> bool:
    metrics = load_json(run_dir / "metrics.json")
    if metrics is None:
        return False
    try:
        return (
            metrics.get("dataset") == job.dataset
            and metrics.get("model") == job.model
            and int(metrics.get("seed", -1)) == int(job.seed)
            and all(finite(metrics.get(key)) for key in ("mae", "crps", "nll"))
        )
    except (TypeError, ValueError):
        return False


def protocol_issues(
    rows: Sequence[Dict[str, str]], job: object, conditions: Sequence[str], calibration: str
) -> List[str]:
    issues: List[str] = []
    if len(rows) != len(conditions):
        issues.append("condition row count %d != %d" % (len(rows), len(conditions)))
    if {row.get("condition") for row in rows} != set(conditions):
        issues.append("condition set mismatch")
    if any(row.get("dataset") != job.dataset for row in rows):
        issues.append("dataset mismatch")
    if any(row.get("model") != job.model for row in rows):
        issues.append("model mismatch")
    if any(str(row.get("seed")) != str(job.seed) for row in rows):
        issues.append("seed mismatch")
    if any(str(row.get("evaluation_split", "test")).lower() != "test" for row in rows):
        issues.append("not a test split")
    if any(str(row.get("full_test", "")).lower() != "true" for row in rows):
        issues.append("full_test is not true")
    if any(row.get("calibration_type", "none") != calibration for row in rows):
        issues.append("calibration type mismatch")
    return issues


def zip_add(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    if path.is_file():
        archive.write(path, arcname=name)


def main() -> int:
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(code_root))
    import run_revision_suite as suite  # type: ignore

    roots: List[Path] = []
    for raw in args.runs_root:
        root = Path(raw).resolve()
        if root.exists() and root not in roots:
            roots.append(root)
    configs: List[Tuple[Path, Dict[str, object], Path]] = []
    for root in roots:
        for path in root.rglob("config.json"):
            payload = load_json(path)
            if payload is not None:
                configs.append((path.parent.resolve(), payload, root))

    stage_jobs = {
        "gate": suite.gate_jobs(),
        "role_ablation": suite.role_ablation_jobs(),
        "long": suite.long_horizon_jobs(),
        "transfer": suite.transfer_jobs(),
        "capacity": suite.capacity_jobs(),
        "highdim": suite.high_dimensional_jobs(False),
        "real_missing": suite.real_missing_jobs(),
        "mixer": suite.mixer_jobs(),
        "sensitivity": suite.sensitivity_jobs(),
    }
    conditions = tuple(suite.STANDARD_AND_UNSEEN_CONDITIONS)
    inventory: List[Dict[str, object]] = []
    stages: List[Dict[str, object]] = []
    canonical_files: List[Path] = []
    chosen_runs: List[Path] = []
    blockers: List[str] = []
    warnings: List[str] = []

    for stage in STAGES:
        expected = stage_jobs[stage]
        n_train = n_result = n_review = n_suite = 0
        for job in expected:
            defaults = suite.model_defaults(job)
            candidates: List[Tuple[Path, Path]] = []
            seen = set()
            for run_dir, config, source_root in configs:
                if str(run_dir) in seen or not exact_config(config, job, defaults):
                    continue
                complete, _ = training_ok(run_dir, job, suite)
                if complete:
                    candidates.append((run_dir, source_root))
                    seen.add(str(run_dir))
            train_complete = bool(candidates)
            n_train += int(train_complete)
            if candidates:
                chosen_runs.append(candidates[0][0])
            if len(candidates) > 1:
                warnings.append(
                    "%s/%s/%s/seed%s has %d exact completed run directories"
                    % (stage, job.dataset, job.model, job.seed, len(candidates))
                )

            result_family, mode = suite.evaluation_protocol(job)
            result = (
                results_root / "robustness" / result_family
                / ("%s_%s_%s.csv" % (job.dataset, job.model, job.run_name))
            )
            calibration = "global_std_temperature_multimetric" if mode == "std-multimetric-auto" else "none"
            rows: List[Dict[str, str]] = []
            issues: List[str] = []
            if result.is_file():
                rows, issues = read_csv(result)
                issues.extend(protocol_issues(rows, job, conditions, calibration))
            else:
                issues.append("result CSV missing")
            result_complete = not issues
            n_result += int(result_complete)
            if result_complete:
                canonical_files.append(result)

            clean_complete = bool(candidates) and clean_metrics_ok(candidates[0][0], job)
            review_complete = train_complete and (clean_complete if stage == "long" else result_complete)
            suite_complete = train_complete and result_complete
            n_review += int(review_complete)
            n_suite += int(suite_complete)
            variant = job.run_name.split("_h%d" % job.pred_len, 1)[-1].lstrip("_") or "default"
            inventory.append({
                "stage": stage, "dataset": job.dataset, "model": job.model,
                "seed": job.seed, "seq_len": job.seq_len, "pred_len": job.pred_len,
                "variant": variant, "run_name": job.run_name,
                "training_complete": train_complete, "exact_run_count": len(candidates),
                "selected_run_dir": str(candidates[0][0]) if candidates else "",
                "selected_run_root": str(candidates[0][1]) if candidates else "",
                "clean_metrics_complete": clean_complete, "result_csv": str(result),
                "result_valid": result_complete,
                "result_sha256": sha256(result) if result_complete else "",
                "review_evidence_complete": review_complete,
                "full_suite_complete": suite_complete, "issues": "; ".join(issues),
            })
        row = {
            "stage": stage, "expected_jobs": len(expected),
            "training_complete": n_train, "valid_full_test_csv": n_result,
            "review_evidence_complete": n_review, "full_suite_complete": n_suite,
            "missing_for_review": len(expected) - n_review,
            "missing_for_full_suite": len(expected) - n_suite,
        }
        stages.append(row)
        if n_review != len(expected):
            blockers.append("%s: %d/%d reviewer-relevant configurations complete" % (stage, n_review, len(expected)))

    efficiency = results_root / "efficiency" / "parameter_matched_gpu0_optimized.csv"
    efficiency_rows: List[Dict[str, str]] = []
    efficiency_issues: List[str] = []
    if efficiency.is_file():
        efficiency_rows, efficiency_issues = read_csv(efficiency)
        # read_csv checks probabilistic columns, inappropriate for efficiency.
        efficiency_issues = [x for x in efficiency_issues if not x.startswith("missing column")]
        if len(efficiency_rows) != 6:
            efficiency_issues.append("row count %d != 6" % len(efficiency_rows))
        for key in ("parameters", "latency_ms_median", "samples_per_second", "peak_memory_mb"):
            if efficiency_rows and key not in efficiency_rows[0]:
                efficiency_issues.append("missing %s" % key)
            elif efficiency_rows and not all(finite(row.get(key)) for row in efficiency_rows):
                efficiency_issues.append("non-finite %s" % key)
        for key in ("device_name", "torch_version", "precision", "timing_scope"):
            if efficiency_rows and key not in efficiency_rows[0]:
                efficiency_issues.append("missing provenance %s" % key)
    else:
        efficiency_issues.append("optimized efficiency CSV missing")
    if efficiency_issues:
        blockers.append("efficiency: " + "; ".join(efficiency_issues))
    else:
        canonical_files.append(efficiency)

    tests: Dict[str, object] = {"requested": bool(args.run_tests), "returncode": None, "pass_lines": 0, "fail_lines": 0, "log": ""}
    if args.run_tests:
        run = subprocess.run(
            [sys.executable, "test_revision_invariants.py"], cwd=str(code_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log = output / "invariant_tests.log"
        log.write_text(run.stdout, encoding="utf-8")
        tests.update({
            "returncode": run.returncode,
            "pass_lines": sum(x.startswith("PASS ") for x in run.stdout.splitlines()),
            "fail_lines": sum(x.startswith("FAIL ") for x in run.stdout.splitlines()),
            "log": str(log),
        })
        if run.returncode != 0 or tests["fail_lines"]:
            blockers.append("implementation invariant tests failed")

    extras: List[Dict[str, object]] = []
    for raw in args.extra_results_root:
        root = Path(raw).resolve()
        if not root.exists():
            extras.append({"root": str(root), "exists": False, "csv": "", "sha256": ""})
        else:
            for path in root.rglob("*.csv"):
                extras.append({"root": str(root), "exists": True, "csv": str(path), "sha256": sha256(path)})

    unique_runs: List[Path] = []
    seen_runs = set()
    for run_dir in chosen_runs:
        if str(run_dir) not in seen_runs:
            unique_runs.append(run_dir)
            seen_runs.add(str(run_dir))
    checkpoints: List[Dict[str, object]] = []
    for run_dir in unique_runs:
        path = run_dir / "best.pt"
        checkpoints.append({
            "run_dir": str(run_dir), "checkpoint": str(path),
            "bytes": path.stat().st_size if path.is_file() else "",
            "sha256": sha256(path) if args.hash_checkpoints and path.is_file() else "not_computed",
        })

    by_stage = {row["stage"]: row for row in stages}
    complete = lambda stage: by_stage[stage]["missing_for_review"] == 0
    all_prob = all(row["valid_full_test_csv"] == row["expected_jobs"] for row in stages if row["stage"] != "long")
    coverage = [
        {"reviewer": "R1", "comment": 1, "status": "PARTIAL", "evidence": "gate, role, capacity, transfer", "remaining": "rewrite novelty as coordinated capability; individual blocks are not claimed new"},
        {"reviewer": "R1", "comment": 2, "status": "COMPLETE" if complete("gate") else "INCOMPLETE", "evidence": "shared/separate/mask-only parameter-matched gate", "remaining": "report noninferiority and mixed results, not necessity"},
        {"reviewer": "R1", "comment": 3, "status": "PARTIAL" if complete("real_missing") else "INCOMPLETE", "evidence": "PhysioNet 2012 native missing-observation forecasting", "remaining": "do not call native missingness verified sensor faults; document set-level split"},
        {"reviewer": "R1", "comment": 4, "status": "COMPLETE" if complete("long") else "INCOMPLETE", "evidence": "ETTh2 and Weather H=192/336/720, five models, three seeds", "remaining": "retain unfavorable rankings and report sample SD"},
        {"reviewer": "R1", "comment": 5, "status": "PARTIAL", "evidence": "parameter-matched capacity controls and optimized efficiency benchmark", "remaining": "document tuning provenance; controlled adaptations are not official reproductions"},
        {"reviewer": "R1", "comment": 6, "status": "COMPLETE" if complete("transfer") else "INCOMPLETE", "evidence": "soft reliability adapters for PatchTST/iTransformer", "remaining": "present backbone-dependent mixed gains"},
        {"reviewer": "R1", "comment": 7, "status": "COMPLETE" if complete("role_ablation") else "INCOMPLETE", "evidence": "four one-role removals", "remaining": "claim complementarity, not universal benefit of every role"},
        {"reviewer": "R1", "comment": 8, "status": "COMPLETE" if complete("sensitivity") else "INCOMPLETE", "evidence": "five factors, two alternatives, three seeds", "remaining": "state one-factor-at-a-time protocol"},
        {"reviewer": "R1", "comment": 9, "status": "COMPLETE" if complete("mixer") else "INCOMPLETE", "evidence": "none/dense/ranks 4,8,16,32", "remaining": "state rank=min(requested, variables); ETTh2 ranks 8/16/32 are equivalent"},
        {"reviewer": "R1", "comment": 10, "status": "COMPLETE" if complete("capacity") else "INCOMPLETE", "evidence": "clean/known/unseen matched-capacity results", "remaining": "quantify the trade-off"},
        {"reviewer": "R1", "comment": 11, "status": "COMPLETE" if all_prob else "INCOMPLETE", "evidence": "condition-level NLL/PICP/MPIW/interval scores", "remaining": "main compact table plus full supplement"},
        {"reviewer": "R1", "comment": 12, "status": "COMPLETE" if complete("gate") and complete("capacity") else "INCOMPLETE", "evidence": "level shift, drift, freeze, heavy tail", "remaining": "keep known and unseen groups separate"},
        {"reviewer": "R1", "comment": 13, "status": "PAPER_ACTION", "evidence": "aggregate mechanism audit", "remaining": "simplify Fig.1 and generate direct sample-level reliability-routing-forecast panel"},
        {"reviewer": "R1", "comment": 14, "status": "COMPLETE" if complete("highdim") else "INCOMPLETE", "evidence": "Electricity high-dimensional three-seed experiment", "remaining": "report variable count and scaling"},
        {"reviewer": "R4", "comment": "tail risk", "status": "COMPLETE" if all_prob else "INCOMPLETE", "evidence": "heavy-tail corruption and 99% interval metrics", "remaining": "define tail-risk analysis and avoid overclaiming"},
    ]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "read_only_audit": True,
        "canonical_results_root": str(results_root),
        "searched_run_roots": [str(x) for x in roots],
        "noncanonical_results_roots": [str(Path(x).resolve()) for x in args.extra_results_root],
        "artifact_audit_status": "PASS" if not blockers else "INCOMPLETE",
        "review_readiness": "READY_FOR_PAPER_EDIT" if not blockers else "ADDITIONAL_EVIDENCE_REQUIRED",
        "blockers": blockers, "warnings": warnings, "stage_summary": stages,
        "efficiency": {"path": str(efficiency), "valid": not efficiency_issues, "issues": efficiency_issues},
        "tests": tests, "reviewer_coverage": coverage,
        "scientific_limits": [
            "Completion is not statistical significance.",
            "Three seeds support mean and sample SD, not universal superiority.",
            "Known and unseen corruptions must remain separate.",
            "PhysioNet native missing observations are not sensor-fault labels.",
            "The shared gate is an engineering screen, not a formal noninferiority test.",
            "Non-canonical duplicate roots are never pooled as extra seeds.",
        ],
    }

    write_csv(output / "configuration_inventory.csv", inventory)
    write_csv(output / "stage_completeness.csv", stages)
    write_csv(output / "reviewer_coverage.csv", coverage)
    write_csv(output / "noncanonical_results_inventory.csv", extras)
    write_csv(output / "checkpoint_inventory.csv", checkpoints)
    (output / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "KAMO CEE FORMAL REVISION AUDIT",
        "artifact_audit_status: " + report["artifact_audit_status"],
        "review_readiness: " + report["review_readiness"],
        "canonical_results_root: " + str(results_root), "", "STAGE COMPLETENESS",
    ]
    lines.extend(
        "{stage}: review {review_evidence_complete}/{expected_jobs}; full suite {full_suite_complete}/{expected_jobs}".format(**row)
        for row in stages
    )
    lines.extend(["", "BLOCKERS"] + (["- " + x for x in blockers] if blockers else ["- none"]))
    lines.extend(["", "WARNINGS"] + (["- " + x for x in warnings] if warnings else ["- none"]))
    (output / "AUDIT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    packet = output / "KAMO_revision_evidence_packet.zip"
    with zipfile.ZipFile(packet, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "AUDIT_SUMMARY.txt", "audit_report.json", "configuration_inventory.csv",
            "stage_completeness.csv", "reviewer_coverage.csv",
            "noncanonical_results_inventory.csv", "checkpoint_inventory.csv",
            "invariant_tests.log",
        ):
            zip_add(archive, output / name, "audit/" + name)
        for path in canonical_files:
            try:
                rel = path.relative_to(results_root)
            except ValueError:
                rel = Path(path.name)
            zip_add(archive, path, "canonical_results/" + str(rel).replace("\\", "/"))
        for run_dir in unique_runs:
            label = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:12]
            for name in ("config.json", "metrics.json", "selection.json", "history.json", "calibration.json", "std_multimetric_calibration.json"):
                zip_add(archive, run_dir / name, "run_metadata/%s/%s" % (label, name))
        for path in sorted(code_root.glob("*.py")) + sorted((code_root / "kamo").glob("*.py")):
            if path.is_file():
                zip_add(archive, path, "code/" + str(path.relative_to(code_root)).replace("\\", "/"))

    print("AUDIT STATUS:", report["artifact_audit_status"])
    print("REVIEW READINESS:", report["review_readiness"])
    for row in stages:
        print("{stage}: review {review_evidence_complete}/{expected_jobs}; suite {full_suite_complete}/{expected_jobs}".format(**row))
    print("BLOCKERS:", len(blockers))
    for item in blockers:
        print("-", item)
    print("WARNINGS:", len(warnings))
    for item in warnings[:20]:
        print("-", item)
    if len(warnings) > 20:
        print("- ... %d more warnings in audit_report.json" % (len(warnings) - 20))
    print("EVIDENCE PACKET:", packet)
    return 2 if args.strict and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
