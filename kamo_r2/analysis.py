"""Auditable R2 summaries and paired dependence-aware uncertainty estimates.

Run: python -m kamo_r2.analysis --results runs --output analysis --reference kamo_s
Confidence intervals are descriptive, conditional on the fitted training seeds.
Training-seed variation is separately tabulated, never counted as extra windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def paired_block_bootstrap(reference, comparator, counts=None, block_length=1,
                           n_boot=2000, seed=20261008, group_ids=None, confidence=0.95):
    """Resample paired windows or complete patient clusters, preserving weights.

    Positive relative_gain_pct means the reference has lower loss. Circular
    moving-block bootstrap assumes a weakly stationary dependent score process;
    patient bootstrap assumes independent patient clusters. Neither is a proof
    of independence, nor a correction for post-selection/multiple comparisons.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(comparator, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Need two aligned loss vectors with at least two windows")
    weights = np.ones(len(a)) if counts is None else np.asarray(counts, dtype=float)
    if weights.shape != a.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Invalid target observation weights")
    valid = weights > 0
    if not valid.any() or not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all():
        raise ValueError("Nonfinite score on a positive-weight window")
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError("Use confidence in (0,1) and at least 100 bootstrap draws")
    aw = np.where(valid, a, 0) * weights
    bw = np.where(valid, b, 0) * weights
    rng = np.random.RandomState(seed)
    grouped = group_ids is not None and np.any(np.asarray(group_ids) >= 0)
    numerator_a, numerator_b, denominators = [], [], []
    if grouped:
        groups = np.asarray(group_ids)
        if groups.shape != a.shape or np.any(groups < 0):
            raise ValueError("Patient group IDs must cover every window")
        ids, inverse = np.unique(groups, return_inverse=True)
        if len(ids) < 2:
            raise ValueError("At least two patient clusters required")
        ga = np.bincount(inverse, weights=aw)
        gb = np.bincount(inverse, weights=bw)
        gw = np.bincount(inverse, weights=weights)
        for start in range(0, n_boot, 128):
            choices = rng.randint(len(ids), size=(min(128, n_boot - start), len(ids)))
            numerator_a.extend(ga[choices].sum(axis=1))
            numerator_b.extend(gb[choices].sum(axis=1))
            denominators.extend(gw[choices].sum(axis=1))
        method, effective_units = "patient_cluster_observation_weighted", len(ids)
    else:
        block_length = int(block_length)
        if not 1 <= block_length < len(a):
            raise ValueError("Block length must be at least one and smaller than window count")
        n = len(a)
        full_blocks, remainder = divmod(n, block_length)
        def sums(length, values):
            extended = np.concatenate([values, values[:length]])
            cumulative = np.concatenate([[0.0], np.cumsum(extended, dtype=np.float64)])
            return cumulative[np.arange(n) + length] - cumulative[np.arange(n)]
        sa, sb, sw = [sums(block_length, v) for v in (aw, bw, weights)]
        ra, rb, rw = [sums(remainder, v) for v in (aw, bw, weights)]
        for start in range(0, n_boot, 128):
            draws = min(128, n_boot - start)
            choices = rng.randint(n, size=(draws, full_blocks))
            last = rng.randint(n, size=draws)
            numerator_a.extend(sa[choices].sum(axis=1) + ra[last])
            numerator_b.extend(sb[choices].sum(axis=1) + rb[last])
            denominators.extend(sw[choices].sum(axis=1) + rw[last])
        method, effective_units = "circular_moving_block_observation_weighted", n / block_length
    denominator = np.asarray(denominators)
    accepted = denominator > 0
    aa = np.asarray(numerator_a)[accepted] / denominator[accepted]
    bb = np.asarray(numerator_b)[accepted] / denominator[accepted]
    if accepted.sum() < 0.9 * n_boot:
        raise ValueError("Too many empty bootstrap samples; report data sparsity")
    difference = aa - bb
    mean_a, mean_b = float(aw.sum() / weights.sum()), float(bw.sum() / weights.sum())
    alpha = (1 - confidence) / 2
    interval = np.quantile(difference, [alpha, 1 - alpha])
    result = {"reference_mean": mean_a, "comparator_mean": mean_b,
              "difference_reference_minus_comparator": mean_a - mean_b,
              "difference_ci_low": float(interval[0]), "difference_ci_high": float(interval[1]),
              "relative_gain_pct": 100 * (mean_b - mean_a) / mean_b if mean_b > 0 else None,
              "method": method, "block_length": None if grouped else block_length,
              "resampling_units": float(effective_units), "bootstrap_draws": int(accepted.sum()),
              "confidence": confidence, "limited_effective_units": bool(effective_units < 10)}
    if np.all(bb > 0):
        gain_interval = np.quantile(100 * (bb - aa) / bb, [alpha, 1 - alpha])
        result.update(relative_gain_ci_low=float(gain_interval[0]), relative_gain_ci_high=float(gain_interval[1]))
    return result


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _record(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("complete") or not (path.parent / "window_metrics.npz").exists():
        return None
    # Explicit metadata is preferred. Ancestor config files are supported for
    # standalone evaluator users and older runner directory layouts.
    metadata = {}
    for parent in reversed(list(path.parents)[:4]):
        for filename in ("config.json", "run.json", "metadata.json"):
            config = parent / filename
            if config.exists():
                found = json.loads(config.read_text(encoding="utf-8"))
                metadata.update({k: v for k, v in found.items() if isinstance(v, (str, int, float, bool))})
    metadata.update(data)
    metadata["num_vars"] = metadata.get("num_vars", metadata.get("dimensions", "unspecified"))
    required = ("dataset", "model", "seed", "pred_len", "seq_len")
    if any(key not in metadata for key in required):
        # Fail visibly instead of accidentally pooling datasets/models.
        raise ValueError("Evaluation metrics missing metadata %s: %s" %
                         ([key for key in required if key not in metadata], path))
    metadata["variant"] = metadata.get("variant", "raw" if float(metadata.get("std_temperature", 1)) == 1 else "calibrated")
    metadata["source"] = str(path)
    return metadata


def _keys(record, with_model=True, with_condition=True):
    names = ["dataset", "seq_len", "pred_len", "num_vars", "variant"]
    if with_condition:
        names.append("condition")
    if with_model:
        names.append("model")
    return tuple(record[k] for k in names)


def _identifiers(record):
    return {key: record[key] for key in ("dataset", "seq_len", "pred_len", "num_vars", "variant", "condition", "model")}


def _load_arrays(record):
    with np.load(Path(record["source"]).parent / "window_metrics.npz", allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _assert_aligned(a, b):
    for key in ("window_id", "time_start", "target_count", "target_hash"):
        if key not in a or key not in b or not np.array_equal(a[key], b[key]):
            raise ValueError("Paired comparison requires identical %s" % key)
    if ("group_id" in a) != ("group_id" in b):
        raise ValueError("Patient-group metadata differs")
    if "group_id" in a and not np.array_equal(a["group_id"], b["group_id"]):
        raise ValueError("Patient-group metadata differs")


def _default_block(arrays, seq_len, pred_len):
    # Window overlap alone gives a conservative minimum; users must inspect
    # longer score autocorrelation and report the accompanying sensitivity rows.
    starts = arrays["time_start"]
    differences = np.diff(starts)
    positive = differences[differences > 0]
    stride = float(np.median(positive)) if len(positive) else 1.0
    overlap = int(math.ceil((int(seq_len) + int(pred_len)) / stride))
    n = len(starts)
    return min(max(1, overlap, int(math.ceil(n ** (1 / 3)))), n - 1)


def summarize_results(results_root, output_dir=None, reference_model="kamo_s",
                      bootstrap_replicates=2000, block_length=None):
    results_root = Path(results_root)
    destination = Path(output_dir) if output_dir else results_root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(results_root.rglob("metrics.json")):
        if destination in path.parents:
            continue
        value = _record(path)
        if value is not None:
            records.append(value)
    if not records:
        raise ValueError("No complete evaluation metrics with window arrays found")
    seen = set()
    for record in records:
        identity = (_keys(record), int(record["seed"]))
        if identity in seen:
            raise ValueError("Duplicate result identity; do not pool multiple checkpoints: " + str(identity))
        seen.add(identity)
    _write_csv(destination / "all_condition_metrics.csv", records)
    groups = defaultdict(list)
    for record in records:
        groups[_keys(record)].append(record)
    summaries = []
    for members in groups.values():
        first = members[0]
        row = _identifiers(first)
        row.update(n_seeds=len(members), seeds=";".join(str(v["seed"]) for v in members))
        numeric = [k for k, v in first.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and k not in {"seed", "pred_len", "seq_len", "num_vars", "corruption_seed"}]
        for key in numeric:
            values = np.asarray([r[key] for r in members], dtype=float)
            row[key + "_mean"] = float(values.mean())
            row[key + "_seed_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summaries.append(row)
    _write_csv(destination / "seed_summary.csv", summaries)

    comparisons, diagnostic_rows, benefit_distributions = [], [], []
    by_context = defaultdict(dict)
    for key, members in groups.items():
        by_context[key[:-1]][key[-1]] = members
    for context, methods in by_context.items():
        if reference_model not in methods:
            continue
        reference = {int(r["seed"]): r for r in methods[reference_model]}
        for comparator_name, comparator_records in methods.items():
            if comparator_name == reference_model:
                continue
            comparison = {int(r["seed"]): r for r in comparator_records}
            seeds = sorted(set(reference) & set(comparison))
            if not seeds:
                continue
            a_list, b_list, first_arrays = [], [], None
            for seed in seeds:
                ra, rb = reference[seed], comparison[seed]
                if ra["corruption_seed"] != rb["corruption_seed"]:
                    raise ValueError("Paired evaluation corruption seeds differ")
                aa, bb = _load_arrays(ra), _load_arrays(rb)
                _assert_aligned(aa, bb)
                if first_arrays is not None:
                    _assert_aligned(first_arrays, aa)
                else:
                    first_arrays = aa
                a_list.append(aa["normalized_crps"])
                b_list.append(bb["normalized_crps"])
            counts = first_arrays["target_count"]
            a, b = np.mean(a_list, axis=0), np.mean(b_list, axis=0)
            benefit = (b - a)[counts > 0]
            distribution = _identifiers(reference[seeds[0]])
            distribution.update(comparator=comparator_name, n_paired_seeds=len(seeds), windows=len(benefit),
                                endpoint="normalized_crps_comparator_minus_reference",
                                weighting="equal nonempty windows", mean=float(np.mean(benefit)),
                                q05=float(np.quantile(benefit, 0.05)), median=float(np.median(benefit)),
                                q95=float(np.quantile(benefit, 0.95)), fraction_improved=float(np.mean(benefit > 0)),
                                fraction_worse=float(np.mean(benefit < 0)))
            benefit_distributions.append(distribution)
            block = int(block_length) if block_length is not None else _default_block(first_arrays, context[1], context[2])
            group_ids = first_arrays.get("group_id")
            clustered = group_ids is not None and np.any(group_ids >= 0)
            # Always export predeclared factor-of-two sensitivity, never select
            # a block length based on whether the interval excludes zero.
            lengths = [block] if clustered else sorted(set([max(1, block // 2), block, min(len(a) - 1, 2 * block)]))
            for length in lengths:
                statistics = paired_block_bootstrap(a, b, counts, length, bootstrap_replicates, group_ids=group_ids)
                row = _identifiers(reference[seeds[0]])
                row.update(comparator=comparator_name, endpoint="normalized_crps", paired_seeds=";".join(map(str, seeds)),
                           n_paired_seeds=len(seeds), block_role="primary" if length == block else "sensitivity",
                           training_uncertainty="CI conditional on these fitted seeds; seed SD reported separately")
                weights = counts / counts.sum()
                seed_effects = [float(np.nansum((aa - bb) * weights)) for aa, bb in zip(a_list, b_list)]
                row.update(statistics)
                row["seed_effect_std"] = float(np.std(seed_effects, ddof=1)) if len(seeds) > 1 else None
                comparisons.append(row)
    _write_csv(destination / "paired_normalized_crps_bootstrap.csv", comparisons)
    _write_csv(destination / "paired_window_improvement_distributions.csv", benefit_distributions)

    for record in records:
        arrays = None
        # Load only when diagnostics exist, while retaining all conditions/seeds.
        with np.load(Path(record["source"]).parent / "window_metrics.npz", allow_pickle=False) as handle:
            diag_keys = [key for key in handle.files if key.startswith("diag_") and (key.endswith("_mean") or "_component_" in key)]
            if diag_keys:
                arrays = {key: handle[key] for key in diag_keys + ["normalized_crps", "target_count"]}
        if arrays is None:
            continue
        for key in diag_keys:
            valid = (arrays["target_count"] > 0) & np.isfinite(arrays[key])
            values = arrays[key][valid]
            scores = arrays["normalized_crps"][valid]
            if not len(values):
                continue
            row = _identifiers(record)
            row.update(seed=record["seed"], diagnostic=key, windows=len(values), mean=float(values.mean()),
                       std=float(values.std()), q05=float(np.quantile(values, 0.05)), median=float(np.median(values)),
                       q95=float(np.quantile(values, 0.95)),
                       spearman_with_crps=float(spearmanr(values, scores).correlation) if np.ptp(values) > 0 and np.ptp(scores) > 0 else None)
            diagnostic_rows.append(row)
    _write_csv(destination / "all_window_mechanism_distributions.csv", diagnostic_rows)
    tradeoffs = _tradeoffs(summaries, reference_model)
    _write_csv(destination / "clean_robustness_tradeoff.csv", tradeoffs)
    plot_paths = _plots(summaries, tradeoffs, diagnostic_rows, destination)
    audit = {
        "records": len(records), "model_condition_groups": len(groups), "paired_rows": len(comparisons),
        "reference_model": reference_model, "primary_endpoint": "normalized_crps",
        "resampling": "paired circular time blocks; patient clusters when IDs supplied",
        "calibration": "clean reserved holdout constant Gaussian scale, frozen under corruption",
        "interpretation": ["Seed SD and conditional test-sample CI are different uncertainty sources.",
                           "Intervals are descriptive, not multiplicity-adjusted significance declarations.",
                           "Circular blocks assume weak stationarity; inspect score dependence and all block-length sensitivity rows.",
                           "Patient clusters use observation weights, not equal-patient scoring.",
                           "Synthetic level shift alters context only; future targets remain unchanged.",
                           "Mechanism-score correlations are descriptive, not causal evidence.",
                           "A partial set of seeds/models is an incomplete run, not a final submission table."],
        "plots": plot_paths,
    }
    (destination / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"output_dir": str(destination), "seed_summary": str(destination / "seed_summary.csv"),
            "paired_bootstrap": str(destination / "paired_normalized_crps_bootstrap.csv"), "audit": audit}


def _tradeoffs(summaries, reference_model):
    index = {(_keys(row)): row for row in summaries}
    rows = []
    for row in summaries:
        if row["model"] != reference_model or row["condition"] == "clean":
            continue
        for other in summaries:
            if other["model"] == reference_model or _keys(other, False) != _keys(row, False):
                continue
            clean_reference = dict(row, condition="clean")
            clean_other = dict(other, condition="clean")
            a, b = index.get(_keys(clean_reference)), index.get(_keys(clean_other))
            if not a or not b:
                continue
            denominator_clean, denominator_corrupt = b["normalized_crps_mean"], other["normalized_crps_mean"]
            if denominator_clean <= 0 or denominator_corrupt <= 0:
                continue
            result = _identifiers(row)
            result.update(comparator=other["model"],
                          clean_cost_pct=100 * (a["normalized_crps_mean"] - denominator_clean) / denominator_clean,
                          robustness_gain_pct=100 * (denominator_corrupt - row["normalized_crps_mean"]) / denominator_corrupt)
            rows.append(result)
    return rows


def _plots(summaries, tradeoffs, diagnostics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    contexts = defaultdict(list)
    for row in summaries:
        contexts[(row["dataset"], row["seq_len"], row["pred_len"], row["num_vars"], row["variant"])].append(row)
    for context, rows in contexts.items():
        label = "%s_L%s_H%s_D%s_%s" % context
        # Reliability and interval-width curves are separate views of the same
        # frozen predictive distributions, at multiple nominal coverage levels.
        conditions = sorted(set(row["condition"] for row in rows))
        selected = [v for v in conditions if v == "clean" or v in {"impulse:0.05:5", "random_missing:0.20", "level_shift:1.0", "level_shift:1"}]
        for condition in selected:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for row in [r for r in rows if r["condition"] == condition]:
                levels = sorted(int(k.split("_")[1]) for k in row if k.startswith("picp_") and k.endswith("_mean"))
                coverage = [row["picp_%d_mean" % k] for k in levels]
                width = [row["normalized_mpiw_%d_mean" % k] for k in levels]
                axes[0].plot(np.asarray(levels) / 100, coverage, marker=".", label=row["model"])
                axes[1].plot(width, coverage, marker=".", label=row["model"])
            axes[0].plot([0.5, 1], [0.5, 1], "k--", lw=1)
            axes[0].set(xlabel="Nominal marginal coverage", ylabel="Observed marginal coverage")
            axes[1].set(xlabel="Standardized interval width", ylabel="Observed marginal coverage")
            axes[0].legend(fontsize=7)
            fig.suptitle(label + " / " + condition)
            fig.tight_layout()
            path = destination / ("calibration_" + label + "_" + condition.replace(":", "_") + ".png")
            fig.savefig(str(path), dpi=160)
            plt.close(fig)
            paths.append(str(path))
        families = defaultdict(list)
        for row in rows:
            parts = row["condition"].split(":")
            if len(parts) > 1:
                # Impulse sweeps hold rate fixed and vary amplitude. Other
                # families vary their leading scale/fraction parameter.
                varying = float(parts[2]) if parts[0] == "impulse" and len(parts) > 2 else float(parts[1])
                families[parts[0]].append((varying, row))
        for family, points in families.items():
            if len(set(x for x, _ in points)) < 3:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in sorted(set(row["model"] for _, row in points)):
                values = sorted(((x, row) for x, row in points if row["model"] == model), key=lambda item: (item[0], item[1]["condition"]))
                x = [v[0] for v in values]
                y = [v[1]["normalized_crps_mean"] for v in values]
                sd = [v[1].get("normalized_crps_seed_std") or 0 for v in values]
                ax.errorbar(x, y, yerr=sd, marker=".", capsize=2, label=model)
            units = {"gaussian": "Noise std (training standardized units)", "impulse": "Impulse amplitude (standardized units; fixed hit rate)",
                     "level_shift": "Context-only offset (standardized units)", "gradual_drift": "Final context drift (standardized units)",
                     "heavy_tail": "Clipped Cauchy scale (standardized units)", "random_missing": "Missing probability", "block_missing": "Context block-length fraction", "sensor_freeze": "Affected-variable probability"}
            ax.set(xlabel=units.get(family, "Corruption severity"), ylabel="Standardized CRPS", title=label + " / " + family)
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = destination / ("severity_" + label + "_" + family + ".png")
            fig.savefig(str(path), dpi=160)
            plt.close(fig)
            paths.append(str(path))
    if tradeoffs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model in sorted(set(row["comparator"] for row in tradeoffs)):
            rows = [r for r in tradeoffs if r["comparator"] == model]
            ax.scatter([r["clean_cost_pct"] for r in rows], [r["robustness_gain_pct"] for r in rows], alpha=0.6, s=18, label=model)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(0, color="k", lw=0.7)
        ax.set(xlabel="Reference clean CRPS cost (%)", ylabel="Reference corrupted CRPS gain (%)", title="Each point: dataset, horizon and corruption; negative regions retained")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = destination / "clean_robustness_tradeoff.png"
        fig.savefig(str(path), dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument("--reference", default="kamo_s")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-length", type=int)
    args = parser.parse_args()
    result = summarize_results(args.results, args.output, args.reference, args.bootstrap, args.block_length)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
