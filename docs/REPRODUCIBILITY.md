# Reproducing the KAMO experiments

Run all commands from the repository root. Generated artifacts default to `runs/` and `results/`; both directories are ignored by Git.

## 1. Verify the code

```bash
python3 -m compileall kamo *.py
python3 test_revision_invariants.py
python3 test_c5_tuning_audit.py
```

The invariant tests check causal robust statistics, reliability sharing and ablations, missing-target masking, probabilistic metrics, seeded corruptions, calibration safeguards, and parameter-matching rules.

## 2. Inspect planned jobs

Every suite stage supports `--dry-run`:

```bash
python3 run_revision_suite.py \
  --stage gate \
  --data-root ./datasets \
  --runs-root ./runs \
  --results-root ./results \
  --dry-run
```

Use dry-run output to verify datasets, models, seeds, horizons, widths, and output locations before consuming GPU time.

## 3. Experiment stages

| Stage | Purpose |
|---|---|
| `gate` | shared vs separately learned vs mask-only reliability controls |
| `role_ablation` | normalization, routing, residual-gate, and uncertainty-scale roles |
| `long` | horizons 192, 336, and 720 with three seeds |
| `transfer` | reliability adapters on PatchTST and iTransformer backbones |
| `capacity` | parameter-matched robust PatchTST and TimeMixer controls |
| `highdim` | Electricity, optionally Traffic |
| `real_missing` | PhysioNet 2012 naturally missing observations |
| `mixer` | no mixer, low-rank values, and dense mixer |
| `sensitivity` | robust-window, Huber threshold, patch, recent-window, and scale settings |
| `postgate` | all post-gate stages |
| `all` | gate plus all post-gate stages; use only after reviewing the gate |

The predeclared reporting seeds are `2026,2027,2028`.

## 4. Run the gate first

```bash
python3 run_revision_suite.py \
  --stage gate \
  --data-root ./datasets \
  --runs-root ./runs \
  --results-root ./results \
  --seeds 2026,2027,2028 \
  --device cuda \
  --workers 4

python3 collect_revision_results.py --results-root ./results
python3 audit_revision_runs.py \
  --runs-root ./runs \
  --results-root ./results
```

The gate is a resource-management check, not a replacement for reporting seed-level outcomes and uncertainty.

## 5. Run later stages

After reviewing the gate:

```bash
python3 run_revision_suite.py \
  --stage postgate \
  --data-root ./datasets \
  --runs-root ./runs \
  --results-root ./results \
  --seeds 2026,2027,2028 \
  --device cuda \
  --workers 4
```

Individual stages can be run by replacing `postgate` with the stage name. Add `--include-traffic` to the high-dimensional stage only when the Traffic dataset has been prepared.

## 6. Safe multi-GPU sharding

Use independent terminals and disjoint filters. For example:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python3 run_revision_suite.py \
  --stage long --datasets ETTh2 --seeds 2026,2027 \
  --data-root ./datasets --runs-root ./runs --results-root ./results \
  --device cuda --workers 4

# GPU 1
CUDA_VISIBLE_DEVICES=1 python3 run_revision_suite.py \
  --stage long --datasets ETTh2 --seeds 2028 \
  --data-root ./datasets --runs-root ./runs --results-root ./results \
  --device cuda --workers 4
```

Do not launch overlapping dataset/model/seed jobs into the same run directory.

## 7. Validation-only tuning

The compact generic tuner is:

```bash
python3 tune_validation.py \
  --stage search \
  --data-root ./datasets \
  --search-root ./runs/tuning_search \
  --final-root ./runs/tuned_final \
  --output ./results/best_validation_configs.json \
  --device cuda
```

Search jobs use validation data only. Final test evaluation must occur only after configurations have been frozen. `run_c5_tuning_audit.py` preserves the more extensive equal-budget tuning and integrity-audit protocol used during revision; inspect its plan first with `--stage plan`.

## 8. Corruption evaluation

`evaluate_robustness.py` accepts explicit conditions. The paper protocol groups Gaussian noise, impulses, random missingness, and block missingness as known corruptions; level shift, gradual drift, sensor freezing, and heavy-tailed noise are held-out corruption families.

```bash
python3 evaluate_robustness.py \
  --run-dir ./runs/ETTh2/kamo_v5_shared/example_s2026_h96 \
  --data-root ./datasets \
  --split test \
  --calibration none \
  --device cuda \
  --conditions clean gaussian:0.1 impulse:0.05:5.0 random_missing:0.3 \
               block_missing:0.2 level_shift:2.0 gradual_drift:2.0 \
               sensor_freeze:0.30:0.25 heavy_tail:0.20:20.0 \
  --output ./results/example_robustness.csv
```

## 9. Efficiency

Benchmark selected checkpoints on the same device and batch size:

```bash
python3 benchmark_efficiency.py \
  --run-dir ./runs/ETTh2/kamo_v5_shared/example_s2026_h96 \
  --data-root ./datasets \
  --device cuda \
  --batch-size 16 \
  --warmup 20 \
  --repeats 200 \
  --output ./results/efficiency.csv
```

Report trainable parameters, median batch latency, throughput, and peak allocated GPU memory. Hardware and software versions should accompany any comparison.

## 10. Artifact audit

For a complete revision-style run, use `formal_revision_audit.py` with every run root and result root that is intentionally in scope. The audit verifies expected configurations and evidence files; it does not retrain models.

