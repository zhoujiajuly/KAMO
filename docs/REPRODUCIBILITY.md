# R2 reproducibility protocol

The authoritative runner is `run_r2.py`. Every run writes its complete configuration, protocol fingerprint, data audit, candidate records, selected checkpoint, final-seed records, and test metrics below the requested output directory.

## Data partitions

For continuous datasets, `kamo_r2/data.py` uses chronological target partitions of 70% training, 5% tuning, 5% clean calibration, and 20% test. Overlapping windows may use preceding context, but every target belongs to exactly one partition. Standardizers are fitted only on training observations. PhysioNet uses patient-disjoint partitions.

## Search and final training

Each searched configuration evaluates 12 deterministic candidates. The candidate score is the mean normalized CRPS across five tuning conditions: clean, Gaussian noise (0.10), impulse corruption (rate 0.01 and magnitude 5), random missingness (0.10), and block missingness (0.10).

Candidate search and checkpoint selection use the tuning partition only. Final models are trained for seeds 2026, 2027, and 2028. The scalar standard-deviation temperature is fitted by Gaussian NLL on the disjoint clean calibration partition and is frozen for all test conditions.

## Frozen recipes

```bash
python3 run_r2.py --recipe core      --stage plan --data-root ./datasets --out ./r2_runs
python3 run_r2.py --recipe controls  --stage plan --data-root ./datasets --out ./r2_runs
python3 run_r2.py --recipe highdim   --stage plan --data-root ./datasets --out ./r2_runs
python3 run_r2.py --recipe physionet --stage plan --data-root ./datasets --out ./r2_runs
python3 run_r2.py --recipe long      --stage plan --data-root ./datasets --out ./r2_runs
```

Together, the non-control recipes define the 93 principal configurations in the final R2 manuscript. The controls recipe adds 28 inherited component controls, giving 121 active configurations. The long recipe has 38 main-table configurations. Six additional completed H=192 configurations are retained only in the electronic evidence archive and are explicitly marked as not used in Table 6.

## Resumption and parallel execution

Completed artifacts are reused only after their protocol fingerprint and configuration digest match. The launcher shards cases, not seeds within a case:

```bash
python3 launch_r2.py --data-root ./datasets --out ./r2_runs --gpus 0,1,2 --recipe long --stage train
python3 launch_r2.py --data-root ./datasets --out ./r2_runs --gpus 0,1,2 --recipe long --stage evaluate --eval-set all
python3 run_r2.py --data-root ./datasets --out ./r2_runs --recipe long --stage report
```

Profiling must be serialized on one idle physical GPU:

```bash
python3 launch_r2.py --data-root ./datasets --out ./r2_runs --gpus 0 --recipe core --stage profile
```

## Scope

The repository implements a common probabilistic evaluation protocol. Controlled baseline adaptations are identified in `BASELINE_SOURCES.md`; results should not be described as exact reproductions of unrelated official training recipes.
