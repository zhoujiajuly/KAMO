# KAMO

Official research code for **KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts**.

KAMO estimates a causal observation-reliability signal and reuses it for robust normalization, multiscale branch routing, residual control, and predictive-scale adjustment. The repository also contains controlled probabilistic implementations of the comparison backbones and the scripts used for corruption, ablation, sensitivity, calibration, transfer, long-horizon, high-dimensional, real-missingness, and efficiency experiments.

## Repository status

This folder is a GitHub release candidate prepared from the final revision code. It contains source code and reproducibility scripts only. Datasets, checkpoints, tuning runs, logs, and generated result files are intentionally excluded.

The included PatchTST, iTransformer, TimeMixer, and VPNet-style models are controlled probabilistic adaptations implemented in this repository for a common evaluation protocol. They must not be described as unmodified code copied from the respective authors' official repositories.

## Contents

```text
kamo/
  model.py                KAMO and forecasting model definitions
  robustifier.py          causal robust statistics and reliability modules
  baselines.py            controlled probabilistic comparison backbones
  data.py                 dataset loaders and chronological splits
  engine.py               training, corruption, and evaluation engine
  metrics.py              probabilistic and point metrics
  plotting.py             plotting utilities
train.py                  train/validation/test entry point
evaluate_robustness.py    clean and corrupted-context evaluation
run_revision_suite.py     resumable paper experiment suite
tune_validation.py        validation-only equal-budget tuning
benchmark_efficiency.py   latency, throughput, and memory benchmark
calibrate_*.py            checkpoint-bound uncertainty calibration
run_transfer_*.py         reliability-transfer validation protocols
test_*.py                 mechanism and protocol tests
docs/DATASETS.md          download sources and required directory layout
docs/REPRODUCIBILITY.md   experiment stages and reproduction commands
```

## Installation

Python 3.8 or newer is recommended. Install a PyTorch build appropriate for your CPU/CUDA environment first, following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), and then install the remaining dependencies:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd KAMO

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Confirm that the implementation-level invariants pass:

```bash
python3 test_revision_invariants.py
python3 test_c5_tuning_audit.py
```

These tests use synthetic tensors and do not require downloading a dataset.

## Datasets

Datasets are not redistributed in this repository. Download them from their original or benchmark-maintainer sources and place them under `./datasets`. Exact sources, expected filenames, PhysioNet layout, and preprocessing cautions are given in [docs/DATASETS.md](docs/DATASETS.md).

The shortest setup for ETTh2 is:

```text
datasets/
  ETTh2.csv
```

## Quick start

Train the final shared-reliability KAMO configuration on ETTh2:

```bash
python3 train.py \
  --dataset ETTh2 \
  --data-root ./datasets \
  --output-root ./runs \
  --run-name example_s2026_h96 \
  --model kamo_v5_shared \
  --seq-len 96 \
  --pred-len 96 \
  --seed 2026 \
  --device auto
```

The run directory contains the frozen configuration, scaler, validation-selected `best.pt`, training history, and test metrics.

Evaluate the checkpoint under clean and corrupted contexts:

```bash
python3 evaluate_robustness.py \
  --run-dir ./runs/ETTh2/kamo_v5_shared/example_s2026_h96 \
  --data-root ./datasets \
  --device auto \
  --calibration none \
  --conditions \
    clean gaussian:0.1 gaussian:0.3 impulse:0.05:5.0 \
    random_missing:0.1 random_missing:0.3 block_missing:0.2 \
    level_shift:2.0 gradual_drift:2.0 sensor_freeze:0.30:0.25 \
    heavy_tail:0.20:20.0 \
  --output ./results/ETTh2_example.csv
```

Use `--split validation` for architecture selection and `--split test` only for final reporting.

## Reproducing the experiment suite

Inspect the frozen job plan without training:

```bash
python3 run_revision_suite.py \
  --stage gate \
  --data-root ./datasets \
  --runs-root ./runs \
  --results-root ./results \
  --dry-run
```

Run the three-seed shared-reliability gate:

```bash
python3 run_revision_suite.py \
  --stage gate \
  --data-root ./datasets \
  --runs-root ./runs \
  --results-root ./results \
  --seeds 2026,2027,2028 \
  --device cuda \
  --workers 4
```

Do not launch `--stage all` before checking the gate output. The full order, stage meanings, multi-GPU sharding examples, and audit commands are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Checkpoint policy

Checkpoints are not committed to Git because they are large and configuration-specific. Every run records the seed and complete configuration needed to retrain it. If pretrained weights are released later, publish only the selected final checkpoints as a versioned GitHub Release or a DOI-backed archive such as Zenodo, together with checksums and matching configuration files.

## Reproducibility safeguards

- chronological train/validation/test splits;
- train-split-only standardization;
- validation CRPS for checkpoint and hyperparameter selection;
- seeds 2026, 2027, and 2028 for reported multi-seed experiments;
- corruption generation with explicit, seeded condition specifications;
- separate validation and test evaluation modes;
- parameter, latency, throughput, and peak-memory reporting;
- fail-closed artifact audits for the revision experiment suite.

## Citation

The manuscript is currently under review. Until final bibliographic metadata are available, cite the repository and manuscript title:

```bibtex
@article{zhou2026kamo,
  title  = {KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts},
  author = {Zhou, Jia and Miao, Xiaoyu and Liu, Tao},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

No open-source license has been selected yet. Add the chosen license before making the repository public; without a license, GitHub users may view the code but do not automatically receive reuse rights.

