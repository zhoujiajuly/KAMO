# KAMO

Code for **KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts**.

The repository has one supported experiment workflow. `run.py` trains, evaluates, and summarizes individual experiment recipes; `launch.py` distributes cases across GPUs. The `kamo_experiments/` package implements the current protocol. The `kamo/` package supplies model components, data parsers, and shared utilities used by that workflow. Older standalone training scripts were removed from the current tree; their source remains available in Git history.

## Install

Use Python 3.8 or newer. Install a PyTorch build appropriate for your machine, then run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Datasets are downloaded separately. Follow [dataset preparation](docs/DATASETS.md) and place them under `datasets/`, or pass another directory with `--data-root`.

## First run

Inspect the predefined cases without training or downloading data:

```bash
python3 run.py --recipe core --stage plan --out ./runs
```

To train and evaluate one example, first place `ETTh2.csv` in `datasets/`, then run:

```bash
python3 run.py --recipe core --datasets ETTh2 --models kamo_s --stage train --data-root ./datasets --out ./runs
python3 run.py --recipe core --datasets ETTh2 --models kamo_s --stage evaluate --eval-set core --data-root ./datasets --out ./runs
python3 run.py --recipe core --stage report --out ./runs
```

Training includes the declared hyperparameter search and three final seeds, so it can take hours even for one case. `--stage plan` is the fast way to inspect the workload. A completed case is reused only when its saved configuration, data, and source fingerprint agree. Use a new `--out` directory if the protocol or source changes.

To run across three GPUs:

```bash
python3 launch.py --data-root ./datasets --out ./runs --gpus 0,1,2 --recipe core --stage train
python3 launch.py --data-root ./datasets --out ./runs --gpus 0,1,2 --recipe core --stage evaluate --eval-set all
python3 run.py --recipe core --stage report --out ./runs
```

`launch.py` writes worker logs under `runs/logs/`. It executes comma-separated recipes in order; for example, `--recipe core,controls` trains the main cases before controls that inherit their selected settings.

## Experiment recipes

| Recipe | Purpose |
|---|---|
| `core` | Main clean and corrupted-context comparisons on ETTh1, ETTh2, Weather, and Exchange |
| `controls` | Component controls; run `core` first in the same output directory |
| `highdim` | Electricity comparison |
| `physionet` | Naturally missing clinical observations |
| `long` | Long-horizon comparisons on ETTh2 and Weather |
| `dimension` | Controlled variable-count scaling |
| `pilot` | Small protocol check before a larger run |

The results reported in the manuscript use 93 main model–dataset–window configurations and 28 controls. The `long` recipe contains the 38 configurations used in the long-horizon table. The script's `plan` stage prints the exact cases and search budget. See [reproduction details](docs/REPRODUCIBILITY.md) for data partitions, tuning, calibration, resumption, and profiling.

## Evaluation protocol

Continuous datasets use chronological 70% training, 5% tuning, 5% calibration, and 20% test target partitions. PhysioNet uses patient-disjoint partitions. Standardization is fitted on training observations only. Each searched configuration receives 12 deterministic candidates; selection uses mean normalized CRPS over five tuning conditions. Final results use seeds 2026, 2027, and 2028. Calibration fits one Gaussian scale temperature on the separate clean calibration partition; test data do not select settings.

The probabilistic comparison models share this evaluation protocol. Some are local adaptations of published backbones, not unmodified runs of their authors' original training pipelines. Their provenance and changes are documented in [model sources](docs/MODEL_SOURCES.md).

Datasets, checkpoints, logs, and generated results are not stored in Git. Third-party license notices are included beside vendored models. No project-wide open-source license is granted unless a root license file states otherwise.

## Citation

```bibtex
@article{zhou2026kamo,
  title  = {KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts},
  author = {Zhou, Jia and Miao, Xiaoyu and Liu, Tao},
  year   = {2026},
  note   = {Manuscript under review}
}
```
