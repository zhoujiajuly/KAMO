# KAMO

Official research code for **KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts**.

This revision reproduces the R2 protocol used by the manuscript: a chronological 70%/5%/5%/20% train/tuning/calibration/test split for continuous datasets, 12 deterministic candidates per searched configuration, candidate selection by the mean normalized CRPS over five tuning conditions, and final seeds 2026, 2027, and 2028.

## Current R2 entry points

```text
run_r2.py             authoritative single-process experiment runner
launch_r2.py          disjoint multi-GPU launcher
kamo_r2/data.py       70/5/5/20 split and patient-disjoint PhysioNet split
kamo_r2/protocol.py   frozen recipes, 12-candidate spaces, five tuning conditions
kamo_r2/training.py   tuning-only checkpoint selection and final training
kamo_r2/evaluation.py fixed corruption, calibration, and probability metrics
kamo_r2/analysis.py   summaries and paired bootstrap records
tests/test_r2_*.py    protocol and implementation tests
BASELINE_SOURCES.md   model provenance and adaptation scope
```

The older top-level revision utilities are retained for provenance. They are not the entry points for the R2 tables. `tune_validation.py` is now only a compatibility wrapper around `run_r2.py`; the earlier six-candidate workflow is retired.

## Installation

Install a PyTorch build appropriate for the local CUDA environment, then:

```bash
python3 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Datasets are not redistributed. Their sources and required layout are described in [`docs/DATASETS.md`](docs/DATASETS.md).

## Verify the frozen plan without training

```bash
python3 run_r2.py --recipe core --stage plan --data-root ./datasets --out ./r2_runs
python3 run_r2.py --recipe long --stage plan --data-root ./datasets --out ./r2_runs
```

The final long-horizon recipe contains the 38 configurations reported in Table 6: eight methods at H=336 and H=720 for ETTh2 and Weather, plus KAMO, iTransformer, and VPNet at H=192.

## Run on multiple GPUs

`launch_r2.py` assigns disjoint model/dataset cases to the listed physical GPUs and fails closed before advancing to another recipe:

```bash
python3 launch_r2.py \
  --data-root ./datasets \
  --out ./r2_runs \
  --gpus 0,1,2 \
  --recipe core,controls \
  --stage train
```

Evaluation and reporting are separate stages:

```bash
python3 launch_r2.py --data-root ./datasets --out ./r2_runs --gpus 0,1,2 --recipe core --stage evaluate --eval-set all
python3 run_r2.py --data-root ./datasets --out ./r2_runs --recipe core --stage report
```

## R2 safeguards

- Continuous data: chronological 70%/5%/5%/20% train/tuning/calibration/test split.
- PhysioNet: patient-disjoint partitions.
- Standardization fitted on training data only.
- Twelve deterministic candidates per searched configuration.
- Candidate score: mean normalized CRPS over clean, Gaussian noise, impulse noise, random missingness, and block missingness on the tuning split.
- Checkpoints selected by clean tuning normalized CRPS.
- Scalar Gaussian temperature fitted only on the disjoint clean calibration partition and frozen for every test condition.
- Final seeds: 2026, 2027, and 2028.
- Test conditions never select a candidate, checkpoint, or calibration temperature.

The controlled PatchTST-, iTransformer-, TimeMixer-, and VPNet-style probabilistic models are local adaptations for the common R2 protocol; they are not claimed to be unmodified official-repository reproductions. See `BASELINE_SOURCES.md`.

## Tests

```bash
python3 -m unittest discover -s tests -p "test_r2_*.py"
```

## Citation

```bibtex
@article{zhou2026kamo,
  title  = {KAMO: Reliability-Routed Multiscale Probabilistic Forecasting under Corrupted Contexts},
  author = {Zhou, Jia and Miao, Xiaoyu and Liu, Tao},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

The repository includes third-party license notices under `kamo_r2/vendor/`. No additional project-wide open-source license is granted unless a root license file states otherwise.
