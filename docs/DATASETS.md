# Dataset preparation

KAMO does not download or redistribute datasets automatically. This avoids repackaging third-party data and keeps provenance explicit. Pass the common parent directory with `--data-root`; the default is `./datasets`.

## Recommended benchmark source

The preprocessed long-term forecasting datasets can be obtained from the dataset links maintained by the official [THUML Time-Series-Library](https://github.com/thuml/Time-Series-Library#prepare-data), including its [Hugging Face dataset mirror](https://huggingface.co/datasets/thuml/Time-Series-Library). After downloading, copy or link the required files into the layout below.

The ETT files are also available from the original [ETDataset repository](https://github.com/zhouhaoyi/ETDataset).

## Datasets used in the KAMO experiments

| Dataset | Expected location under `datasets/` | Source |
|---|---|---|
| ETTh1 | `ETTh1.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) |
| ETTh2 | `ETTh2.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) |
| ETTm1 | `ETTm1.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) |
| ETTm2 | `ETTm2.csv` | [ETDataset](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small) |
| Weather | `weather.csv` | [Time-Series-Library data bundle](https://github.com/thuml/Time-Series-Library#prepare-data) |
| Traffic | `traffic.csv` | [Time-Series-Library data bundle](https://github.com/thuml/Time-Series-Library#prepare-data) |
| Exchange | `exchange_rate/exchange_rate.txt` or `.txt.gz` | [Time-Series-Library data bundle](https://github.com/thuml/Time-Series-Library#prepare-data) |
| Electricity | `electricity/electricity.txt/electricity.txt` or `electricity/electricity.txt.gz` | [Time-Series-Library data bundle](https://github.com/thuml/Time-Series-Library#prepare-data) |
| PhysioNet 2012 | `physionet2012/set-a`, `set-b`, and `set-c` | [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) |

The original Electricity Load Diagrams data are documented by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014). The KAMO loader expects the comma-delimited benchmark-preprocessed `electricity.txt` distributed with the forecasting benchmark bundle; the raw UCI `LD2011_2014.txt` must not simply be renamed because its delimiter and preprocessing differ.

## Example directory tree

```text
datasets/
  ETTh1.csv
  ETTh2.csv
  ETTm1.csv
  ETTm2.csv
  weather.csv
  traffic.csv
  exchange_rate/
    exchange_rate.txt
  electricity/
    electricity.txt.gz
  physionet2012/
    set-a/
      132539.txt
      ...
    set-b/
      ...
    set-c/
      ...
```

For PhysioNet, the current runner creates hourly cached arrays under the chosen output directory's `data_cache/`.

## Additional supported loaders

`kamo/data.py` contains additional parsers inherited from earlier research. They are not part of the predefined manuscript recipes. Inspect `kamo/data.py` and `kamo_experiments/data.py` before using a custom dataset; the predefined runner excludes prewindowed traffic data.

## Split and normalization behavior

- Continuous datasets use chronological 70% training, 5% tuning, 5% calibration, and 20% test target partitions. This is not the standard ETT calendar split.
- Standardization statistics are computed from observed training values only.
- PhysioNet uses set A for training, patient-disjoint subsets of set B for tuning and calibration, and set C for test; observation masks are preserved.
- Static PhysioNet descriptors are excluded; only the 37 dynamic variables listed in `kamo/data.py` are used.

