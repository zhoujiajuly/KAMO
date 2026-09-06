from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


DATASETS: Dict[str, Dict[str, object]] = {
    "ETTh1": {"kind": "csv", "path": "ETTh1.csv"},
    "ETTh2": {"kind": "csv", "path": "ETTh2.csv"},
    "ETTm1": {"kind": "csv", "path": "ETTm1.csv"},
    "ETTm2": {"kind": "csv", "path": "ETTm2.csv"},
    "weather": {"kind": "csv", "path": "weather.csv"},
    "traffic": {"kind": "csv", "path": "traffic.csv"},
    "illness": {"kind": "csv", "path": "national_illness.csv"},
    "electricity": {
        "kind": "text",
        "paths": [
            "electricity/electricity.txt/electricity.txt",
            "electricity/electricity.txt.gz",
        ],
    },
    "exchange": {
        "kind": "text",
        "paths": [
            "exchange_rate/exchange_rate.txt",
            "exchange_rate/exchange_rate.txt.gz",
        ],
    },
    "solar": {
        "kind": "text_search",
        "directory": "solar-energy",
    },
    "Los_loop": {
        "kind": "csv_no_header",
        "path": "Los_loop/los_speed.csv",
        "adjacency": "Los_loop/los_adj.csv",
    },
    "Chengdu": {
        "kind": "npy",
        "path": "Chengdu/dataset.npy",
        "channel": 0,
        "adjacency": "Chengdu/matrix.npy",
    },
    "PEMS03": {
        "kind": "npz_raw",
        "path": "PEMS03/PEMS03.npz",
        "channel": 0,
        "edge_file": "PEMS03/PEMS03.csv",
    },
    "PEMS04": {
        "kind": "npz_raw",
        "path": "pems04/PEMS04.npz",
        "channel": 0,
        "edge_file": "pems04/PEMS04.csv",
    },
    "PEMS07": {
        "kind": "npz_raw",
        "path": "PEMS07/PEMS07.npz",
        "channel": 0,
        "edge_file": "PEMS07/PEMS07.csv",
    },
    "PEMS08": {
        "kind": "npz_raw",
        "path": "PEMS08/pems08.npz",
        "channel": 0,
        "adjacency": "PEMS08/pems08_adj.csv",
    },
    "METR-LA": {"kind": "prewindowed", "directory": "METR-LA"},
    "PEMS-BAY": {"kind": "prewindowed", "directory": "PEMS-BAY"},
    "physics": {"kind": "csv", "path": "physics/coupled_oscillators.csv"},
    "physionet2012": {
        "kind": "physionet2012",
        "directory": "physionet2012",
    },
}


# Dynamic variables used by the PhysioNet/CinC 2012 challenge.  Static patient
# descriptors (RecordID, Age, Gender, Height and ICUType) are deliberately
# excluded: this experiment tests forecasting with naturally missing variables,
# not outcome prediction from demographic covariates.
PHYSIONET_VARIABLES: Tuple[str, ...] = (
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
    "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP",
    "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
    "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
    "Weight",
)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def inverse_std(self, x: np.ndarray) -> np.ndarray:
        return x * self.std


class LongSeriesDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        starts: Sequence[int],
        seq_len: int,
        pred_len: int,
        scaler: Standardizer,
    ) -> None:
        self.values = values
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = np.asarray(self.values[start:mid], dtype=np.float32)
        y = np.asarray(self.values[mid:end], dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32, copy=False)
        y = self.scaler.transform(y).astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


class MaskedEpisodeDataset(Dataset):
    """Fixed-length patient episodes with explicit input and target masks."""

    def __init__(
        self,
        values: np.ndarray,
        masks: np.ndarray,
        seq_len: int,
        pred_len: int,
        scaler: Standardizer,
        stride: int = 12,
    ) -> None:
        if values.shape != masks.shape or values.ndim != 3:
            raise ValueError("Episode values and masks must share [episode,time,variable] shape")
        self.values = values
        self.masks = masks.astype(bool, copy=False)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = scaler
        last = values.shape[1] - seq_len - pred_len
        if last < 0:
            raise ValueError(
                f"Episode length {values.shape[1]} is too short for "
                f"{seq_len}->{pred_len}"
            )
        starts = list(range(0, last + 1, max(1, stride)))
        if starts[-1] != last:
            starts.append(last)
        self.indices = np.asarray(
            [(episode, start) for episode in range(values.shape[0]) for start in starts],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        episode, start = (int(value) for value in self.indices[index])
        mid = start + self.seq_len
        end = mid + self.pred_len
        x_mask = self.masks[episode, start:mid]
        y_mask = self.masks[episode, mid:end]
        x = self.values[episode, start:mid]
        y = self.values[episode, mid:end]
        x = np.where(x_mask, self.scaler.transform(x), 0.0).astype(np.float32)
        y = np.where(y_mask, self.scaler.transform(y), 0.0).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(x_mask),
            torch.from_numpy(y_mask),
        )


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    scaler: Standardizer
    num_vars: int
    seq_len: int
    pred_len: int
    adjacency: Optional[np.ndarray]
    dataset_name: str


def _read_text(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return np.loadtxt(handle, delimiter=",", dtype=np.float32)


def _first_existing(root: Path, paths: Sequence[str]) -> Path:
    for relative in paths:
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of these files exists: {paths}")


def _load_values(root: Path, spec: Dict[str, object]) -> np.ndarray:
    kind = str(spec["kind"])
    if kind == "csv":
        frame = pd.read_csv(root / str(spec["path"]))
        numeric = frame.select_dtypes(include=[np.number])
        return numeric.to_numpy(dtype=np.float32)
    if kind == "csv_no_header":
        return pd.read_csv(root / str(spec["path"]), header=None).to_numpy(dtype=np.float32)
    if kind == "text":
        return _read_text(_first_existing(root, spec["paths"]))
    if kind == "text_search":
        directory = root / str(spec["directory"])
        candidates = [
            path
            for path in sorted(directory.rglob("*.txt"))
            + sorted(directory.rglob("*.txt.gz"))
            if path.is_file()
        ]
        if not candidates:
            raise FileNotFoundError(f"No text data found below {directory}")
        return _read_text(candidates[0])
    if kind == "npy":
        arr = np.load(root / str(spec["path"]), mmap_mode="r")
        if arr.ndim == 3:
            arr = arr[..., int(spec.get("channel", 0))]
        return arr
    if kind == "npz_raw":
        with np.load(root / str(spec["path"])) as archive:
            arr = archive["data"]
        if arr.ndim == 3:
            arr = arr[..., int(spec.get("channel", 0))]
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported dataset kind: {kind}")


def _load_adjacency(root: Path, spec: Dict[str, object], n: int) -> Optional[np.ndarray]:
    if "adjacency" in spec:
        path = root / str(spec["adjacency"])
        if path.suffix == ".npy":
            adj = np.load(path)
        else:
            adj = pd.read_csv(path, header=None).to_numpy()
    elif "edge_file" in spec:
        edge_path = root / str(spec["edge_file"])
        edge_frame = pd.read_csv(edge_path, header=None)
        edges = edge_frame.to_numpy()
        try:
            int(edges[0, 0])
        except (TypeError, ValueError):
            edges = edges[1:]
        adj = np.zeros((n, n), dtype=np.float32)
        for row in edges:
            i, j = int(row[0]), int(row[1])
            if 0 <= i < n and 0 <= j < n:
                weight = 1.0 if len(row) < 3 else 1.0 / max(float(row[2]), 1e-6)
                adj[i, j] = weight
                adj[j, i] = weight
    else:
        return None
    adj = np.asarray(adj, dtype=np.float32)
    if adj.shape != (n, n):
        return None
    adj = np.maximum(adj, adj.T)
    adj += np.eye(n, dtype=np.float32)
    degree = np.maximum(adj.sum(axis=1, keepdims=True), 1e-6)
    return adj / degree


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def _build_long_series(
    name: str,
    root: Path,
    spec: Dict[str, object],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    workers: int,
) -> DataBundle:
    values = _load_values(root, spec)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all():
        frame = pd.DataFrame(np.asarray(values, dtype=np.float32))
        values = frame.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)
    total = len(values)
    train_end = int(total * 0.7)
    val_end = int(total * 0.8)
    if train_end <= seq_len + pred_len:
        raise ValueError(f"{name} is too short for seq_len={seq_len}, pred_len={pred_len}")
    train_values = np.asarray(values[:train_end], dtype=np.float64)
    mean = train_values.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_values.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-5)
    scaler = Standardizer(mean, std)

    def starts(target_start: int, target_end: int) -> np.ndarray:
        first = max(0, target_start - seq_len)
        last = target_end - seq_len - pred_len
        return np.arange(first, last + 1, dtype=np.int64)

    train_ds = LongSeriesDataset(values, starts(seq_len, train_end), seq_len, pred_len, scaler)
    val_ds = LongSeriesDataset(values, starts(train_end, val_end), seq_len, pred_len, scaler)
    test_ds = LongSeriesDataset(values, starts(val_end, total), seq_len, pred_len, scaler)
    adjacency = _load_adjacency(root, spec, values.shape[1])
    return DataBundle(
        _loader(train_ds, batch_size, True, workers),
        _loader(val_ds, batch_size, False, workers),
        _loader(test_ds, batch_size, False, workers),
        scaler,
        values.shape[1],
        seq_len,
        pred_len,
        adjacency,
        name,
    )


def _reconstruct_split(path: Path) -> np.ndarray:
    """Collapse overlapping DCRNN windows to one compact chronological series."""
    with np.load(path) as archive:
        first = np.asarray(archive["x"][0, :, :, 0], dtype=np.float32)
        tail = np.asarray(archive["y"][:, -1, :, 0], dtype=np.float32)
    return np.concatenate([first, tail], axis=0)


def _build_prewindowed(
    name: str,
    root: Path,
    spec: Dict[str, object],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    workers: int,
    cache_dir: Optional[Path],
) -> DataBundle:
    directory = root / str(spec["directory"])
    if seq_len != 12 or pred_len != 12:
        raise ValueError(f"{name} files use the standard 12->12 protocol; set both lengths to 12")
    cache_dir = cache_dir or (directory / "kamo_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_series = []
    for split in ("train", "val", "test"):
        cache = cache_dir / f"{name}_{split}_series.npy"
        if not cache.exists():
            np.save(cache, _reconstruct_split(directory / f"{split}.npz"))
        split_series.append(np.load(cache, mmap_mode="r"))
    train, val, test = split_series
    mean = np.asarray(train, dtype=np.float64).mean(axis=0, keepdims=True).astype(np.float32)
    std = np.asarray(train, dtype=np.float64).std(axis=0, keepdims=True).astype(np.float32)
    scaler = Standardizer(mean, np.maximum(std, 1e-5))

    def make(values: np.ndarray) -> LongSeriesDataset:
        last = len(values) - seq_len - pred_len
        return LongSeriesDataset(values, np.arange(last + 1), seq_len, pred_len, scaler)

    return DataBundle(
        _loader(make(train), batch_size, True, workers),
        _loader(make(val), batch_size, False, workers),
        _loader(make(test), batch_size, False, workers),
        scaler,
        train.shape[1],
        seq_len,
        pred_len,
        None,
        name,
    )


def _physionet_cache(
    directory: Path, split: str, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    cache = cache_dir / f"physionet2012_{split}_hourly.npz"
    if cache.exists():
        with np.load(cache) as archive:
            values = np.asarray(archive["values"], dtype=np.float32)
            masks = np.asarray(archive["masks"], dtype=bool)
        if values.shape == masks.shape and values.shape[1:] == (
            48, len(PHYSIONET_VARIABLES)
        ):
            return values, masks
        raise ValueError(f"Invalid PhysioNet cache shape in {cache}: {values.shape}")

    source = directory / f"set-{split}"
    files = sorted(source.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No PhysioNet patient records found below {source}")
    variable_index = {name: index for index, name in enumerate(PHYSIONET_VARIABLES)}
    values = np.full(
        (len(files), 48, len(PHYSIONET_VARIABLES)), np.nan, dtype=np.float32
    )
    for episode, path in enumerate(files):
        frame = pd.read_csv(path, usecols=["Time", "Parameter", "Value"])
        frame = frame[frame["Parameter"].isin(variable_index)]
        if frame.empty:
            continue
        frame["hour"] = frame["Time"].str.split(":").str[0].astype(int).clip(0, 47)
        grouped = frame.groupby(["hour", "Parameter"], sort=False)["Value"].mean()
        for (hour, parameter), value in grouped.items():
            if np.isfinite(value) and float(value) != -1.0:
                values[episode, int(hour), variable_index[str(parameter)]] = float(value)
    masks = np.isfinite(values)
    # Missing slots remain zero in storage and are never interpreted without masks.
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, values=values, masks=masks)
    return values, masks


def _masked_standardizer(values: np.ndarray, masks: np.ndarray) -> Standardizer:
    weights = masks.astype(np.float64)
    count = weights.sum(axis=(0, 1), keepdims=False).clip(min=1.0)
    total = (values.astype(np.float64) * weights).sum(axis=(0, 1))
    mean = total / count
    variance = (
        ((values.astype(np.float64) - mean[None, None, :]) ** 2) * weights
    ).sum(axis=(0, 1)) / count
    return Standardizer(
        mean[None, :].astype(np.float32),
        np.maximum(np.sqrt(variance), 1e-5)[None, :].astype(np.float32),
    )


def _build_physionet2012(
    name: str,
    root: Path,
    spec: Dict[str, object],
    seq_len: int,
    pred_len: int,
    batch_size: int,
    workers: int,
    cache_dir: Optional[Path],
) -> DataBundle:
    directory = root / str(spec["directory"])
    cache = cache_dir or (directory / "kamo_cache")
    train_values, train_masks = _physionet_cache(directory, "a", cache)
    val_values, val_masks = _physionet_cache(directory, "b", cache)
    test_values, test_masks = _physionet_cache(directory, "c", cache)
    scaler = _masked_standardizer(train_values, train_masks)

    def make(values: np.ndarray, masks: np.ndarray) -> MaskedEpisodeDataset:
        return MaskedEpisodeDataset(
            values, masks, seq_len, pred_len, scaler, stride=12
        )

    return DataBundle(
        _loader(make(train_values, train_masks), batch_size, True, workers),
        _loader(make(val_values, val_masks), batch_size, False, workers),
        _loader(make(test_values, test_masks), batch_size, False, workers),
        scaler,
        len(PHYSIONET_VARIABLES),
        seq_len,
        pred_len,
        None,
        name,
    )


def build_dataloaders(
    name: str,
    data_root: str,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    workers: int = 0,
    cache_dir: Optional[str] = None,
) -> DataBundle:
    aliases = {key.lower(): key for key in DATASETS}
    canonical = aliases.get(name.lower())
    if canonical is None:
        raise KeyError(f"Unknown dataset {name!r}. Choices: {', '.join(DATASETS)}")
    root = Path(data_root)
    spec = DATASETS[canonical]
    if spec["kind"] == "physionet2012":
        return _build_physionet2012(
            canonical,
            root,
            spec,
            seq_len,
            pred_len,
            batch_size,
            workers,
            Path(cache_dir) if cache_dir else None,
        )
    if spec["kind"] == "prewindowed":
        return _build_prewindowed(
            canonical,
            root,
            spec,
            seq_len,
            pred_len,
            batch_size,
            workers,
            Path(cache_dir) if cache_dir else None,
        )
    return _build_long_series(
        canonical, root, spec, seq_len, pred_len, batch_size, workers
    )
