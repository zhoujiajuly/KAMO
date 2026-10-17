"""Train-only scaling, explicit masks, and disjoint tuning/calibration targets."""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from kamo.data import (DATASETS, Standardizer, _load_values, _load_adjacency,
                       _build_physionet2012)


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


class SeriesWindows(Dataset):
    def __init__(self, values, masks, starts, seq_len, pred_len, scaler):
        self.values, self.masks = values, masks
        self.window_starts = np.asarray(starts, dtype=np.int64)
        self.window_ids = self.window_starts.copy()
        self.group_ids = np.full(len(starts), -1, dtype=np.int64)
        self.seq_len, self.pred_len, self.scaler = seq_len, pred_len, scaler

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, i):
        s = int(self.window_starts[i]); m = s + self.seq_len; e = m + self.pred_len
        xm, ym = self.masks[s:m], self.masks[m:e]
        x = np.where(xm, self.scaler.transform(self.values[s:m]), 0).astype(np.float32)
        y = np.where(ym, self.scaler.transform(self.values[m:e]), 0).astype(np.float32)
        return dict(x=torch.from_numpy(x), y=torch.from_numpy(y),
                    x_mask=torch.from_numpy(xm), y_mask=torch.from_numpy(ym),
                    window_start=s, window_id=int(self.window_ids[i]), group_id=-1)


class EpisodeWindows(Dataset):
    def __init__(self, base, indices=None):
        self.base = base
        self.indices = np.arange(len(base)) if indices is None else np.asarray(indices)
        pairs = base.indices[self.indices]
        self.group_ids = pairs[:, 0].astype(np.int64)
        self.window_starts = pairs[:, 1].astype(np.int64)
        self.window_ids = self.indices.astype(np.int64)
        self.seq_len, self.pred_len = base.seq_len, base.pred_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, xm, ym = self.base[int(self.indices[i])]
        return dict(x=x, y=y, x_mask=xm, y_mask=ym,
                    window_start=int(self.window_starts[i]),
                    window_id=int(self.window_ids[i]), group_id=int(self.group_ids[i]))


def make_loader(dataset, batch_size, shuffle=False, workers=0):
    if not len(dataset):
        raise ValueError("Empty split: shorten horizon or provide the complete dataset")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False)


def _resolve_values(root, name):
    spec = dict(DATASETS[name])
    candidates = [root, root / "datasets", root / "data", root / "ETT-small",
                  root / "ETT", root / "weather", root / "electricity", root / "exchange_rate"]
    errors = []
    for candidate in candidates:
        try:
            return _load_values(candidate, spec), candidate, spec
        except (FileNotFoundError, NotADirectoryError) as exc:
            errors.append(str(exc))
        # Common Time-Series-Library CSV distribution, in addition to R1 text layout.
        if name in {"electricity", "exchange"}:
            for filename in (["electricity.csv"] if name == "electricity" else ["exchange_rate.csv", "exchange.csv"]):
                alt = dict(kind="csv", path=filename)
                if (candidate / filename).is_file():
                    return _load_values(candidate, alt), candidate, alt
    raise FileNotFoundError("Cannot locate {} under {}. Last error: {}".format(name, root, errors[-1]))


def build_bundle(name, data_root, seq_len=96, pred_len=96, batch_size=16,
                 workers=0, cache_dir=None, dimensions=None, feature_seed=73021):
    names = {key.lower(): key for key in DATASETS}
    name = names.get(name.lower(), name)
    if name not in DATASETS:
        raise ValueError("Unknown dataset: " + name)
    root = Path(data_root)
    if name == "physionet2012":
        if dimensions is not None:
            raise ValueError("Dimension subsets are for continuous datasets, not patients")
        old = _build_physionet2012(name, root, DATASETS[name], seq_len, pred_len,
                                  batch_size, workers, Path(cache_dir) if cache_dir else root / "kamo_r2_cache")
        val = old.val.dataset
        patients = np.unique(val.indices[:, 0])
        cut = max(1, int(len(patients) * .7))
        tune_ids = np.flatnonzero(np.isin(val.indices[:, 0], patients[:cut]))
        cal_ids = np.flatnonzero(np.isin(val.indices[:, 0], patients[cut:]))
        sets = [EpisodeWindows(old.train.dataset), EpisodeWindows(val, tune_ids),
                EpisodeWindows(val, cal_ids), EpisodeWindows(old.test.dataset)]
        fingerprint = array_hash(old.train.dataset.values, old.train.dataset.masks,
                                 val.values, val.masks, old.test.dataset.values, old.test.dataset.masks)
        scaler, adjacency, n = old.scaler, old.adjacency, old.num_vars
        audit = dict(protocol="PhysioNet a=train; b=70% tuning patients / 30% calibration patients; c=test",
                     tuning_patients=patients[:cut].tolist(), calibration_patients=patients[cut:].tolist())
    else:
        if DATASETS[name]["kind"] == "prewindowed":
            raise ValueError("R2 recipe excludes prewindowed traffic; do not reconstruct it as a continuous stream")
        values, actual_root, spec = _resolve_values(root, name)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("Expected [time, variable] dataset")
        original_n = values.shape[1]
        columns = np.arange(original_n)
        if dimensions is not None:
            if not 1 <= dimensions <= original_n:
                raise ValueError("Requested D={} but dataset has D={}".format(dimensions, original_n))
            columns = np.sort(np.random.RandomState(feature_seed).permutation(original_n)[:dimensions])
            values = values[:, columns]
        masks = np.isfinite(values)
        total = len(values); train_end = int(total * .7); tune_end = int(total * .75); val_end = int(total * .8)
        observed_train = masks[:train_end]
        counts = observed_train.sum(0)
        if (counts == 0).any():
            raise ValueError("A feature has no finite training observations")
        train = np.where(observed_train, values[:train_end], 0).astype(np.float64)
        mean = train.sum(0) / counts
        variance = np.where(observed_train, (train - mean) ** 2, 0).sum(0) / counts
        scaler = Standardizer(mean[None].astype(np.float32), np.maximum(np.sqrt(variance), 1e-5)[None].astype(np.float32))
        # No interpolation: missing inputs have an explicit mask; missing targets are excluded.
        values = np.where(masks, values, 0).astype(np.float32)
        def make(target_start, target_end):
            starts = np.arange(max(0, target_start - seq_len), target_end - seq_len - pred_len + 1)
            return SeriesWindows(values, masks, starts, seq_len, pred_len, scaler)
        sets = [make(seq_len, train_end), make(train_end, tune_end), make(tune_end, val_end), make(val_end, total)]
        adjacency = _load_adjacency(actual_root, spec, original_n)
        if adjacency is not None:
            adjacency = adjacency[np.ix_(columns, columns)]
        n = values.shape[1]
        fingerprint = array_hash(values, masks, columns)
        audit = dict(protocol="chronological 70/5/5/20% train/tuning/calibration/test (NOT standard ETT calendar split)",
                     total_timepoints=total, target_boundaries=[0, train_end, tune_end, val_end, total],
                     columns=columns.tolist(), original_dimensions=original_n,
                     missing_values=int((~masks).sum()), missing_policy="explicit masks; no interpolation; train-only observed scaling")
    audit.update(dataset=name, seq_len=seq_len, pred_len=pred_len, dimensions=n,
                 data_sha256=fingerprint, split_windows=dict(zip(["train", "tune", "calibration", "test"], map(len, sets))))
    loaders = [make_loader(ds, batch_size, i == 0, workers) for i, ds in enumerate(sets)]
    return SimpleNamespace(train=loaders[0], val=loaders[1], calibration=loaders[2], test=loaders[3],
                           scaler=scaler, adjacency=adjacency, num_vars=n, seq_len=seq_len,
                           pred_len=pred_len, dataset_name=name, audit=audit)
