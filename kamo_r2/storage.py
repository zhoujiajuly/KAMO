from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(str(temp), str(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(list((root / "kamo").glob("*.py")) + list((root / "kamo_r2").rglob("*.py")) + [root / "legacy_train.py", root / "run_r2.py"])
    hashed = hashlib.sha256()
    for path in paths:
        if path.is_file():
            hashed.update(str(path.relative_to(root)).replace("\\", "/").encode())
            hashed.update(path.read_bytes())
    return hashed.hexdigest()


def save_checkpoint(path, value):
    path = Path(path); temp = path.with_name(path.name + ".tmp")
    torch.save(value, str(temp)); os.replace(str(temp), str(path))


def load_checkpoint(path, device="cpu"):
    # Only load checkpoints produced locally by this package, never untrusted pickles.
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:  # torch 1.11 support
        return torch.load(str(path), map_location=device)
