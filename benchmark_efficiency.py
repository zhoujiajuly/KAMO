from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from kamo.data import build_dataloaders
from kamo.engine import set_seed, unpack_batch
from train import build_model, choose_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_one(run_dir, device, batch_size, warmup, repeats, data_root=None):
    with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    args = SimpleNamespace(**config)
    if data_root is not None:
        args.data_root = data_root
    set_seed(int(args.seed))
    bundle = build_dataloaders(
        args.dataset,
        args.data_root,
        args.seq_len,
        args.pred_len,
        batch_size,
        0,
        getattr(args, "cache_dir", None),
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = build_model(args, bundle).to(device).eval()
    try:
        checkpoint = torch.load(
            run_dir / "best.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    # Do not include a second GPU-resident checkpoint copy in inference memory.
    del checkpoint
    x, _, x_mask, _ = unpack_batch(next(iter(bundle.test)))
    x, mask = x.to(device), x_mask.to(device)
    actual_batch = int(x.shape[0])

    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask=mask)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for _ in range(repeats):
            synchronize(device)
            started = time.perf_counter()
            model(x, mask=mask)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        if device.type == "cuda"
        else float("nan")
    )
    median_ms = statistics.median(latencies)
    sorted_ms = sorted(latencies)
    p95_index = min(len(sorted_ms) - 1, math.ceil(0.95 * len(sorted_ms)) - 1)
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_size_mb_fp32": parameters * 4.0 / (1024.0 ** 2),
        "batch_size": actual_batch,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_median": median_ms,
        "latency_ms_p95": sorted_ms[p95_index],
        "samples_per_second": actual_batch * 1000.0 / median_ms,
        "peak_memory_mb": peak_mb,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float32",
        "timing_scope": "model_forward_only_data_loading_excluded",
        "warmup": warmup,
        "repeats": repeats,
        "run_dir": str(run_dir),
    }
    print(
        f"{args.dataset:10s} {args.model:18s} params={parameters:,} "
        f"median={median_ms:.3f} ms "
        f"throughput={result['samples_per_second']:.2f}/s peak={peak_mb:.1f} MB"
    )
    del model, x, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark checkpoint inference efficiency"
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", default="runs/efficiency.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default=None, help="Override the path stored in checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error(
            "batch-size and repeats must be positive; warmup must be non-negative"
        )

    device = choose_device(args.device)
    rows = [
        benchmark_one(
            Path(raw), device, args.batch_size, args.warmup, args.repeats, args.data_root
        )
        for raw in args.run_dir
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved efficiency rows={len(rows)}: {output.resolve()}")


if __name__ == "__main__":
    main()
