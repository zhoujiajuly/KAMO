#!/usr/bin/env python
"""Launch disjoint model/dataset cases on one or more GPUs, without shell dependencies."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root")
    p.add_argument("--out", default="runs")
    p.add_argument("--gpus", default="0,1", help="Physical visible GPU IDs, or cpu for smoke tests")
    p.add_argument("--recipe", default="core", help="Comma list, executed in order; e.g. core,controls")
    p.add_argument("--stage", default="train", choices=["plan", "check", "train", "evaluate", "all", "profile"])
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args()
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(not g for g in gpu_ids):
        p.error("GPU IDs must be nonempty and unique")
    if "cpu" in gpu_ids and gpu_ids != ["cpu"]:
        p.error("cpu cannot be mixed with GPUs")
    if args.stage == "profile" and len(gpu_ids) != 1:
        p.error("Profile every method serially on ONE idle GPU; pass --gpus 0")
    if any(x.split("=")[0] in {"--device", "--shards", "--shard-index"} for x in extra):
        p.error("Device and sharding are controlled by launcher")
    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    data_root = str(Path(args.data_root).resolve()) if args.data_root else None
    log_root = out / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for recipe in args.recipe.split(","):
        processes, handles = [], []
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for index, gpu in enumerate(gpu_ids):
                cmd = [sys.executable, "-u", str(root / "run.py"), "--recipe", recipe,
                       "--stage", args.stage, "--out", str(out), "--shards", str(len(gpu_ids)),
                       "--shard-index", str(index), "--device", "cpu" if gpu == "cpu" else "cuda:0"]
                if data_root: cmd.extend(["--data-root", data_root])
                cmd.extend(extra)
                print("GPU {}: {}".format(gpu, subprocess.list2cmdline(cmd)), flush=True)
                if args.dry_run: continue
                environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                if gpu != "cpu": environment["CUDA_VISIBLE_DEVICES"] = gpu
                log_file = log_root / ("{}_{}_gpu{}_{}.log".format(recipe, args.stage, gpu, timestamp))
                handle = log_file.open("w", encoding="utf-8")
                handles.append(handle)
                processes.append(subprocess.Popen(cmd, cwd=str(root), env=environment, stdout=handle, stderr=subprocess.STDOUT))
                print("LOG " + str(log_file), flush=True)
            if args.dry_run: continue
            while any(process.poll() is None for process in processes):
                time.sleep(2)
            codes = [process.returncode for process in processes]
            if any(codes):
                raise RuntimeError("Phase {} failed (exit codes {}). Inspect logs; the next recipe was NOT launched.".format(recipe, codes))
            print("PHASE COMPLETE {} {}".format(recipe, args.stage), flush=True)
        except BaseException:
            for process in processes:
                if process.poll() is None: process.terminate()
            for process in processes:
                if process.poll() is None:
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill()
            raise
        finally:
            for handle in handles: handle.close()


if __name__ == "__main__":
    main()
