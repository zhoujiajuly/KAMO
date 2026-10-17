"""Full-loader probabilistic training with atomic, strictly resumable epochs."""
from __future__ import annotations
import contextlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import RandomSampler

from kamo.engine import augment_context, gaussian_nll, masked_mae, set_seed
from .factory import build_model
from .evaluation import evaluate_model
from .storage import digest, read_json, write_json, save_checkpoint, load_checkpoint


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Visible CUDA device count changed during resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_identity(record, fingerprint, config_digest):
    if record.get("fingerprint") != fingerprint or record.get("config_digest") != config_digest:
        raise ValueError("Run fingerprint/config mismatch; choose a new output directory")


def _augment_cpu(x, config, epoch, batch_number):
    # Independent augmentation RNG provides the same examples to every model
    # at the same seed/epoch/batch; dropout cannot perturb corruption draws.
    seed = (int(config.get("seed", 2026)) + epoch * 1000003 + batch_number * 1009) % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        return augment_context(x, noise_std=float(config.get("train_noise", .02)),
                               missing_rate=float(config.get("train_missing", .05)),
                               impulse_rate=float(config.get("train_impulse", .01)),
                               block_rate=float(config.get("train_block", .10)),
                               block_probability=float(config.get("train_block_prob", .25)))


def train_model(config, bundle, output_dir, device="cpu", fingerprint=None):
    """Train all batches, select epochs on clean tuning normalized CRPS only.

    Test/calibration loaders are never touched. Candidate ranking over the
    predeclared known corruption conditions is the caller's separate step.
    Interrupted epochs restart from their preceding completed epoch. Change a
    configuration (including epoch budget) only in a new output directory.
    """
    config = dict(config)
    config_digest = digest(config)
    fingerprint = fingerprint if fingerprint is not None else digest(
        dict(config=config, data=getattr(bundle, "audit", {})))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    complete_path, last_path, best_path = root / "complete.json", root / "last.pt", root / "best.pt"
    if complete_path.exists():
        complete = read_json(complete_path)
        _check_identity(complete, fingerprint, config_digest)
        if not best_path.is_file() or not last_path.is_file():
            raise RuntimeError("Completed run is missing best.pt or last.pt")
        return complete
    if best_path.exists() and not last_path.exists():
        raise RuntimeError("Orphan best.pt without last.pt; move this incomplete run aside before retrying")
    epochs, patience = int(config.get("epochs", 30)), int(config.get("patience", 7))
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    if any(config.get(k) is not None for k in ("max_train_batches", "max_val_batches", "max_test_batches")):
        raise ValueError("Production R2 training must use complete loaders, not max-batch caps")
    device = torch.device(device)
    amp_enabled = bool(config.get("amp", False))
    if amp_enabled and device.type != "cuda":
        raise ValueError("AMP requires CUDA; use amp=False for CPU")
    set_seed(int(config.get("seed", 2026)))
    model = build_model(config, bundle).to(device)
    lr = float(config.get("lr", 1e-3))
    groups = model.optimizer_groups(lr) if hasattr(model, "optimizer_groups") else model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=float(config.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=2)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    history, best, best_epoch, stale, start_epoch = [], float("inf"), 0, 0, 1
    if last_path.exists():
        state = load_checkpoint(last_path, "cpu")
        _check_identity(state, fingerprint, config_digest)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        amp_scaler.load_state_dict(state["amp_scaler"])
        history, best, best_epoch, stale = state["history"], state["best"], state["best_epoch"], state["stale"]
        start_epoch = int(state["epoch"]) + 1
        if not best_path.exists() or load_checkpoint(best_path, "cpu")["epoch"] != best_epoch:
            if state["epoch"] != best_epoch:
                raise RuntimeError("Best checkpoint is missing/inconsistent with last.pt")
            save_checkpoint(best_path, state)
        _restore_rng(state["rng"])  # after model/optimizer initialization
    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience:
            break
        # Keep sample order independent of architecture/dropout RNG consumption.
        sampler = getattr(bundle.train, "sampler", None)
        if isinstance(sampler, RandomSampler):
            sampler.generator = torch.Generator().manual_seed(int(config.get("seed", 2026)) + epoch)
        model.train()
        _sync(device)
        epoch_start = time.perf_counter()
        weighted_loss, observations, batches = 0.0, 0, 0
        for batch_number, batch in enumerate(bundle.train):
            x, y = batch["x"].cpu(), batch["y"].to(device)
            xm = batch["x_mask"].bool().cpu() & torch.isfinite(x)
            ym = batch["y_mask"].bool().to(device) & torch.isfinite(y)
            count = int(ym.sum().item())
            if count == 0:
                raise ValueError("Training batch has no observed targets")
            x = torch.where(xm, x, torch.zeros_like(x))
            corrupted, augment_mask = _augment_cpu(x, config, epoch, batch_number)
            observed = xm & augment_mask
            corrupted = torch.where(observed, corrupted, torch.zeros_like(corrupted)).to(device)
            y = torch.where(ym, y, torch.zeros_like(y))
            starts = batch["window_start"].to(device)
            optimizer.zero_grad(set_to_none=True)
            precision_context = torch.cuda.amp.autocast() if amp_enabled else contextlib.nullcontext()
            with precision_context:
                out = model(corrupted, mask=observed.to(device), cycle_index=starts, return_diagnostics=False)
                if not torch.isfinite(out.mean).all() or not torch.isfinite(out.std).all() or not (out.std > 0).all():
                    raise FloatingPointError("Nonfinite mean/std or nonpositive std during training")
                loss = gaussian_nll(y, out.mean, out.std, ym)
                loss = loss + float(config.get("mae_weight", .1)) * masked_mae(y, out.mean, ym)
                if getattr(out, "auxiliary_loss", None) is not None:
                    loss = loss + out.auxiliary_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite gradients; refusing to silently skip a step")
            clip = float(config.get("grad_clip", 1.0))
            if clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
            amp_scaler.step(optimizer)
            amp_scaler.update()
            weighted_loss += float(loss.detach().cpu()) * count
            observations += count
            batches += 1
        if batches == 0:
            raise ValueError("Training loader is empty")
        _sync(device)
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        metrics, _ = evaluate_model(model, bundle.val, device, scaler=bundle.scaler,
                                    condition="clean", diagnostics=False, save_predictions=False)
        score = float(metrics["normalized_crps"])
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite clean tuning CRPS")
        _sync(device)
        validation_seconds = time.perf_counter() - validation_start
        scheduler.step(score)
        improved = score < best
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        history.append(dict(epoch=epoch, train_loss=weighted_loss / observations,
                            training_batches=batches, training_targets=observations,
                            tuning_normalized_crps=score, learning_rates=[g["lr"] for g in optimizer.param_groups],
                            train_seconds=train_seconds, validation_seconds=validation_seconds,
                            end_to_end_seconds=time.perf_counter() - epoch_start))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), amp_scaler=amp_scaler.state_dict(), rng=_rng_state(),
                     history=history, best=best, best_epoch=best_epoch, stale=stale, epoch=epoch,
                     config=config, fingerprint=fingerprint, config_digest=config_digest,
                     checkpoint_selection="clean tuning normalized_crps")
        # last is the authoritative resume point. If interrupted before best save,
        # resume can recover the current best from last (checked below).
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        write_json(root / "history.json", history)
        print("epoch {}/{} loss={:.6g} tune_crps={:.6g} best={:.6g} train={:.1f}s val={:.1f}s".format(
            epoch, epochs, history[-1]["train_loss"], score, best, train_seconds, validation_seconds), flush=True)
    if not history:
        raise RuntimeError("No completed epoch is available")
    # Handles interruption after an atomic last save and before its best save.
    if not best_path.exists() or load_checkpoint(best_path, "cpu")["epoch"] != best_epoch:
        last = load_checkpoint(last_path, "cpu")
        if last["epoch"] != best_epoch:
            raise RuntimeError("Best checkpoint missing/inconsistent with last.pt")
        save_checkpoint(best_path, last)
    result = dict(complete=True, fingerprint=fingerprint, config_digest=config_digest,
                  best_epoch=best_epoch, best_score=best, best_tuning_normalized_crps=best,
                  epochs_completed=len(history), stopped_early=stale >= patience,
                  train_seconds=sum(h["train_seconds"] for h in history),
                  validation_seconds=sum(h["validation_seconds"] for h in history),
                  end_to_end_seconds=sum(h["end_to_end_seconds"] for h in history),
                  timing_scope="completed epochs; excludes setup/checkpoint I/O and discarded interrupted epochs",
                  best_checkpoint=str(best_path), last_checkpoint=str(last_path), history=history,
                  checkpoint_selection="clean tuning normalized_crps", amp=amp_enabled)
    write_json(complete_path, result)
    return result
