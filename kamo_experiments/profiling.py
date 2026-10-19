"""Reproducible, non-destructive timing for efficiency experiments.

All methods must be run on an otherwise idle device with the same batch, dtype,
warmup and repetition counts. CPU allocator peak memory is intentionally absent;
CUDA total peaks are process-level and incremental peaks remove the measured
baseline. FLOPs from torch.profiler have incomplete operator coverage and MUST
NOT be reported as complete model FLOPs. Dimension sweeps instantiate new models
and measure compute only; they do not establish forecasting accuracy.
"""
from __future__ import annotations

import gc
import math
import platform
import statistics
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values):
    ordered = sorted(values)
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "p95_ms": float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "samples_ms": [float(value) for value in values],
    }


def _measure(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    _synchronize(device)
    baseline = None
    if device.type == "cuda":
        baseline = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        fn()
        _synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    result = _summary(values)
    peak = int(torch.cuda.max_memory_allocated(device)) if baseline is not None else None
    result.update({
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "peak_incremental_bytes": peak - baseline if peak is not None else None,
        "memory_status": "cuda_allocator_process_peak" if peak is not None else "unavailable_on_cpu",
        "warmup": warmup,
        "repeats": repeats,
    })
    return result


def _forecast_parts(output):
    if isinstance(output, Tensor):
        return output, None, None
    if isinstance(output, Mapping):
        return output["mean"], output.get("std"), output.get("auxiliary_loss")
    if isinstance(output, (tuple, list)):
        return output[0], output[1] if len(output) > 1 else None, None
    return output.mean, output.std, getattr(output, "auxiliary_loss", None)


def _gaussian_training_loss(output, target, target_mask):
    """The same NLL + 0.1 MAE objective, without data augmentation/loading."""
    mean, std, auxiliary = _forecast_parts(output)
    valid = torch.isfinite(target) if target_mask is None else target_mask.bool() & torch.isfinite(target)
    target = torch.where(valid, target, torch.zeros_like(target))
    if std is None:
        values = (target - mean).square()
    else:
        std = std.clamp_min(1e-4)
        values = 0.5 * ((target - mean) / std).square() + std.log() + 0.5 * math.log(2 * math.pi)
        values = values + 0.1 * (target - mean).abs()
    loss = torch.where(valid, values, torch.zeros_like(values)).sum() / valid.sum().clamp_min(1)
    if auxiliary is not None:
        loss = loss + auxiliary
    return loss


def _partial_flops(model, x, mask, device, forward_kwargs):
    result = {
        "status": "unavailable",
        "counted_flops_per_batch": None,
        "counted_flops_per_sample": None,
        "counted_macs_equivalent_per_batch": None,
        "coverage": "incomplete_operator_coverage_not_total_model_flops",
        "macs_convention": "counted_flops / 2; arithmetic equivalent, not an exact MAC census",
        "counted_operators": [],
        "operators_without_flop_estimate": [],
    }
    try:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with torch.inference_mode():
            with profile(activities=activities, with_flops=True, record_shapes=True) as profiler:
                model(x, mask=mask, **forward_kwargs)
                _synchronize(device)
        positive = []
        uncounted = []
        for event in profiler.key_averages():
            count = getattr(event, "flops", 0) or 0
            if count > 0:
                positive.append({"operator": str(event.key), "flops": int(count)})
            elif str(event.key).startswith("aten::"):
                uncounted.append(str(event.key))
        total = sum(row["flops"] for row in positive)
        result["counted_operators"] = sorted(positive, key=lambda row: -row["flops"])
        result["operators_without_flop_estimate"] = sorted(set(uncounted))
        if total > 0:
            result.update({
                "status": "partial_count",
                "counted_flops_per_batch": total,
                "counted_flops_per_sample": total / int(x.shape[0]),
                "counted_macs_equivalent_per_batch": total / 2.0,
            })
        else:
            result["status"] = "no_supported_operator_counts"
    except Exception as error:
        # Unsupported profiler builds are common on older PyTorch servers. A
        # missing estimate is not a zero-cost model and does not spoil timing.
        result["error"] = "%s: %s" % (type(error).__name__, error)
    return result


def profile_model(
    model: nn.Module,
    example_x: Tensor,
    example_mask: Optional[Tensor] = None,
    device: Any = "cpu",
    warmup: int = 5,
    repeats: int = 20,
    train_warmup: int = 2,
    train_repeats: int = 5,
    example_y: Optional[Tensor] = None,
    example_y_mask: Optional[Tensor] = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    measure_flops: bool = True,
    loss_fn: Optional[Callable] = None,
    optimizer_factory: Optional[Callable] = None,
    forward_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Profile a forecasting module accepting ``model(x, mask=mask)``.

    The actual supplied model is measured, avoiding a second device-resident
    clone that would inflate total peak memory. Parameters, buffers, gradients,
    all module training modes and PyTorch RNG state are restored even on failure.
    Timing excludes data loading/transfers, corruption augmentation and validation.
    Training timings are representative *steps*, never estimated epoch duration.
    No AMP is used, and the existing model's floating dtype is preserved.
    ``loss_fn`` receives (output, target, target_mask); ``optimizer_factory``
    receives the model, and should return a fresh optimizer.
    ``forward_kwargs`` provides known inputs such as CycleNet's ``cycle_index``;
    Tensor values move to the measured device while keeping their dtype.
    """
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA have implemented synchronization/memory protocols")
    if example_x.ndim != 3 or any(int(size) < 1 for size in example_x.shape):
        raise ValueError("example_x must have nonempty [batch, context, variables] shape")
    if example_mask is not None and example_mask.shape != example_x.shape:
        raise ValueError("example_mask must have the same shape as example_x")
    if warmup < 0 or train_warmup < 0 or repeats < 1 or train_repeats < 0:
        raise ValueError("warmups/train_repeats must be nonnegative and repeats must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    tensors = list(model.parameters()) + list(model.buffers())
    original_devices = {tensor.device for tensor in tensors}
    if len(original_devices) > 1:
        raise ValueError("Profiling a model spread over several devices is not supported")
    original_device = next(iter(original_devices), torch.device("cpu"))
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    dtype = floating[0].dtype if floating else example_x.dtype
    if not example_x.is_floating_point():
        raise ValueError("example_x must be floating point")
    modes = [(module, module.training) for module in model.modules()]
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                 for name, parameter in model.named_parameters()}
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "ok",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
        "precision": str(dtype).replace("torch.", ""),
        "autocast": False,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if hasattr(torch.backends.cuda.matmul, "allow_tf32") else None,
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "batch_size": int(example_x.shape[0]),
        "seq_len": int(example_x.shape[1]),
        "num_vars": int(example_x.shape[2]),
        "parameters": parameter_count,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        "timing_scope": "model_compute_only; input_transfer/loading/augmentation/validation excluded",
        "memory_scope": "CUDA allocator process total plus increment above each phase baseline; CPU not measured",
        "training_target": "provided_target" if example_y is not None else "zero_target_shape_matched_proxy",
        "training_loss": "custom_callback" if loss_fn is not None else "Gaussian NLL + 0.1 MAE + auxiliary; MSE for point output",
        "optimizer": "custom_factory" if optimizer_factory is not None else "AdamW using model.optimizer_groups if present",
        "training_grad_clip": grad_clip,
    }
    optimizer = None
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            model.to(device)
            x = example_x.detach().to(device=device, dtype=dtype)
            mask = torch.isfinite(x) if example_mask is None else example_mask.detach().to(device=device, dtype=torch.bool)
            kwargs = {key: value.detach().to(device) if isinstance(value, Tensor) else value
                      for key, value in dict(forward_kwargs or {}).items()}
            if "mask" in kwargs:
                raise ValueError("Pass mask through example_mask, not forward_kwargs")
            model.eval()
            with torch.inference_mode():
                probe = model(x, mask=mask, **kwargs)
                mean, _, _ = _forecast_parts(probe)
                output_shape = tuple(mean.shape)
                if len(output_shape) != 3 or output_shape[0] != x.shape[0] or output_shape[2] != x.shape[2]:
                    raise ValueError("Forecast must have [batch, horizon, variables] shape")
                result["pred_len"] = int(output_shape[1])
                del probe, mean
                result["inference"] = _measure(lambda: model(x, mask=mask, **kwargs), device, warmup, repeats)
            result["inference"]["samples_per_second"] = result["batch_size"] * 1000.0 / result["inference"]["median_ms"]
            if measure_flops:
                result["flops"] = _partial_flops(model, x, mask, device, kwargs)
            else:
                result["flops"] = {"status": "not_requested", "counted_flops_per_batch": None,
                                   "coverage": "incomplete_operator_coverage_not_total_model_flops"}
            if train_repeats and result["trainable_parameters"]:
                model.train()
                y = torch.zeros(output_shape, device=device, dtype=dtype) if example_y is None else example_y.detach().to(device=device, dtype=dtype)
                if tuple(y.shape) != output_shape:
                    raise ValueError("example_y shape must equal model forecast shape")
                y_mask = None if example_y_mask is None else example_y_mask.detach().to(device=device, dtype=torch.bool)
                if y_mask is not None and y_mask.shape != y.shape:
                    raise ValueError("example_y_mask shape must equal example_y shape")
                if not (torch.isfinite(y) if y_mask is None else y_mask & torch.isfinite(y)).any():
                    raise ValueError("Training profile has no valid target elements")
                if optimizer_factory is None:
                    groups = model.optimizer_groups(learning_rate) if hasattr(model, "optimizer_groups") else model.parameters()
                    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
                else:
                    optimizer = optimizer_factory(model)
                objective = loss_fn or _gaussian_training_loss

                def train_step():
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x, mask=mask, **kwargs)
                    loss = objective(output, y, y_mask)
                    loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                result["train_step"] = _measure(train_step, device, train_warmup, train_repeats)
                result["train_step"]["samples_per_second"] = result["batch_size"] * 1000.0 / result["train_step"]["median_ms"]
                if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise FloatingPointError("Profiling train step produced nonfinite parameters")
            else:
                result["train_step"] = {"status": "not_requested" if not train_repeats else "no_trainable_parameters"}
    finally:
        # Release optimizer state before restoring the original parameter tensors.
        optimizer = None
        model.to(original_device)
        model.load_state_dict(original_state, strict=True)
        for name, parameter in model.named_parameters():
            saved = gradients[name]
            parameter.grad = None if saved is None else saved.to(device=parameter.device, dtype=parameter.dtype)
        for module, training in modes:
            module.training = training
    return result


def profile_sweep(
    build_case: Callable,
    cases: Iterable[Mapping[str, Any]],
    device: Any = "cpu",
    expected_batch_size: Optional[int] = None,
    **profile_kwargs,
):
    """Profile fresh models for predeclared variable/context/horizon cases.

    ``build_case(case)`` returns (model, x, mask) or (model, x, mask, y, y_mask).
    A callback may alternatively return a dict with these named entries. Case
    failures are explicit rows; OOM never silently reduces batch size or precision.
    Fresh random weights are acceptable for compute scaling, but not for accuracy.
    """
    rows = []
    device = torch.device(device)
    for raw_case in cases:
        case = dict(raw_case)
        built = model = x = mask = y = y_mask = None
        try:
            built = build_case(case)
            if isinstance(built, Mapping):
                model, x, mask = built["model"], built["x"], built.get("mask")
                y, y_mask = built.get("y"), built.get("y_mask")
            elif len(built) == 3:
                model, x, mask = built
            elif len(built) == 5:
                model, x, mask, y, y_mask = built
            else:
                raise ValueError("build_case must return 3 or 5 values, or a mapping")
            if expected_batch_size is not None and int(x.shape[0]) != expected_batch_size:
                raise ValueError("Actual batch differs from the predeclared common batch size")
            result = profile_model(model, x, mask, device=device, example_y=y,
                                   example_y_mask=y_mask, **profile_kwargs)
            rows.append({"case": case, "result": result})
        except Exception as error:
            message = "%s: %s" % (type(error).__name__, error)
            is_oom = "out of memory" in str(error).lower() or "outofmemory" in type(error).__name__.lower()
            rows.append({"case": case, "result": {"status": "oom" if is_oom else "error",
                         "error": message, "batch_size_policy": "unchanged_no_automatic_reduction"}})
        finally:
            built = model = x = mask = y = y_mask = None
            gc.collect()
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows
