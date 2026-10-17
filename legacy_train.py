from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kamo.data import DATASETS, build_dataloaders
from kamo.engine import evaluate, fit, save_json, set_seed
from kamo.baselines import (
    ITransformerForecaster,
    PatchTSTForecaster,
    TimeMixerForecaster,
    VPNetForecaster,
)
from kamo.model import DLinear, GRUForecaster, KAMO, KAMOV5
from kamo.robustifier import (
    ReliabilityConditionedWrapper,
    ReliabilitySoftInputWrapper,
    RobustifiedInputWrapper,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final KAMO model for robust probabilistic forecasting")
    parser.add_argument("--dataset", default="ETTh1", choices=list(DATASETS))
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\34872_wfgzfou\Documents\Codex\2026-07-23\0723\datasets",
    )
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default=None, help="Optional leaf directory; prevents tuning runs from overwriting each other")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--model",
        choices=[
            "kamo", "kamo_no_robust", "kamo_v5", "kamo_v5_no_robust",
            "kamo_v5_no_clip", "kamo_v5_no_rel_norm",
            "kamo_v5_no_rel_routing",
            "kamo_v5_uniform_router", "kamo_v5_no_rel_gate",
            "kamo_v5_no_unrel_scale", "kamo_v5_no_rel_gate_scale",
            "kamo_v5_shared", "kamo_v5_separate", "kamo_v5_mask_only",
            "kamo_v5_shared_no_rel_norm", "kamo_v5_shared_no_rel_routing",
            "kamo_v5_shared_no_rel_gate", "kamo_v5_shared_no_unrel_scale",
            "kamo_v5_shared_no_mixer", "kamo_v5_shared_dense_mixer",
            "kamo_v5_no_mixer", "kamo_v5_dense_mixer",
            "dlinear", "gru",
            "patchtst", "itransformer", "timemixer", "vpnet",
            "patchtst_robust", "itransformer_robust",
            "patchtst_reliability", "itransformer_reliability",
            "patchtst_reliability_soft", "itransformer_reliability_soft",
        ],
        default="kamo",
    )
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--num-scales", type=int, default=3)
    parser.add_argument("--mixer-rank", type=int, default=16)
    parser.add_argument(
        "--mixer-type", choices=["none", "low_rank", "dense"], default="low_rank"
    )
    parser.add_argument("--robust-window", type=int, default=7)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--kamo-patch-scales",
        default="8,16,32",
        help="Comma-separated KAMO-v5 patch lengths",
    )
    parser.add_argument("--kamo-patch-layers", type=int, default=1)
    parser.add_argument("--reliability-hidden", type=int, default=16)
    parser.add_argument("--recent-window", type=int, default=12)
    parser.add_argument("--residual-trust-floor", type=float, default=0.1)
    parser.add_argument("--scale-adjustment-coefficient", type=float, default=0.2)
    parser.add_argument(
        "--reliability-strength",
        type=float,
        default=1.0,
        help="Soft strength of reliability in normalization/routing/residual roles; select on validation only.",
    )
    parser.add_argument(
        "--uncertainty-reliability-strength",
        type=float,
        default=1.0,
        help="Strength of reliability-driven variance inflation; select on validation only.",
    )
    parser.add_argument("--vp-middle-size", type=int, default=256)
    parser.add_argument("--vp-kernel-vars", type=int, default=7)
    parser.add_argument("--vp-kernel-patches", type=int, default=3)
    parser.add_argument("--vp-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--train-noise", type=float, default=0.02)
    parser.add_argument("--train-missing", type=float, default=0.05)
    parser.add_argument("--train-impulse", type=float, default=0.01)
    parser.add_argument("--train-block", type=float, default=0.10)
    parser.add_argument("--train-block-prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/validate only. Required for hyperparameter selection to avoid test leakage.",
    )
    return parser.parse_args(argv)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def build_model(args: argparse.Namespace, bundle) -> torch.nn.Module:
    if args.model.startswith("kamo_v5"):
        huber_delta = 1e6 if args.model == "kamo_v5_no_robust" else args.huber_delta
        ablation = {
            "kamo_v5_no_clip": "no_clipping",
            "kamo_v5_no_rel_norm": "no_reliability_normalization",
            "kamo_v5_shared_no_rel_norm": "no_reliability_normalization",
            "kamo_v5_no_rel_routing": "no_reliability_routing",
            "kamo_v5_shared_no_rel_routing": "no_reliability_routing",
            "kamo_v5_uniform_router": "uniform_router",
            "kamo_v5_no_rel_gate": "no_reliability_residual_gate",
            "kamo_v5_shared_no_rel_gate": "no_reliability_residual_gate",
            "kamo_v5_no_unrel_scale": "no_unreliability_scale",
            "kamo_v5_shared_no_unrel_scale": "no_unreliability_scale",
            "kamo_v5_no_rel_gate_scale": "no_reliability_gate_scale",
        }.get(args.model, "none")
        raw_scales = getattr(args, "kamo_patch_scales", "8,16,32")
        patch_scales = tuple(
            int(value.strip()) for value in str(raw_scales).split(",") if value.strip()
        )
        reliability_mode = {
            "kamo_v5_shared": "shared_learned",
            "kamo_v5_shared_no_rel_norm": "shared_learned",
            "kamo_v5_shared_no_rel_routing": "shared_learned",
            "kamo_v5_shared_no_rel_gate": "shared_learned",
            "kamo_v5_shared_no_unrel_scale": "shared_learned",
            "kamo_v5_shared_no_mixer": "shared_learned",
            "kamo_v5_shared_dense_mixer": "shared_learned",
            "kamo_v5_separate": "separate_learned",
            "kamo_v5_mask_only": "mask_only",
        }.get(args.model, "fixed_shared")
        mixer_type = {
            "kamo_v5_no_mixer": "none",
            "kamo_v5_shared_no_mixer": "none",
            "kamo_v5_dense_mixer": "dense",
            "kamo_v5_shared_dense_mixer": "dense",
        }.get(args.model, getattr(args, "mixer_type", "low_rank"))
        return KAMOV5(
            num_vars=bundle.num_vars,
            seq_len=bundle.seq_len,
            pred_len=bundle.pred_len,
            d_model=args.d_model,
            num_scales=args.num_scales,
            mixer_rank=args.mixer_rank,
            adjacency=bundle.adjacency,
            robust_window=getattr(args, "robust_window", 7),
            huber_delta=huber_delta,
            dropout=getattr(args, "dropout", 0.1),
            patch_scales=patch_scales,
            patch_layers=getattr(args, "kamo_patch_layers", 1),
            n_heads=getattr(args, "n_heads", 4),
            ff_mult=getattr(args, "ff_mult", 4),
            ablation=ablation,
            mixer_type=mixer_type,
            reliability_mode=reliability_mode,
            reliability_hidden=getattr(args, "reliability_hidden", 16),
            recent_window=getattr(args, "recent_window", 12),
            residual_trust_floor=getattr(args, "residual_trust_floor", 0.1),
            scale_adjustment_coefficient=getattr(
                args, "scale_adjustment_coefficient", 0.2
            ),
            reliability_strength=getattr(args, "reliability_strength", 1.0),
            uncertainty_reliability_strength=getattr(
                args, "uncertainty_reliability_strength", 1.0
            ),
        )
    if args.model in {"kamo", "kamo_no_robust", "kamo_no_feedback"}:
        # The matched ablation keeps the architecture and training protocol fixed,
        # but makes the Huber threshold effectively infinite. This is exactly the
        # server-side ablation that established the robustifier's contribution.
        huber_delta = 1e6 if args.model == "kamo_no_robust" else args.huber_delta
        return KAMO(
            num_vars=bundle.num_vars,
            seq_len=bundle.seq_len,
            pred_len=bundle.pred_len,
            d_model=args.d_model,
            num_scales=args.num_scales,
            mixer_rank=args.mixer_rank,
            adjacency=bundle.adjacency,
            robust_window=getattr(args, "robust_window", 7),
            huber_delta=huber_delta,
            mixer_type=getattr(args, "mixer_type", "low_rank"),
        )
    if args.model == "dlinear":
        return DLinear(bundle.num_vars, bundle.seq_len, bundle.pred_len)
    if args.model == "gru":
        return GRUForecaster(bundle.num_vars, bundle.pred_len, hidden=args.d_model)
    if args.model in {
        "patchtst", "patchtst_robust", "patchtst_reliability",
        "patchtst_reliability_soft",
    }:
        backbone = PatchTSTForecaster(
            bundle.num_vars, bundle.seq_len, bundle.pred_len,
            d_model=args.d_model,
            patch_len=getattr(args, "patch_len", 16),
            stride=getattr(args, "patch_stride", 8),
            n_layers=getattr(args, "n_layers", 2),
            n_heads=getattr(args, "n_heads", 4),
            ff_mult=getattr(args, "ff_mult", 4),
            dropout=getattr(args, "dropout", 0.1),
        )
        if args.model == "patchtst_robust":
            return RobustifiedInputWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        if args.model == "patchtst_reliability":
            return ReliabilityConditionedWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        if args.model == "patchtst_reliability_soft":
            return ReliabilitySoftInputWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        return backbone
    if args.model in {
        "itransformer", "itransformer_robust", "itransformer_reliability",
        "itransformer_reliability_soft",
    }:
        backbone = ITransformerForecaster(
            bundle.num_vars, bundle.seq_len, bundle.pred_len,
            d_model=args.d_model,
            n_layers=getattr(args, "n_layers", 2),
            n_heads=getattr(args, "n_heads", 4),
            ff_mult=getattr(args, "ff_mult", 4),
            dropout=getattr(args, "dropout", 0.1),
        )
        if args.model == "itransformer_robust":
            return RobustifiedInputWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        if args.model == "itransformer_reliability":
            return ReliabilityConditionedWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        if args.model == "itransformer_reliability_soft":
            return ReliabilitySoftInputWrapper(
                backbone,
                robust_window=getattr(args, "robust_window", 7),
                huber_delta=getattr(args, "huber_delta", 3.0),
            )
        return backbone
    if args.model == "timemixer":
        return TimeMixerForecaster(
            bundle.num_vars, bundle.seq_len, bundle.pred_len,
            d_model=args.d_model,
            num_scales=getattr(args, "num_scales", 3),
            n_layers=getattr(args, "n_layers", 2),
            dropout=getattr(args, "dropout", 0.1),
        )
    if args.model == "vpnet":
        return VPNetForecaster(
            bundle.num_vars, bundle.seq_len, bundle.pred_len,
            d_model=args.d_model,
            patch_len=getattr(args, "patch_len", 8),
            middle_size=getattr(args, "vp_middle_size", 256),
            n_layers=getattr(args, "n_layers", 2),
            kernel_vars=getattr(args, "vp_kernel_vars", 7),
            kernel_patches=getattr(args, "vp_kernel_patches", 3),
            ff_mult=getattr(args, "ff_mult", 4),
            dropout=getattr(args, "dropout", 0.1),
            reconstruction_weight=getattr(args, "vp_reconstruction_weight", 0.1),
        )
    raise ValueError(args.model)


def main() -> None:
    args = parse_args()
    if args.dataset == "physionet2012" and args.seq_len + args.pred_len > 48:
        print("Using the predeclared PhysioNet 2012 24->12 forecasting protocol.")
        args.seq_len, args.pred_len = 24, 12
    if args.dataset in {"METR-LA", "PEMS-BAY"} and (
        args.seq_len != 12 or args.pred_len != 12
    ):
        print("Using the standard 12->12 protocol for pre-windowed traffic data.")
        args.seq_len = args.pred_len = 12
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = args.max_train_batches or 2
        args.max_val_batches = args.max_val_batches or 1
        args.max_test_batches = args.max_test_batches or 1
        args.batch_size = min(args.batch_size, 4)
        args.d_model = min(args.d_model, 24)
    set_seed(args.seed)
    device = choose_device(args.device)
    bundle = build_dataloaders(
        args.dataset,
        args.data_root,
        args.seq_len,
        args.pred_len,
        args.batch_size,
        args.workers,
        args.cache_dir,
    )
    leaf = args.run_name or f"seed_{args.seed}"
    run_dir = Path(args.output_root) / args.dataset / args.model / leaf
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "num_vars": bundle.num_vars,
            "device_used": str(device),
            "parameterization": (
                "diagonal covariance, bounded-influence reliability-routed multiscale patches"
                if args.model.startswith("kamo_v5")
                else "diagonal covariance, bounded-influence robust innovation encoding"
            ),
            "effective_huber_delta": (
                1e6
                if args.model in {"kamo_no_robust", "kamo_v5_no_robust"}
                else args.huber_delta
            ),
            "reliability_mode": {
                "kamo_v5_shared": "shared_learned",
                "kamo_v5_shared_no_rel_norm": "shared_learned",
                "kamo_v5_shared_no_rel_routing": "shared_learned",
                "kamo_v5_shared_no_rel_gate": "shared_learned",
                "kamo_v5_shared_no_unrel_scale": "shared_learned",
                "kamo_v5_shared_no_mixer": "shared_learned",
                "kamo_v5_shared_dense_mixer": "shared_learned",
                "kamo_v5_separate": "separate_learned",
                "kamo_v5_mask_only": "mask_only",
            }.get(args.model, "fixed_shared" if args.model.startswith("kamo_v5") else "not_applicable"),
            "effective_mixer_type": {
                "kamo_v5_no_mixer": "none",
                "kamo_v5_shared_no_mixer": "none",
                "kamo_v5_dense_mixer": "dense",
                "kamo_v5_shared_dense_mixer": "dense",
            }.get(args.model, args.mixer_type),
            "ablation": {
                "kamo_v5_no_clip": "no_clipping_only",
                "kamo_v5_no_rel_norm": "no_reliability_weighted_normalization_only",
                "kamo_v5_shared_no_rel_norm": "no_reliability_weighted_normalization_only",
                "kamo_v5_no_rel_routing": "no_reliability_input_to_router_only",
                "kamo_v5_shared_no_rel_routing": "no_reliability_input_to_router_only",
                "kamo_v5_uniform_router": "uniform_router_only",
                "kamo_v5_no_rel_gate": "no_reliability_residual_gate_only",
                "kamo_v5_shared_no_rel_gate": "no_reliability_residual_gate_only",
                "kamo_v5_no_unrel_scale": "no_unreliability_scale_factor_only",
                "kamo_v5_shared_no_unrel_scale": "no_unreliability_scale_factor_only",
                "kamo_v5_no_rel_gate_scale": "no_reliability_residual_gate_or_scale_inflation",
            }.get(args.model, "none"),
        }
    )
    save_json(run_dir / "config.json", config)
    np.savez(
        run_dir / "scaler.npz",
        mean=bundle.scaler.mean.astype(np.float32),
        std=bundle.scaler.std.astype(np.float32),
    )

    model = build_model(args, bundle).to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"dataset={bundle.dataset_name} variables={bundle.num_vars} "
        f"model={args.model} parameters={parameters:,} device={device}"
    )
    training = fit(
        model,
        bundle.train,
        bundle.val,
        bundle.scaler,
        device,
        run_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        train_noise=args.train_noise,
        train_missing=args.train_missing,
        train_impulse=args.train_impulse,
        train_block=args.train_block,
        train_block_prob=args.train_block_prob,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    if args.skip_test:
        save_json(
            run_dir / "selection.json",
            {
                "dataset": args.dataset,
                "model": args.model,
                "seed": args.seed,
                "selection_split": "validation",
                "selection_metric": "crps",
                "best_val_crps": training["best_val_crps"],
                "best_epoch": training["best_epoch"],
                "test_accessed": False,
            },
        )
        print(f"Validation-only run saved to: {run_dir.resolve()}")
        return
    try:
        checkpoint = torch.load(training["checkpoint"], map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        checkpoint = torch.load(training["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    metrics, arrays = evaluate(
        model,
        bundle.test,
        bundle.scaler,
        device,
        max_batches=args.max_test_batches,
        diagnostics=args.model.startswith("kamo"),
    )
    metrics.update(
        {
            "dataset": args.dataset,
            "model": args.model,
            "seed": args.seed,
            "parameters": parameters,
            "best_epoch": training["best_epoch"],
        }
    )
    save_json(run_dir / "metrics.json", metrics)
    np.savez_compressed(run_dir / "predictions.npz", **arrays)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved run to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
