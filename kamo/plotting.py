from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#222222",
    "gray": "#777777",
    "light_gray": "#D9D9D9",
}


def mm(value: float) -> float:
    return value / 25.4


def publication_style() -> None:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        family = "Arial"
    except ValueError:
        family = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": None,
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=PALETTE["black"],
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), transparent=True, bbox_inches=None)
    fig.savefig(base.with_suffix(".png"), dpi=600, facecolor="white", bbox_inches=None)
    plt.close(fig)


def _box(ax, xy, width, height, text, color, subtitle=None):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.7,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.10),
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.60, text, ha="center", va="center", fontsize=7)
    if subtitle:
        ax.text(
            xy[0] + width / 2,
            xy[1] + height * 0.27,
            subtitle,
            ha="center",
            va="center",
            fontsize=5.5,
            color=PALETTE["gray"],
        )


def _arrow(ax, start, end, color="#555555", curved=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.65,
            color=color,
            connectionstyle=f"arc3,rad={curved}",
        )
    )


def architecture_figure(output_dir: Path) -> None:
    publication_style()
    fig, ax = plt.subplots(figsize=(mm(183), mm(62)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.02, 0.37), 0.11, 0.25, "Observed\ncontext", PALETTE["black"], r"$y_t,\,m_t$")
    _box(ax, (0.16, 0.37), 0.14, 0.25, "Causal\nrobustifier", PALETTE["orange"], "median / MAD / Huber")
    _box(ax, (0.35, 0.57), 0.16, 0.25, "Direct backbone", PALETTE["blue"], "multi-horizon safety path")
    _box(ax, (0.35, 0.16), 0.16, 0.25, "Robust innovation\nencoder", PALETTE["green"], "bounded features / GRU")
    _box(ax, (0.56, 0.16), 0.16, 0.25, "Variable mixer", PALETTE["sky"], "low-rank / graph")
    _box(ax, (0.77, 0.16), 0.13, 0.25, "Bounded\nresidual", PALETTE["vermillion"], "zero-start correction")
    _box(ax, (0.77, 0.57), 0.20, 0.25, "Forecast distribution", PALETTE["purple"], r"$\mu_{t+h},\,\sigma_{t+h}$")
    _arrow(ax, (0.13, 0.495), (0.16, 0.495))
    _arrow(ax, (0.30, 0.52), (0.35, 0.68))
    _arrow(ax, (0.30, 0.45), (0.35, 0.29))
    _arrow(ax, (0.51, 0.285), (0.56, 0.285))
    _arrow(ax, (0.72, 0.285), (0.77, 0.285))
    _arrow(ax, (0.51, 0.695), (0.77, 0.695))
    _arrow(ax, (0.84, 0.41), (0.84, 0.57))
    ax.text(0.535, 0.055, "bounded innovations and reliability-aware uncertainty", ha="center", va="center", fontsize=6, color=PALETTE["green"])
    ax.text(0.02, 0.92, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    save_figure(fig, output_dir / "figure_1_architecture")


def forecast_figure(run_dir: Path, output_dir: Path) -> None:
    publication_style()
    arrays = np.load(run_dir / "predictions.npz")
    context = arrays["context"]
    target, mean, std = arrays["target"], arrays["mean"], arrays["std"]
    n_vars = target.shape[-1]
    variable_ids = np.unique(np.linspace(0, n_vars - 1, 4, dtype=int))
    fig, axes = plt.subplots(2, 2, figsize=(mm(183), mm(95)), constrained_layout=True)
    axes = axes.ravel()
    for panel, (ax, variable) in enumerate(zip(axes, variable_ids)):
        c = context[0, :, variable]
        y = target[0, :, variable]
        mu = mean[0, :, variable]
        sigma = std[0, :, variable]
        tc = np.arange(-len(c) + 1, 1)
        tf = np.arange(1, len(y) + 1)
        ax.plot(tc, c, color=PALETTE["gray"], label="Context")
        ax.plot(tf, y, color=PALETTE["black"], label="Observation")
        ax.plot(tf, mu, color=PALETTE["blue"], label="KAMO")
        ax.fill_between(
            tf,
            mu - 1.645 * sigma,
            mu + 1.645 * sigma,
            color=PALETTE["blue"],
            alpha=0.16,
            linewidth=0,
            label="90% interval",
        )
        ax.axvline(0, color=PALETTE["light_gray"], linewidth=0.7)
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("Value")
        ax.set_title(f"Variable {variable}")
        clean_axis(ax)
        panel_letter(ax, chr(ord("a") + panel))
    axes[0].legend(ncol=2, loc="best")
    save_figure(fig, output_dir / "figure_forecasts")


def mechanism_figure(run_dir: Path, output_dir: Path) -> None:
    publication_style()
    arrays = np.load(run_dir / "predictions.npz")
    required = ["diag_standardized_innovation", "diag_reliability"]
    if not all(key in arrays for key in required):
        return
    innovation = np.abs(arrays["diag_standardized_innovation"])
    reliability = arrays["diag_reliability"]
    z = innovation[0, :, 0]
    r = reliability[0, :, 0]
    time = np.arange(len(z))
    fig, axes = plt.subplots(1, 3, figsize=(mm(183), mm(51)), constrained_layout=True)
    axes[0].plot(time, z, color=PALETTE["orange"])
    axes[0].axhline(3.0, color=PALETTE["gray"], linestyle="--", linewidth=0.7)
    axes[0].set(xlabel="Context step", ylabel=r"$|z_t|$", title="Robust innovation")
    axes[1].plot(time, r, color=PALETTE["blue"])
    axes[1].set(xlabel="Context step", ylabel="Reliability", title="Huber reliability")
    flat_z = innovation.reshape(-1)
    flat_r = reliability.reshape(-1)
    valid = np.isfinite(flat_z) & np.isfinite(flat_r)
    flat_z, flat_r = flat_z[valid], flat_r[valid]
    edges = np.quantile(flat_z, np.linspace(0, 1, 9))
    centers, medians, lower, upper = [], [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        keep = (flat_z >= left) & (flat_z <= right)
        if keep.sum() == 0:
            continue
        centers.append(np.median(flat_z[keep]))
        medians.append(np.median(flat_r[keep]))
        lower.append(np.quantile(flat_r[keep], 0.25))
        upper.append(np.quantile(flat_r[keep], 0.75))
    centers, medians = np.asarray(centers), np.asarray(medians)
    axes[2].plot(centers, medians, "o-", color=PALETTE["vermillion"])
    axes[2].fill_between(centers, lower, upper, color=PALETTE["vermillion"], alpha=0.15, linewidth=0)
    axes[2].set(xlabel=r"$|z_t|$", ylabel="Observation reliability", title="Bounded influence")
    for i, ax in enumerate(axes):
        clean_axis(ax)
        panel_letter(ax, chr(ord("a") + i))
    save_figure(fig, output_dir / "figure_mechanism")


def training_figure(run_dir: Path, output_dir: Path) -> None:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return
    publication_style()
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(mm(89), mm(37)), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], color=PALETTE["blue"])
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(history["epoch"], history["val_crps"], color=PALETTE["orange"])
    axes[1].set(xlabel="Epoch", ylabel="Validation CRPS")
    for i, ax in enumerate(axes):
        clean_axis(ax)
        panel_letter(ax, chr(ord("a") + i))
    save_figure(fig, output_dir / "figure_training")


def robustness_figure(csv_path: Path, output_dir: Path) -> None:
    publication_style()
    frame = pd.read_csv(csv_path)
    metrics = [metric for metric in ("mae", "crps", "calibration_error_90") if metric in frame]
    fig, axes = plt.subplots(1, len(metrics), figsize=(mm(183), mm(52)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    models = list(frame["model"].drop_duplicates())
    conditions = list(frame["condition"].drop_duplicates())
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    x = np.arange(len(conditions))
    for ax, metric in zip(axes, metrics):
        for model, color in zip(models, colors):
            subset = frame[frame["model"] == model].set_index("condition").reindex(conditions)
            ax.plot(x, subset[metric], marker="o", label=model, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in conditions], rotation=25, ha="right")
        ax.set_ylabel(metric.replace("_", " ").upper())
        clean_axis(ax)
    axes[0].legend()
    for i, ax in enumerate(axes):
        panel_letter(ax, chr(ord("a") + i))
    save_figure(fig, output_dir / "figure_robustness")


def make_run_figures(run_dir: Path, robustness_csv: Optional[Path] = None) -> Path:
    output_dir = run_dir / "figures"
    architecture_figure(output_dir)
    if (run_dir / "predictions.npz").exists():
        forecast_figure(run_dir, output_dir)
        mechanism_figure(run_dir, output_dir)
    training_figure(run_dir, output_dir)
    if robustness_csv and robustness_csv.exists():
        robustness_figure(robustness_csv, output_dir)
    return output_dir
