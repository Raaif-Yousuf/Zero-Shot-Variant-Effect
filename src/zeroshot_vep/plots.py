"""Benchmark figures: ROC/PR overlays, score distributions, AUROC vs context window.

Colors follow a fixed categorical order (never reassigned per filter) so a
scorer keeps its color across figures. Figures are saved as light-background
PNGs at 200 dpi; nothing calls ``plt.show``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Fixed categorical order (blue, orange, aqua, yellow, magenta, green, violet, red).
_CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_DAMAGING_COLOR = "#e34948"
_BENIGN_COLOR = "#2a78d6"
_DPI = 200


def _style_axes(ax: Axes) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def _save(fig: plt.Figure, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(_SURFACE)
    fig.savefig(out_path, dpi=_DPI, facecolor=_SURFACE)
    plt.close(fig)
    return out_path


def _valid_groups(df: pd.DataFrame, score_col: str, label_col: str, group_col: str) -> list[str]:
    return [
        name
        for name in df[group_col].dropna().unique()
        if df.loc[
            (df[group_col] == name) & df[score_col].notna() & df[label_col].notna(), label_col
        ].nunique()
        >= 2
    ]


def plot_roc_overlay(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "scorer",
) -> Path:
    """One ROC curve per distinct value of ``group_col``, AUROC in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _style_axes(ax)
    groups = _valid_groups(df, score_col, label_col, group_col)
    for i, name in enumerate(groups):
        sub = df[(df[group_col] == name) & df[score_col].notna() & df[label_col].notna()]
        y = sub[label_col].to_numpy(dtype=float)
        s = sub[score_col].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, s)
        auroc = roc_auc_score(y, s)
        ax.plot(
            fpr,
            tpr,
            color=_CATEGORICAL[i % len(_CATEGORICAL)],
            linewidth=2,
            solid_capstyle="round",
            label=f"{name} (AUROC {auroc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color=_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    if groups:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_pr_overlay(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "scorer",
) -> Path:
    """One precision-recall curve per distinct value of ``group_col``, AUPRC in the legend."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    _style_axes(ax)
    groups = _valid_groups(df, score_col, label_col, group_col)
    for i, name in enumerate(groups):
        sub = df[(df[group_col] == name) & df[score_col].notna() & df[label_col].notna()]
        y = sub[label_col].to_numpy(dtype=float)
        s = sub[score_col].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, s)
        auprc = average_precision_score(y, s)
        ax.plot(
            recall,
            precision,
            color=_CATEGORICAL[i % len(_CATEGORICAL)],
            linewidth=2,
            solid_capstyle="round",
            label=f"{name} (AUPRC {auprc:.3f})",
        )
    labelled = df[df[label_col].notna()]
    if len(labelled):
        prevalence = float((labelled[label_col] == 1).mean())
        ax.axhline(prevalence, color=_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall")
    if groups:
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_score_distributions(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    score_col: str = "score",
    label_col: str = "label",
    group_col: str = "scorer",
    pos_name: str = "damaging",
    neg_name: str = "benign",
) -> Path:
    """Small multiples: one panel per scorer, damaging vs benign score histograms."""
    groups = [g for g in df[group_col].dropna().unique()]
    n = max(len(groups), 1)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    handles: list = []
    labels: list[str] = []
    for idx, name in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        _style_axes(ax)
        sub = df[(df[group_col] == name) & df[score_col].notna() & df[label_col].notna()]
        neg = sub.loc[sub[label_col] == 0, score_col].to_numpy(dtype=float)
        pos = sub.loc[sub[label_col] == 1, score_col].to_numpy(dtype=float)
        if len(neg):
            ax.hist(neg, bins=30, color=_BENIGN_COLOR, alpha=0.55, density=True, label=neg_name)
        if len(pos):
            ax.hist(pos, bins=30, color=_DAMAGING_COLOR, alpha=0.55, density=True, label=pos_name)
        ax.set_title(str(name), fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04)
        )
    fig.tight_layout()
    return _save(fig, out_path)


def plot_auroc_vs_window(
    df: pd.DataFrame,
    out_path: Path | str,
    *,
    model_col: str = "model",
    window_col: str = "window",
    auroc_col: str = "auroc",
    lo_col: str = "auroc_lo",
    hi_col: str = "auroc_hi",
) -> Path:
    """AUROC vs context window (log2 x-axis), one line per model, with CI bands."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _style_axes(ax)
    models = list(df[model_col].dropna().unique())
    for i, name in enumerate(models):
        sub = df[df[model_col] == name].sort_values(window_col)
        color = _CATEGORICAL[i % len(_CATEGORICAL)]
        ax.plot(
            sub[window_col],
            sub[auroc_col],
            color=color,
            marker="o",
            markersize=6,
            linewidth=2,
            solid_capstyle="round",
            label=str(name),
        )
        if lo_col in sub.columns and hi_col in sub.columns:
            lo = sub[lo_col].to_numpy(dtype=float)
            hi = sub[hi_col].to_numpy(dtype=float)
            if np.isfinite(lo).any() and np.isfinite(hi).any():
                ax.fill_between(sub[window_col], lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context window (bp)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs context window")
    if models:
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, out_path)
