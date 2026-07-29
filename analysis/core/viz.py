"""Plotting helpers shared by every batch notebook.

Style, palette, and mark specs follow the project's `dataviz` skill:
categorical hues are assigned in a fixed order (never cycled, never tied to
a hardcoded scenario/method name -- callers pass their own ordered list),
sequential magnitude uses one hue light->dark, diverging uses two hues
around a neutral gray midpoint, and every multi-series chart carries a
legend so identity never rests on color alone.

Ported and adapted from ``../clover-pilot-bench/analysis/v2_fixed_size/clover_viz.py``
(single-seed v1 bench) for v2's seed-averaged (3 seeds) data: bar/line
helpers here take a tidy per-task DataFrame with a ``seed`` column and
aggregate mean +/- SEM across seeds rather than plotting one seed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------
# Palette (validated: see analysis/README.md and the dataviz skill's
# references/palette.md). Fixed order -- position N always gets slot N,
# regardless of which scenario/method happens to occupy it.
# --------------------------------------------------------------------------
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
MUTED_GRAY = "#898781"  # chrome/"other" -- zero chroma, never confusable with a hue slot

SEQUENTIAL_BLUE = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#104281", "#0d366b",
]
DIVERGING_BLUE_RED = LinearSegmentedColormap.from_list(
    "clover_diverging", ["#104281", "#f0efec", "#e34948"]
)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

FIG_DPI = 300


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#898781",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(fig, figures_dir: Path, name: str) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{ext}")
    print(f"saved {figures_dir / name}.png  +  .pdf")


def assign_colors(items: Sequence[str]) -> Dict[str, str]:
    """Fixed-order categorical color per item, by the order given.

    Past the 8 validated slots, items fall back to muted gray rather than a
    cycled/regenerated hue (a chart form with >8 series should facet or fold
    into "Other" instead; this is the safety net, not the recommended path).
    """
    colors = {}
    for i, item in enumerate(items):
        colors[item] = CATEGORICAL[i] if i < len(CATEGORICAL) else MUTED_GRAY
    return colors


# --------------------------------------------------------------------------
# Seed aggregation
# --------------------------------------------------------------------------
def seed_mean_sem(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    """Mean and SEM across the ``seed`` column for one metric.

    *group_cols* should not include ``seed`` or ``value`` -- typically
    ``["scenario", "task_idx"]`` or ``["method", "scenario"]``.
    """
    d = df[df["metric"] == metric].dropna(subset=["value"])
    g = d.groupby(group_cols, observed=True)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["sem"] = out["std"] / out["n"].clip(lower=1) ** 0.5
    out["sem"] = out["sem"].fillna(0.0)
    return out


def final_task_per_seed(df: pd.DataFrame, metric: str, group_cols: List[str]) -> pd.DataFrame:
    """Each (group, seed)'s value at its own last recorded task_idx, then
    mean/SEM across seeds -- for metrics only defined at the final task."""
    d = df[df["metric"] == metric].dropna(subset=["value"])
    last_idx = d.groupby(group_cols + ["seed"], observed=True)["task_idx"].transform("max")
    last = d[d["task_idx"] == last_idx]
    g = last.groupby(group_cols, observed=True)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["sem"] = out["std"] / out["n"].clip(lower=1) ** 0.5
    out["sem"] = out["sem"].fillna(0.0)
    return out


# --------------------------------------------------------------------------
# Bar / trajectory plots (scenario or method on the categorical axis)
# --------------------------------------------------------------------------
def fig_bar_by_group(
    agg: pd.DataFrame, group_col: str, order: Sequence[str], ylabel: str, title: str,
    labels: Optional[Dict[str, str]] = None,
):
    """Bar chart with SEM error bars, one bar per *group_col* value in *order*."""
    colors = assign_colors(list(order))
    agg = agg.set_index(group_col).reindex(order)
    fig, ax = plt.subplots(figsize=(0.9 * len(order) + 2, 4.8))
    xs = np.arange(len(order))
    ax.bar(xs, agg["mean"].to_numpy(), yerr=agg["sem"].to_numpy(),
           color=[colors[g] for g in order], edgecolor="white", linewidth=0.6,
           capsize=3, error_kw={"linewidth": 1, "ecolor": "#52514e"})
    ax.set_xticks(xs)
    tick_labels = [labels.get(g, g) if labels else g for g in order]
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_trajectory_by_group(
    traj: pd.DataFrame, group_col: str, order: Sequence[str], ylabel: str, title: str,
    labels: Optional[Dict[str, str]] = None,
):
    """One line per *group_col* value (mean across seeds), shaded +/- SEM band."""
    colors = assign_colors(list(order))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for g in order:
        sub = traj[traj[group_col] == g].sort_values("task_idx")
        if sub.empty:
            continue
        x = sub["task_idx"].to_numpy()
        y = sub["mean"].to_numpy()
        sem = sub["sem"].to_numpy()
        lbl = labels.get(g, g) if labels else g
        ax.plot(x, y, marker="o", ms=4, lw=1.8, color=colors[g], label=lbl)
        ax.fill_between(x, y - sem, y + sem, color=colors[g], alpha=0.15, linewidth=0)
    ax.set_xlabel("task")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# R-matrix heatmaps
# --------------------------------------------------------------------------
def mean_r_matrix(r_matrices: Dict[str, np.ndarray], run_ids: Sequence[str]) -> np.ndarray:
    """Elementwise mean of the R matrices for *run_ids* (e.g. the 3 seeds of
    one method x scenario), NaN-aware (each cell stays NaN only if every
    seed is NaN there)."""
    stack = np.stack([r_matrices[r] for r in run_ids])
    with warnings.catch_warnings():
        # below-diagonal cells are NaN in every seed by construction (R[i,j]
        # only defined for j >= i) -- not a data problem, just triangular.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(stack, axis=0)


def fig_r_matrix(R: np.ndarray, title: str, ax=None):
    """Single R[task_evaluated, task_trained] heatmap, sequential blue."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
    cmap = LinearSegmentedColormap.from_list("clover_seq_blue", SEQUENTIAL_BLUE)
    cmap.set_bad("#f0efec")
    im = ax.imshow(R, vmin=0, vmax=1, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("trained task $t$")
    ax.set_ylabel("eval task $k$")
    n = R.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(labelsize=7)
    if own_fig:
        fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy")
        fig.tight_layout()
        return fig
    return im


def fig_source_block_trajectory(
    mean_r_by_scenario: Dict[str, np.ndarray],
    source_row_by_scenario: Dict[str, int],
    order: Sequence[str],
    labels: Optional[Dict[str, str]] = None,
):
    """Source-block accuracy R[source_row, :] per scenario, echo-task drop annotated.

    One line per scenario (mean-of-seeds R matrix); the final step is where
    the echo class appears for scenarios that have one. An arrow + delta
    label mark the collapse at that step.
    """
    colors = assign_colors(list(order))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for sc in order:
        R = mean_r_by_scenario[sc]
        row = source_row_by_scenario[sc]
        y = R[row, row:]
        x = np.arange(row, R.shape[1])
        lbl = labels.get(sc, sc) if labels else sc
        ax.plot(x, y, marker="o", ms=4, lw=1.8, color=colors[sc], label=lbl)
        drop = y[-2] - y[-1] if len(y) >= 2 else 0.0
        if drop > 0.02:
            ax.annotate(
                f"-{drop:.2f}",
                xy=(x[-1], y[-1]), xytext=(x[-1] - 0.15, y[-1] - 0.09),
                fontsize=8.5, color=colors[sc], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=colors[sc], lw=1.2),
            )
    ax.set_xlabel("training checkpoint (task)")
    ax.set_ylabel("source-block accuracy")
    ax.set_title("Source-block accuracy trajectory: the echo-task drop is aliasing, not forgetting")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Framework-validation plots (b02-style batches: completeness, not comparison)
# --------------------------------------------------------------------------
def fig_completeness_heatmap(counts: pd.DataFrame, method_order: Sequence[str], scenario_order: Sequence[str]):
    """*counts*: rows method, columns scenario, values = seeds completed (0-3)."""
    piv = counts.reindex(index=method_order, columns=scenario_order).fillna(0)
    fig, ax = plt.subplots(figsize=(1.1 * len(scenario_order) + 2, 0.5 * len(method_order) + 2))
    cmap = LinearSegmentedColormap.from_list("clover_seq_blue", SEQUENTIAL_BLUE)
    im = ax.imshow(piv.to_numpy(), vmin=0, vmax=3, cmap=cmap, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = int(piv.to_numpy()[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v >= 2 else "#0b0b0b", fontsize=9)
    ax.set_xticks(range(len(scenario_order)))
    ax.set_xticklabels(scenario_order, rotation=30, ha="right")
    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(method_order)
    ax.set_title("Runs completed (of 3 seeds) -- framework validation (untrained stand-in backbones)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="seeds done")
    fig.tight_layout()
    return fig
