"""Diagnostic visualisations for CLOVER managers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from clover.core.data_manager import OverlapDataManager


def plot_overlap_matrix(
    manager: "OverlapDataManager",
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Save a heatmap of the task-vs-task class-overlap matrix.

    Cell ``(i, j)`` shows the number of classes shared between tasks *i* and *j*.
    The diagonal shows each task's total class count.

    Args:
        manager: A configured :class:`~clover.core.data_manager.OverlapDataManager`.
        save_path: File path for the saved PNG image.
        title: Optional plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisations. "
            "Install it with: pip install matplotlib"
        ) from exc

    mat = manager.get_overlap_matrix()
    T = mat.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, T), max(5, T - 1)))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="# shared classes")

    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"T{i}" for i in range(T)])
    ax.set_yticklabels([f"T{i}" for i in range(T)])
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")
    ax.set_title(title or f"Class Overlap Matrix — {manager.dataset_name}")

    for i in range(T):
        for j in range(T):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_class_frequency(
    manager: "OverlapDataManager",
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Save a bar chart of how many tasks each class appears in.

    Args:
        manager: A configured :class:`~clover.core.data_manager.OverlapDataManager`.
        save_path: File path for the saved PNG image.
        title: Optional plot title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualisations. "
            "Install it with: pip install matplotlib"
        ) from exc


    freq = {}
    for cls_list in manager._task_class_lists:
        for cls in cls_list:
            freq[cls] = freq.get(cls, 0) + 1

    classes = sorted(freq.keys())
    counts = [freq[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) // 4), 4))
    ax.bar(classes, counts, color="steelblue", width=0.8)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8, label="appears once")
    ax.set_xlabel("Remapped Class ID")
    ax.set_ylabel("# Tasks Containing Class")
    ax.set_title(title or f"Class Frequency Across Tasks — {manager.dataset_name}")
    ax.legend()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
