"""Standard continual-learning metrics (SPEC §10): pure reductions over an
R-matrix (``R[task_evaluated, task_trained]``, upper-triangular, NaN
elsewhere -- clover-cl's ``RMatrix.to_array()`` convention). Ported from
the bench's ``metrics/standard.py`` (confirmed via read-only research, no
code copied): each function is a plain reduction over rows/columns/the
diagonal, NaN-graceful throughout (NaN entries are skipped; the result is
NaN when nothing valid remains).
"""

from __future__ import annotations

import numpy as np


def aggregate_accuracy(R: np.ndarray, t: int) -> float:
    """A_t: mean accuracy over every task evaluated once training has
    reached task *t* (column *t*, valid rows ``0..t``)."""
    column = R[: t + 1, t]
    valid = column[~np.isnan(column)]
    return float(np.mean(valid)) if valid.size else float("nan")


def average_incremental_accuracy(R: np.ndarray, t: int) -> float:
    """AIA: mean of ``A_0..A_t`` (the running mean of aggregate accuracy)."""
    values = [aggregate_accuracy(R, k) for k in range(t + 1)]
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def backward_transfer(R: np.ndarray, t: int) -> float:
    """BWT_t: mean of ``R[k,t] - R[k,k]`` over ``k < t`` -- negative means
    forgetting, positive means later training improved an earlier task's
    accuracy (a revisit signal). NaN at ``t == 0`` (no earlier tasks)."""
    if t == 0:
        return float("nan")
    diffs = []
    for k in range(t):
        later, first = R[k, t], R[k, k]
        if not (np.isnan(later) or np.isnan(first)):
            diffs.append(later - first)
    return float(np.mean(diffs)) if diffs else float("nan")


def forgetting(R: np.ndarray, t: int) -> float:
    """Mean drop from each earlier task's peak accuracy (over the valid
    slice ``R[k, k..t]``) to its accuracy at *t*. NaN at ``t == 0``."""
    if t == 0:
        return float("nan")
    drops = []
    for k in range(t):
        window = R[k, k : t + 1]
        valid_window = window[~np.isnan(window)]
        if valid_window.size == 0 or np.isnan(R[k, t]):
            continue
        peak = float(np.max(valid_window))
        drops.append(peak - R[k, t])
    return float(np.mean(drops)) if drops else float("nan")


def forward_transfer(R: np.ndarray, t: int) -> float:
    """FWT_t: mean of ``R[k-1,k-1]`` for ``k = 1..t``. This is PILOT's own
    approximation (confirmed via research) -- the zero-shot baseline is
    hardcoded to 0 ("no zero-shot knowledge assumed"), so this is really
    "how good were earlier tasks the moment they were first learned," not
    a true forward-transfer measurement against an independent baseline.
    Kept as-is for fidelity to the bench. NaN at ``t == 0``.
    """
    if t == 0:
        return float("nan")
    diagonals = []
    for k in range(1, t + 1):
        value = R[k - 1, k - 1]
        if not np.isnan(value):
            diagonals.append(value)
    return float(np.mean(diagonals)) if diagonals else float("nan")
