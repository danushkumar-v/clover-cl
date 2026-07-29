"""Derived metrics computed on top of an ``R_matrix.npy``, not shipped by
``clover report`` itself.

See ``clover/evaluation/evaluator.py``: ``R[task_evaluated, task_trained]``,
upper-triangular (``R[i, j]`` defined for ``j >= i``). Row ``i`` is the
held-out accuracy of the class block first introduced at experience ``i``;
column ``j`` is the training checkpoint after experience ``j``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def category_level_retention(R: np.ndarray, source_row: int, echo_row: int) -> Dict[str, float]:
    """Source-block accuracy plus echo-block accuracy at the final checkpoint.

    In echo scenarios (``exact_replay``, ``long_range_revisit``,
    ``mid_range_revisit``, ``partial_overlap``) a returning category is
    relabeled under a fresh id (an "echo") rather than keeping its original
    id. ``Long_Range_Retention`` (the source row's final-column accuracy)
    reads this as forgetting: it collapses once the echo appears. But the
    model hasn't actually forgotten the category -- it has split its
    prediction mass between the original id (``source_row``) and the new
    one (``echo_row``). Summing the two final-column accuracies recovers
    (approximately) the pre-echo accuracy, which is the signature of label
    aliasing rather than genuine forgetting.

    Args:
        R: ``[n, n]`` accuracy matrix, ``R[task_evaluated, task_trained]``.
        source_row: row of the class block whose identity is echoed later
            (e.g. row 0 for most scenarios, row 4 for ``mid_range_revisit``).
        echo_row: row of the block holding the echo's fresh label id
            (the last row in every scenario that has one).

    Returns:
        ``{"source_pre": ..., "source_post": ..., "echo": ..., "sum": ...}``
        where ``source_pre`` is the source block's accuracy one checkpoint
        before the echo appears, ``source_post`` is its accuracy after
        (this equals ``Long_Range_Retention``), ``echo`` is the echo
        block's own accuracy, and ``sum = source_post + echo``.
    """
    final_col = R.shape[1] - 1
    source_pre = float(R[source_row, final_col - 1]) if final_col - 1 >= source_row else float("nan")
    source_post = float(R[source_row, final_col])
    echo = float(R[echo_row, final_col])
    return {
        "source_pre": source_pre,
        "source_post": source_post,
        "echo": echo,
        "sum": source_post + echo,
    }
