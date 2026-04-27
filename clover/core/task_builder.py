"""Converts an :class:`~clover.core.overlap_spec.OverlapSpec` and raw dataset
metadata into the concrete per-task class lists that
:class:`~clover.core.data_manager.OverlapDataManager` consumes.
"""

from __future__ import annotations

import logging
from typing import List

from clover.core.overlap_spec import OverlapSpec

logger = logging.getLogger(__name__)


def build_tasks(
    total_classes: int,
    init_cls: int,
    increment: int,
    class_order: List[int],
    overlap_spec: OverlapSpec,
    preserve_size: bool = True,
) -> List[List[int]]:
    """Return per-task lists of *remapped* class IDs.

    When ``overlap_spec.mode == "none"`` the result is identical to PILOT's
    default slicing behaviour: task 0 gets ``class_order[0:init_cls]``, task k
    gets ``class_order[init_cls + (k-1)*increment : init_cls + k*increment]``,
    and any remainder forms a final task.

    For overlap modes the baseline split is computed first, then shared classes
    from ``overlap_spec.pairs`` are injected.  When ``preserve_size=True``
    (default), injecting a shared class into a non-final task evicts the
    highest-index non-shared class from that task so its size stays constant.
    Evicted classes are collected and appended to the final task at the end;
    only the final task is allowed to grow.  When every class in a task is
    already shared (nothing evictable), the task grows and a warning is emitted.

    When ``preserve_size=False`` the v0.1 behaviour is used: tasks simply grow
    when shared classes are appended.

    Args:
        total_classes: Number of classes in the underlying dataset.
        init_cls: Number of classes in the first task.
        increment: Number of new classes per subsequent task.
        class_order: Permutation of ``range(total_classes)`` produced by the
            seeded shuffle.
        overlap_spec: Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
        preserve_size: When ``True`` (default) non-final tasks keep their
            original size by evicting the highest non-shared class to the final
            task.  When ``False`` tasks may grow.

    Returns:
        ``task_class_lists[t]`` is a list of remapped class IDs for task *t*.
    """
    assert init_cls <= total_classes, "init_cls exceeds total number of classes."

    # --- Step 1: baseline PILOT-style slicing ---
    increments: List[int] = [init_cls]
    while sum(increments) + increment < total_classes:
        increments.append(increment)
    offset = total_classes - sum(increments)
    if offset > 0:
        increments.append(offset)

    task_class_lists: List[List[int]] = []
    cursor = 0
    for inc in increments:
        task_class_lists.append(list(range(cursor, cursor + inc)))
        cursor += inc

    if overlap_spec.mode == "none":
        return task_class_lists

    # --- Step 2: inject shared classes from pairs ---
    n_tasks = len(task_class_lists)
    last = n_tasks - 1

    # Union of all shared classes across every pair — used to identify evictable classes.
    all_shared: set = set()
    for pair in overlap_spec.pairs:
        all_shared.update(pair.shared_classes)

    displaced: List[int] = []

    for pair in overlap_spec.pairs:
        t_a, t_b = pair.tasks
        if t_a >= n_tasks or t_b >= n_tasks:
            raise ValueError(
                f"OverlapPair references task index {max(t_a, t_b)} but only "
                f"{n_tasks} tasks exist (0-based)."
            )
        for cls in pair.shared_classes:
            for t in (t_a, t_b):
                if cls not in task_class_lists[t]:
                    if preserve_size and t != last:
                        evictable = [
                            c for c in task_class_lists[t] if c not in all_shared
                        ]
                        if evictable:
                            victim = max(evictable)
                            task_class_lists[t].remove(victim)
                            displaced.append(victim)
                        else:
                            logger.warning(
                                "Task %d: all classes are shared; cannot evict to "
                                "preserve size — growing task instead.",
                                t,
                            )
                    task_class_lists[t].append(cls)

    # Displaced classes land in the final task (may grow it).
    for cls in displaced:
        if cls not in task_class_lists[last]:
            task_class_lists[last].append(cls)

    return task_class_lists


def compute_increments(total_classes: int, init_cls: int, increment: int) -> List[int]:
    """Return the PILOT-style increments list for diagnostic use.

    Args:
        total_classes: Total number of classes.
        init_cls: Classes in the first task.
        increment: Classes added per subsequent task.

    Returns:
        List of per-task class counts.
    """
    increments: List[int] = [init_cls]
    while sum(increments) + increment < total_classes:
        increments.append(increment)
    offset = total_classes - sum(increments)
    if offset > 0:
        increments.append(offset)
    return increments
