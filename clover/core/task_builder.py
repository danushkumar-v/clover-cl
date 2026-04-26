"""Converts an :class:`~clover.core.overlap_spec.OverlapSpec` and raw dataset
metadata into the concrete per-task class lists that
:class:`~clover.core.data_manager.OverlapDataManager` consumes.
"""

from __future__ import annotations

from typing import List

from clover.core.overlap_spec import OverlapSpec


def build_tasks(
    total_classes: int,
    init_cls: int,
    increment: int,
    class_order: List[int],
    overlap_spec: OverlapSpec,
) -> List[List[int]]:
    """Return per-task lists of *remapped* class IDs.

    When ``overlap_spec.mode == "none"`` the result is identical to PILOT's
    default slicing behaviour: task 0 gets ``class_order[0:init_cls]``, task k
    gets ``class_order[init_cls + (k-1)*increment : init_cls + k*increment]``,
    and any remainder forms a final task.

    For overlap modes, the baseline split is computed first, then shared classes
    declared in ``overlap_spec.pairs`` are injected into the listed tasks.  If
    a shared class was already assigned to a task naturally, it is a no-op.
    Tasks may grow beyond *increment* when new shared classes are injected.

    Args:
        total_classes: Number of classes in the underlying dataset.
        init_cls: Number of classes in the first task.
        increment: Number of new classes per subsequent task.
        class_order: Permutation of ``range(total_classes)`` produced by the
            seeded shuffle (PILOT-compatible via
            :func:`~clover.utils.seeding.pilot_class_order`).
        overlap_spec: Validated :class:`~clover.core.overlap_spec.OverlapSpec`.

    Returns:
        ``task_class_lists[t]`` is a list of remapped class IDs for task *t*.
        Remapped class ID *i* corresponds to ``class_order[i]`` in original space.
    """
    assert init_cls <= total_classes, "init_cls exceeds total number of classes."

    # --- Step 1: baseline PILOT-style slicing ---
    # We work in *remapped* index space (0 .. total_classes-1).
    # class_order[i] = original class that maps to remapped index i.
    # PILOT's increments list:
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
    for pair in overlap_spec.pairs:
        t_a, t_b = pair.tasks
        if t_a >= n_tasks or t_b >= n_tasks:
            raise ValueError(
                f"OverlapPair references task index {max(t_a, t_b)} but only "
                f"{n_tasks} tasks exist (0-based)."
            )
        for cls in pair.shared_classes:
            if cls not in task_class_lists[t_a]:
                task_class_lists[t_a].append(cls)
            if cls not in task_class_lists[t_b]:
                task_class_lists[t_b].append(cls)

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
