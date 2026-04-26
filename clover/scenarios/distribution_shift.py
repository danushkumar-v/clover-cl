"""Distribution shift scenario.

The same classes appear in two tasks but each task receives a different
slice of the image pool, approximating a shift in capture conditions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    pair: Tuple[int, int] = (0, 1),
    shift_classes: Optional[List[int]] = None,
    split_ratio: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a distribution-shift overlap spec.

    Both tasks see the same class labels but get disjoint image subsets,
    modelling a change in feature distribution for the same concept.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        pair: ``(task_a, task_b)`` indices.
        shift_classes: Explicit list of remapped class IDs to shift.  If
            ``None``, all classes of task_a are used.
        split_ratio: Fraction of images task_a receives (task_b gets the rest).
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    t_a, t_b = pair
    for t in (t_a, t_b):
        if t < 0 or t >= nb_tasks:
            raise ValueError(f"Task index {t} out of range [0, {nb_tasks - 1}].")

    if shift_classes is not None:
        shared = list(shift_classes)
    else:
        task_a_start = sum(increments[:t_a])
        shared = list(range(task_a_start, task_a_start + increments[t_a]))

    pair_obj = OverlapPair(tasks=(t_a, t_b), shared_classes=shared)
    spec = OverlapSpec(
        mode="distribution_shift",
        pairs=[pair_obj],
        image_split=ImageSplit(strategy="disjoint", ratio=split_ratio),
        seed=seed,
    )
    spec.validate()
    return spec
