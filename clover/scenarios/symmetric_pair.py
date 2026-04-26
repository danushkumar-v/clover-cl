"""Symmetric pair scenario.

Tasks M and N share exactly 50% of their classes; each retains 50% unique
classes.  Provides a clean controlled overlap baseline for ablation studies.
"""

from __future__ import annotations

from typing import Tuple

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments
from clover.utils.seeding import get_rng


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    pair: Tuple[int, int] = (0, 1),
    image_split_strategy: str = "duplicate",
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a symmetric-pair overlap spec.

    Exactly half of task_a's classes are injected into task_b and vice versa,
    resulting in each task having 50% shared + 50% unique classes.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        pair: ``(task_a, task_b)`` — the two symmetric tasks.
        image_split_strategy: Image assignment strategy.
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

    task_a_start = sum(increments[:t_a])
    task_a_classes = list(range(task_a_start, task_a_start + increments[t_a]))

    n_shared = max(1, len(task_a_classes) // 2)
    rng = get_rng(seed)
    shared = sorted(rng.choice(task_a_classes, size=n_shared, replace=False).tolist())

    pair_obj = OverlapPair(tasks=(t_a, t_b), shared_classes=shared)
    spec = OverlapSpec(
        mode="symmetric_pair",
        pairs=[pair_obj],
        image_split=ImageSplit(strategy=image_split_strategy),
        seed=seed,
    )
    spec.validate()
    return spec
