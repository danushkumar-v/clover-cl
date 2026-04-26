"""Exact replay scenario: Task N's classes are identical to Task 0's classes."""

from __future__ import annotations

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    revisit_at_task: int = -1,
    image_split_strategy: str = "duplicate",
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build an :class:`~clover.core.overlap_spec.OverlapSpec` for exact replay.

    Task 0's full class set reappears verbatim at *revisit_at_task*.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Number of classes in task 0.
        increment: Classes added per subsequent task.
        revisit_at_task: Task index that replays task 0.  ``-1`` means the
            last task (default).
        image_split_strategy: How images are split for shared classes.
            ``"duplicate"`` (default) gives both tasks the full image pool.
        seed: RNG seed for image assignment.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    if revisit_at_task < 0:
        revisit_at_task = nb_tasks + revisit_at_task

    if revisit_at_task <= 0 or revisit_at_task >= nb_tasks:
        raise ValueError(
            f"revisit_at_task={revisit_at_task} is out of range [1, {nb_tasks - 1}]."
        )

    shared = list(range(init_cls))
    pair = OverlapPair(tasks=(0, revisit_at_task), shared_classes=shared)
    spec = OverlapSpec(
        mode="exact_replay",
        pairs=[pair],
        image_split=ImageSplit(strategy=image_split_strategy),
        seed=seed,
    )
    spec.validate()
    return spec
