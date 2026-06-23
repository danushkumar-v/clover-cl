"""Cumulative drift scenario.

The class set grows task by task: {a} → {a,b} → {a,b,c}, so anchor classes
appear in many consecutive tasks.  This simulates a stream where core
concepts accumulate additional related classes over time.
"""

from __future__ import annotations

from typing import List

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    n_anchors: int = 3,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a fixed-size cumulative-drift spec.

    A small set of ``n_anchors`` anchor classes appears in **every** task
    (keeping the same label id, so the model must retain them), while the rest
    of each task is fresh.  Every task is exactly ``init_cls`` classes:
    ``n_anchors`` anchors plus ``init_cls - n_anchors`` fresh categories.  The
    anchors' images are split disjointly across tasks, so each re-exposure
    shows new instances — the "drift" of context around a stable concept.

    Anchor Retention is read at the final task as the accuracy on the anchor
    ids.

    Args:
        total_classes: Total number of real dataset classes (an upper bound;
            this scenario uses ``n_anchors + (init_cls - n_anchors) * nb_tasks``
            of them).
        init_cls: Task size (anchors + fresh) held constant across the stream.
        increment: Used only to derive the number of tasks (kept equal to the
            other scenarios for comparability).
        n_anchors: Number of always-present anchor classes.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    nb_tasks = len(compute_increments(total_classes, init_cls, increment))
    if nb_tasks < 2:
        raise ValueError("Need at least 2 tasks for cumulative_drift.")
    if not (1 <= n_anchors < init_cls):
        raise ValueError(
            f"n_anchors={n_anchors} must be in [1, init_cls={init_cls})."
        )

    anchors = list(range(n_anchors))
    fresh_per_task = init_cls - n_anchors

    task_lists: List[List[int]] = [list(range(init_cls))]  # task 0: anchors + fresh
    cursor = init_cls
    for _ in range(1, nb_tasks):
        fresh = list(range(cursor, cursor + fresh_per_task))
        cursor += fresh_per_task
        task_lists.append(anchors + fresh)

    if cursor > total_classes:
        raise ValueError(
            f"cumulative_drift needs {cursor} classes but only {total_classes} "
            f"are available. Reduce nb_tasks or n_anchors."
        )

    # Link task 0 to every later task so the image assigner treats each anchor
    # as shared across all tasks and splits its images disjointly between them.
    pairs = [
        OverlapPair(tasks=(0, t), shared_classes=anchors)
        for t in range(1, nb_tasks)
    ]

    spec = OverlapSpec(
        mode="cumulative_drift",
        pairs=pairs,
        task_class_lists=task_lists,
        image_split=ImageSplit(strategy="disjoint"),
        total_classes_override=cursor,
        seed=seed,
    )
    spec.validate()
    return spec


def build_stream_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    n_revisit_classes: int = 3,
    placement: str = "random",
    min_gap: int = 2,
    image_strategy: str = "disjoint",
    stream_seed: int = 42,
    shuffle_seed: int = 1993,
    dataset: str = "cifar100",
    data_root: str = "./data",
):
    """Return a StreamSpec for the cumulative_drift pattern."""
    from clover.core.stream_spec import RevisitSpec, StreamSpec

    return StreamSpec(
        dataset=dataset,
        init_cls=init_cls,
        increment=increment,
        revisits=[
            RevisitSpec(
                classes=n_revisit_classes,
                times=1,
                placement=placement,
                min_gap=min_gap,
            )
        ],
        image_strategy=image_strategy,
        shuffle_seed=shuffle_seed,
        stream_seed=stream_seed,
        data_root=data_root,
    )
