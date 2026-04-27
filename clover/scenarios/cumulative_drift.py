"""Cumulative drift scenario.

The class set grows task by task: {a} → {a,b} → {a,b,c}, so anchor classes
appear in many consecutive tasks.  This simulates a stream where core
concepts accumulate additional related classes over time.
"""

from __future__ import annotations

from typing import List, Optional

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    drift_classes: Optional[List[int]] = None,
    growth_schedule: Optional[List[int]] = None,
    image_split_strategy: str = "partial_duplicate",
    overlap_pct: float = 0.3,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a cumulative drift spec.

    Anchor (drift) classes appear in every task alongside new classes.  The
    overlap between consecutive tasks is declared via :class:`OverlapPair`
    entries linking each task to the next.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        drift_classes: Remapped class IDs that act as anchors.  Defaults to
            the first ``init_cls`` classes (i.e. task 0's full set).
        growth_schedule: Explicit list of task indices where drift classes
            appear.  Defaults to all tasks.
        image_split_strategy: Image assignment strategy for shared classes.
        overlap_pct: For ``partial_duplicate``, fraction of images shared.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    if drift_classes is None:
        drift_classes = list(range(init_cls))

    if growth_schedule is None:
        growth_schedule = list(range(nb_tasks))

    pairs: List[OverlapPair] = []
    valid_tasks = [t for t in growth_schedule if 0 <= t < nb_tasks]
    for i in range(len(valid_tasks) - 1):
        t_a = valid_tasks[i]
        t_b = valid_tasks[i + 1]
        pairs.append(OverlapPair(tasks=(t_a, t_b), shared_classes=list(drift_classes)))

    if not pairs:
        raise ValueError(
            "growth_schedule produced no consecutive task pairs to link. "
            f"nb_tasks={nb_tasks}, growth_schedule={growth_schedule}."
        )

    spec = OverlapSpec(
        mode="cumulative_drift",
        pairs=pairs,
        image_split=ImageSplit(
            strategy=image_split_strategy, overlap_pct=overlap_pct
        ),
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
