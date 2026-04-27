"""Long-range revisit scenario.

Task 0 and Task T-1 share classes; all intermediate tasks are unrelated.
Tests whether a model can re-use knowledge from the distant past.
"""

from __future__ import annotations

from typing import List, Optional

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    shared_classes: Optional[List[int]] = None,
    image_split_strategy: str = "duplicate",
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a long-range revisit spec.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        shared_classes: Explicit list of remapped class IDs shared between task 0
            and the last task.  If ``None``, all of task 0's classes are shared.
        image_split_strategy: Image assignment strategy.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    increments = compute_increments(total_classes, init_cls, increment)
    last_task = len(increments) - 1

    if last_task == 0:
        raise ValueError("Need at least 2 tasks for long_range_revisit.")

    if shared_classes is not None:
        shared = list(shared_classes)
    else:
        shared = list(range(init_cls))

    pair = OverlapPair(tasks=(0, last_task), shared_classes=shared)
    spec = OverlapSpec(
        mode="long_range_revisit",
        pairs=[pair],
        image_split=ImageSplit(strategy=image_split_strategy),
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
    """Return a StreamSpec for the long_range_revisit pattern."""
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
