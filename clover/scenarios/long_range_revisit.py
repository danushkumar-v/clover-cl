"""Long-range revisit scenario.

Task 0 and Task T-1 share classes; all intermediate tasks are unrelated.
Tests whether a model can re-use knowledge from the distant past.
"""

from __future__ import annotations

from typing import List, Optional

from clover.core.overlap_spec import EchoSpec, ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments, disjoint_backbone


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a long-range revisit spec (fixed task size).

    Identical structure to :mod:`clover.scenarios.exact_replay` — the full
    disjoint backbone plus one echo task re-presenting task 0's categories —
    except the echo uses ``image_relation="new"``: a disjoint, previously
    unseen half of each category's images.  This tests retention of the
    *concept* across the maximum stream distance rather than memorisation of
    specific pixels.

    Args:
        total_classes: Total number of real dataset classes.
        init_cls: Classes in task 0 (and the echo task).
        increment: Classes per subsequent task.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    backbone = disjoint_backbone(total_classes, init_cls, increment)
    if len(backbone) < 2:
        raise ValueError("Need at least 2 tasks for long_range_revisit.")

    source_ids = list(backbone[0])
    echo_ids = list(range(total_classes, total_classes + len(source_ids)))
    echoes = [
        EchoSpec(new_id=e, source_id=s, image_relation="new")
        for e, s in zip(echo_ids, source_ids)
    ]

    spec = OverlapSpec(
        mode="long_range_revisit",
        task_class_lists=backbone + [echo_ids],
        echoes=echoes,
        total_classes_override=total_classes + len(echo_ids),
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
