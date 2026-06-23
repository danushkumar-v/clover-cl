"""Exact replay scenario: Task N's classes are identical to Task 0's classes."""

from __future__ import annotations

from clover.core.overlap_spec import EchoSpec, ImageSplit, OverlapPair, OverlapSpec
from clover.core.stream_spec import RevisitSpec, StreamSpec
from clover.core.task_builder import compute_increments, disjoint_backbone


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    seed: int = 42,
    image_relation: str = "same",
    **kwargs,
) -> OverlapSpec:
    """Build an :class:`~clover.core.overlap_spec.OverlapSpec` for exact replay.

    The stream is the full disjoint backbone (every task is exactly
    ``increment`` classes) followed by one **echo task** that re-presents
    task 0's categories under fresh label ids.  With ``image_relation="same"``
    (the default for exact replay) the echo reuses the *identical* images — the
    strongest possible revisit signal.  Task size is constant throughout, so a
    drop at the echo task is attributable to the revisit, not to a larger task.

    Args:
        total_classes: Total number of real dataset classes (the backbone
            covers all of them; echo ids extend the label space above this).
        init_cls: Number of classes in task 0 (and the echo task).
        increment: Classes per subsequent task.
        seed: RNG seed for image assignment.
        image_relation: ``"same"`` (exact replay) or ``"new"`` (disjoint
            images — see :mod:`clover.scenarios.long_range_revisit`).

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    backbone = disjoint_backbone(total_classes, init_cls, increment)
    if len(backbone) < 2:
        raise ValueError("Need at least 2 tasks for exact_replay.")

    source_ids = list(backbone[0])
    echo_ids = list(range(total_classes, total_classes + len(source_ids)))
    echoes = [
        EchoSpec(new_id=e, source_id=s, image_relation=image_relation)
        for e, s in zip(echo_ids, source_ids)
    ]

    spec = OverlapSpec(
        mode="exact_replay",
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
    image_strategy: str = "duplicate",
    stream_seed: int = 42,
    shuffle_seed: int = 1993,
    dataset: str = "cifar100",
    data_root: str = "./data",
) -> StreamSpec:
    """Return a :class:`~clover.core.stream_spec.StreamSpec` for exact replay.

    All classes from the first experience revisit the final experience once.
    """
    return StreamSpec(
        dataset=dataset,
        init_cls=init_cls,
        increment=increment,
        revisits=[
            RevisitSpec(
                classes=list(range(init_cls)),
                times=1,
                placement="end_of_stream",
                min_gap=1,
            )
        ],
        image_strategy=image_strategy,
        shuffle_seed=shuffle_seed,
        stream_seed=stream_seed,
        data_root=data_root,
    )
