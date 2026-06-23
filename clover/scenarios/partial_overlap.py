"""Partial overlap scenario: a fraction of Task N's classes appear in Task M."""

from __future__ import annotations

from clover.core.overlap_spec import EchoSpec, ImageSplit, OverlapPair, OverlapSpec
from clover.core.stream_spec import RevisitSpec, StreamSpec
from clover.core.task_builder import compute_increments, disjoint_backbone
from clover.utils.seeding import get_rng


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    overlap_fraction: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a fixed-size partial-overlap spec.

    The final task is a **mixed** task of constant size: half its classes are
    brand-new (fresh real categories) and half are echoes of task 0's
    categories returning with new (disjoint) images.  Measuring accuracy on the
    two halves at that single task yields Repetition Gain — the returning vs.
    fresh accuracy gap — with everything else held equal.

    The backbone covers the first ``nb_tasks - 1`` tasks; the last backbone
    task's class budget is rebuilt as the mixed task, so task size never
    changes.

    Args:
        total_classes: Total number of real dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task (also the mixed task's size).
        overlap_fraction: Fraction of the mixed task that is *returning*
            (echoed).  ``0.5`` splits the task evenly into returning and fresh.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    if not 0.0 < overlap_fraction < 1.0:
        raise ValueError(f"overlap_fraction must be in (0, 1), got {overlap_fraction}.")

    backbone = disjoint_backbone(total_classes, init_cls, increment)
    if len(backbone) < 2:
        raise ValueError("Need at least 2 tasks for partial_overlap.")

    last = backbone[-1]
    n_echo = max(1, int(round(len(last) * overlap_fraction)))
    n_fresh = len(last) - n_echo
    if n_fresh < 1:
        raise ValueError(
            "overlap_fraction leaves no fresh classes in the mixed task."
        )

    fresh_ids = last[:n_fresh]
    # Echo ids follow the last fresh real id contiguously, so the head stays
    # gap-free; they reuse the label slots the dropped real classes vacated.
    echo_start = fresh_ids[-1] + 1
    echo_ids = list(range(echo_start, echo_start + n_echo))
    source_ids = list(backbone[0])[:n_echo]
    echoes = [
        EchoSpec(new_id=e, source_id=s, image_relation="new")
        for e, s in zip(echo_ids, source_ids)
    ]

    mixed_task = fresh_ids + echo_ids
    task_lists = backbone[:-1] + [mixed_task]

    spec = OverlapSpec(
        mode="partial_overlap_50",
        task_class_lists=task_lists,
        echoes=echoes,
        total_classes_override=echo_ids[-1] + 1,
        seed=seed,
    )
    spec.validate()
    return spec


def build_stream_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    n_revisit_classes: int = 5,
    placement: str = "random",
    min_gap: int = 3,
    image_strategy: str = "disjoint",
    stream_seed: int = 42,
    shuffle_seed: int = 1993,
    dataset: str = "cifar100",
    data_root: str = "./data",
) -> StreamSpec:
    """Return a :class:`~clover.core.stream_spec.StreamSpec` for partial overlap.

    The framework will randomly select ``n_revisit_classes`` classes and place
    each one back into an experience according to ``placement``.  Use
    ``stream_seed`` to vary the concrete realisation while keeping the same
    statistical structure.

    Args:
        total_classes: Total number of dataset classes (unused here; kept for
            API symmetry with ``build_spec``).
        init_cls: Classes in the first experience.
        increment: Classes added per subsequent experience.
        n_revisit_classes: How many classes revisit once.
        placement: Revisit placement strategy.
        min_gap: Minimum experiences between first appearance and revisit.
        image_strategy: Image-overlap strategy for shared classes.
        stream_seed: Seed for revisit placement and class selection.
        shuffle_seed: Seed for PILOT-compatible class-order shuffle.
        dataset: Dataset name.
        data_root: Root directory for dataset files.

    Returns:
        A :class:`~clover.core.stream_spec.StreamSpec`.
    """
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
