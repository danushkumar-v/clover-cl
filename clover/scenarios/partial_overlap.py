"""Partial overlap scenario: a fraction of Task N's classes appear in Task M."""

from __future__ import annotations

from typing import Tuple

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.stream_spec import RevisitSpec, StreamSpec
from clover.core.task_builder import compute_increments
from clover.utils.seeding import get_rng


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    pair: Tuple[int, int] = (0, 5),
    overlap_fraction: float = 0.5,
    image_split_strategy: str = "duplicate",
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build an :class:`~clover.core.overlap_spec.OverlapSpec` for partial overlap.

    A fraction ``overlap_fraction`` of the smaller task's classes are shared
    between the two tasks.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Number of classes in task 0.
        increment: Classes added per subsequent task.
        pair: ``(task_a, task_b)`` — the two tasks that share classes.
        overlap_fraction: Fraction of task_a's classes that reappear in task_b.
        image_split_strategy: Image assignment strategy for shared classes.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    if not 0.0 < overlap_fraction <= 1.0:
        raise ValueError(f"overlap_fraction must be in (0, 1], got {overlap_fraction}.")

    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    t_a, t_b = pair
    for t in (t_a, t_b):
        if t < 0 or t >= nb_tasks:
            raise ValueError(f"Task index {t} out of range [0, {nb_tasks - 1}].")

    task_a_start = sum(increments[:t_a])
    task_a_classes = list(range(task_a_start, task_a_start + increments[t_a]))
    n_shared = max(1, int(len(task_a_classes) * overlap_fraction))
    rng = get_rng(seed)
    shared = sorted(rng.choice(task_a_classes, size=n_shared, replace=False).tolist())

    pair_obj = OverlapPair(tasks=(t_a, t_b), shared_classes=shared)
    spec = OverlapSpec(
        mode="partial",
        pairs=[pair_obj],
        image_split=ImageSplit(strategy=image_split_strategy),
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
