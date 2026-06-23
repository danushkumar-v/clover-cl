"""Mid-range revisit scenario.

A control for :mod:`clover.scenarios.long_range_revisit`: instead of echoing
task 0 (the very first task), it echoes a *mid-stream* task.  This isolates
whether a method's revisit behaviour is special to the first task or holds for
an arbitrary point in the stream.

Structure mirrors long_range_revisit exactly — the full disjoint backbone plus
one echo task re-presenting the source task's categories with new (disjoint)
images — only the source task differs.
"""

from __future__ import annotations

from clover.core.overlap_spec import EchoSpec, OverlapSpec
from clover.core.task_builder import disjoint_backbone


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    anchor_task: int = 4,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a mid-range revisit spec (fixed task size).

    Args:
        total_classes: Total number of real dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        anchor_task: Index of the mid-stream task whose categories are echoed
            at the end of the stream.  Must be in ``[1, nb_tasks - 1)``.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    backbone = disjoint_backbone(total_classes, init_cls, increment)
    nb_tasks = len(backbone)
    if nb_tasks < 3:
        raise ValueError("Need at least 3 tasks for mid_range_revisit.")
    if not (1 <= anchor_task < nb_tasks):
        raise ValueError(
            f"anchor_task={anchor_task} out of range [1, {nb_tasks - 1}]."
        )

    source_ids = list(backbone[anchor_task])
    echo_ids = list(range(total_classes, total_classes + len(source_ids)))
    echoes = [
        EchoSpec(new_id=e, source_id=s, image_relation="new")
        for e, s in zip(echo_ids, source_ids)
    ]

    spec = OverlapSpec(
        mode="mid_range_revisit",
        task_class_lists=backbone + [echo_ids],
        echoes=echoes,
        total_classes_override=total_classes + len(echo_ids),
        seed=seed,
    )
    spec.validate()
    return spec
