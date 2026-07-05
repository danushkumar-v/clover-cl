"""Mid-range revisit: an interior task's classes return under fresh echo ids."""

from __future__ import annotations

from clover.core.planner import compute_increments, task_class_range
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("mid_range_revisit")
def mid_range_revisit(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    anchor_task: int = 4,
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec that echoes an interior task's classes at the end.

    Args:
        anchor_task: Index (0-based) of the task whose classes are echoed.
            Requires at least ``max(3, anchor_task + 1)`` experiences.
    """
    increments = compute_increments(dataset_info.num_classes, init_cls, increment)
    min_experiences = max(3, anchor_task + 1)
    if len(increments) < min_experiences:
        raise ValueError(
            f"mid_range_revisit needs at least {min_experiences} experiences "
            f"(anchor_task={anchor_task}); dataset only yields {len(increments)}."
        )
    anchor_ids = task_class_range(dataset_info.num_classes, init_cls, increment, anchor_task)

    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(
                classes=anchor_ids,
                placement="end_of_stream",
                label="new",
                images="new",
                min_gap=1,
            )
        ],
    )
