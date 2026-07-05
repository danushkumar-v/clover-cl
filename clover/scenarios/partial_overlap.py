"""Partial overlap: the final task mixes fresh classes with echoes of task 0."""

from __future__ import annotations

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("partial_overlap")
def partial_overlap(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    overlap_fraction: float = 0.5,
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec whose final task is part-fresh, part-echo.

    ``overlap_fraction`` of the final task's budget is echoes of task 0's
    classes (fresh label ids, disjoint images); the rest is genuinely new
    classes. Measuring accuracy on the two halves yields Repetition Gain.
    """
    if not 0.0 < overlap_fraction < 1.0:
        raise ValueError(f"overlap_fraction must be in (0, 1), got {overlap_fraction}.")
    n_echo = max(1, round(increment * overlap_fraction))
    if n_echo >= increment:
        raise ValueError(
            f"overlap_fraction={overlap_fraction} leaves no fresh classes in the "
            f"mixed task (n_echo={n_echo} >= increment={increment})."
        )
    source_ids = list(range(n_echo))

    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(
                classes=source_ids,
                placement="end_of_stream",
                label="new",
                images="new",
                min_gap=1,
            )
        ],
    )
