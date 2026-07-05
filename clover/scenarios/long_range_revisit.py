"""Long-range revisit: task 0's classes return under fresh echo ids with new images."""

from __future__ import annotations

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("long_range_revisit")
def long_range_revisit(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    **params: object,
) -> StreamSpec:
    """Like ``exact_replay`` but with disjoint ("new") images for the echo."""
    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(
                classes="task0",
                placement="end_of_stream",
                label="new",
                images="new",
                min_gap=1,
            )
        ],
    )
