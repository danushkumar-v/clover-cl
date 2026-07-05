"""Exact replay: task 0's classes return under fresh echo ids with identical images."""

from __future__ import annotations

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("exact_replay")
def exact_replay(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    image_relation: str = "same",
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec that echoes task 0's classes at the end of the stream.

    ``image_relation="same"`` (the default) reuses the identical images —
    the strongest possible revisit signal.
    """
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
                images=image_relation,
                min_gap=1,
            )
        ],
    )
