"""Disjoint baseline: the plain PILOT-equivalent stream, no revisits."""

from __future__ import annotations

from clover.core.spec import DatasetInfo, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("disjoint_baseline")
def disjoint_baseline(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec with no revisits (the disjoint CIL baseline)."""
    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
    )
