"""Cumulative drift: a small anchor set keeps its identity in every later task.

Identity persistence is confirmed by design (SPEC §8): anchors keep their
same label ids with disjoint images per task, and the core loss policy (§5)
is what makes this trainable for every method, including the prompt family.
"""

from __future__ import annotations

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("cumulative_drift")
def cumulative_drift(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    n_anchors: int = 3,
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec where ``n_anchors`` of task 0's classes recur, same-id."""
    if not (1 <= n_anchors < init_cls):
        raise ValueError(f"n_anchors={n_anchors} must be in [1, init_cls={init_cls}).")
    if n_anchors >= increment:
        raise ValueError(
            f"n_anchors={n_anchors} must be < increment={increment} so every "
            "later task still has room for fresh classes."
        )
    anchors = list(range(n_anchors))

    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(
                classes=anchors,
                placement="every_task",
                label="same",
                images="new",
                min_gap=1,
            )
        ],
    )
