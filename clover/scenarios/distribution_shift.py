"""Distribution shift: task 0's classes return once, same label id, on a
different image subset -- the label space doesn't grow, but the visual
distribution behind it does.

Distinct from every core scenario (SPEC §8): unlike ``cumulative_drift``
(same-id, but *every* later task), this fires exactly once
(``placement="end_of_stream"``); unlike ``long_range_revisit``/
``exact_replay`` (fresh echo id, ``label="new"``), the class keeps its
*original* id, so no new label ever appears for it -- only its underlying
image distribution shifts.
"""

from __future__ import annotations

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.scenarios import register_scenario


@register_scenario("distribution_shift")
def distribution_shift(
    dataset_info: DatasetInfo,
    init_cls: int,
    increment: int,
    seed: int = 42,
    n_shifted: int = 3,
    split_ratio: float = 1.0,
    **params: object,
) -> StreamSpec:
    """Return a StreamSpec where ``n_shifted`` of task 0's classes reappear
    once, same-id, at the end of the stream.

    ``split_ratio`` controls how much of the reappearance's images are
    disjoint from task 0's: ``1.0`` (default) means entirely new images
    (``images="new"``); a value in ``(0, 1)`` means a partial image split
    (``images="partial:<split_ratio>"``).
    """
    if not (1 <= n_shifted < init_cls):
        raise ValueError(f"n_shifted={n_shifted} must be in [1, init_cls={init_cls}).")
    if n_shifted >= increment:
        raise ValueError(
            f"n_shifted={n_shifted} must be < increment={increment} so the final "
            "task still has room for fresh classes."
        )
    if not 0.0 < split_ratio <= 1.0:
        raise ValueError(f"split_ratio must be in (0, 1], got {split_ratio}.")
    shifted = list(range(n_shifted))
    images = "new" if split_ratio == 1.0 else f"partial:{split_ratio}"

    return StreamSpec(
        dataset=dataset_info.name,
        init_cls=init_cls,
        increment=increment,
        stream_seed=seed,
        revisits=[
            RevisitSpec(
                classes=shifted,
                placement="end_of_stream",
                label="same",
                images=images,
                min_gap=1,
            )
        ],
    )
