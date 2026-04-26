"""Near-miss scenario.

Tasks share no classes but contain visually adjacent classes (e.g. wolf vs
husky).  Used to test false-positive rates of similarity gates in GRAFT.
The adjacency metadata is stored in the manifest for downstream analysis;
no class IDs are actually shared between tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    pair: Tuple[int, int] = (0, 1),
    adjacency_map: Optional[Dict[int, List[int]]] = None,
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a near-miss overlap spec.

    Tasks do **not** share class IDs.  The ``adjacency_map`` is recorded in the
    manifest header for downstream analysis only.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        pair: ``(task_a, task_b)`` indices.
        adjacency_map: ``{class_id_in_task_a: [visually_similar_ids_in_task_b]}``.
            Optional; recorded for analysis only.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec` with empty
        ``shared_classes`` (near_miss is exempt from the non-empty requirement).
    """
    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    t_a, t_b = pair
    for t in (t_a, t_b):
        if t < 0 or t >= nb_tasks:
            raise ValueError(f"Task index {t} out of range [0, {nb_tasks - 1}].")

    # No actual class sharing — empty shared_classes
    pair_obj = OverlapPair(tasks=(t_a, t_b), shared_classes=[])
    spec = OverlapSpec(
        mode="near_miss",
        pairs=[pair_obj],
        image_split=ImageSplit(strategy="disjoint"),
        seed=seed,
    )
    spec.validate()
    return spec
