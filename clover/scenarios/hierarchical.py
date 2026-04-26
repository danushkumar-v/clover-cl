"""Hierarchical overlap scenario.

Classes sharing a taxonomic parent (e.g. all Finches, all Hawks) are
distributed across two tasks, modelling realistic fine-grained data streams.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.task_builder import compute_increments
from clover.utils.seeding import get_rng


def build_spec(
    total_classes: int,
    init_cls: int,
    increment: int,
    pair: Tuple[int, int] = (0, 1),
    hierarchy_map: Optional[Dict[int, List[int]]] = None,
    parent_ids: Optional[List[int]] = None,
    image_split_strategy: str = "duplicate",
    seed: int = 42,
    **kwargs,
) -> OverlapSpec:
    """Build a hierarchical overlap spec.

    Shared classes are all members of the specified parent categories whose
    children overlap across the two tasks.

    Args:
        total_classes: Total number of dataset classes.
        init_cls: Classes in task 0.
        increment: Classes per subsequent task.
        pair: ``(task_a, task_b)`` indices.
        hierarchy_map: ``{parent_id: [child_class_ids, ...]}``.  If ``None``,
            a simple even-split is used (first half of task_a's classes).
        parent_ids: Which parent categories contribute shared children.  If
            ``None``, all parents with children in *both* tasks are used.
        image_split_strategy: Image assignment strategy.
        seed: RNG seed.

    Returns:
        Validated :class:`~clover.core.overlap_spec.OverlapSpec`.
    """
    increments = compute_increments(total_classes, init_cls, increment)
    nb_tasks = len(increments)

    t_a, t_b = pair
    for t in (t_a, t_b):
        if t < 0 or t >= nb_tasks:
            raise ValueError(f"Task index {t} out of range [0, {nb_tasks - 1}].")

    task_a_start = sum(increments[:t_a])
    task_b_start = sum(increments[:t_b])
    task_a_classes = set(range(task_a_start, task_a_start + increments[t_a]))
    task_b_classes = set(range(task_b_start, task_b_start + increments[t_b]))

    if hierarchy_map is not None:
        shared: List[int] = []
        candidates = parent_ids if parent_ids else list(hierarchy_map.keys())
        for pid in candidates:
            children = set(hierarchy_map[pid])
            in_a = children & task_a_classes
            in_b = children & task_b_classes
            if in_a and in_b:
                shared.extend(sorted(in_a | in_b))
        if not shared:
            raise ValueError(
                "hierarchy_map produced no shared classes between the two tasks. "
                "Check that parent_ids have children spanning both tasks."
            )
    else:
        # Fallback: sample half of task_a's classes using the scenario seed
        task_a_list = sorted(task_a_classes)
        n_shared = max(1, len(task_a_list) // 2)
        rng = get_rng(seed)
        shared = sorted(rng.choice(task_a_list, size=n_shared, replace=False).tolist())

    shared = sorted(set(shared))
    pair_obj = OverlapPair(tasks=(t_a, t_b), shared_classes=shared)
    spec = OverlapSpec(
        mode="hierarchical",
        pairs=[pair_obj],
        image_split=ImageSplit(strategy=image_split_strategy),
        seed=seed,
    )
    spec.validate()
    return spec
