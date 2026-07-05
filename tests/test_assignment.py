"""assign_images split correctness (SPEC §4)."""

from __future__ import annotations

import numpy as np

from clover.core.assignment import assign_images
from clover.core.planner import resolve
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec


def _pool(num_classes: int, per_class: int = 100) -> dict[int, list[int]]:
    return {c: list(range(c * per_class, (c + 1) * per_class)) for c in range(num_classes)}


def test_non_revisited_class_gets_its_full_pool():
    spec = StreamSpec(dataset="synthetic", init_cls=5, increment=5)
    info = DatasetInfo("synthetic", 20)
    plan = resolve(spec, info)
    class_to_indices = _pool(20)

    assignment = assign_images(class_to_indices, plan, np.random.default_rng(0))

    # class_to_indices is keyed by *original* dataset class id; positions
    # (training ids) are translated through plan.class_order.
    for t, cls_list in enumerate(plan.task_class_lists):
        for cls in cls_list:
            assert set(assignment[t][cls]) == set(class_to_indices[plan.class_order[cls]])


def test_same_image_relation_duplicates_full_pool():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="same")],
    )
    info = DatasetInfo("synthetic", 20)
    plan = resolve(spec, info)
    class_to_indices = _pool(20)
    assignment = assign_images(class_to_indices, plan, np.random.default_rng(0))

    for echo_id, entry in plan.echo_table.items():
        owning_task = next(t for t, c in enumerate(plan.task_class_lists) if echo_id in c)
        assert set(assignment[owning_task][echo_id]) == set(
            class_to_indices[plan.class_order[entry.source_id]]
        )


def test_new_image_relation_is_disjoint_from_source_appearance():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="new")],
    )
    info = DatasetInfo("synthetic", 20)
    plan = resolve(spec, info)
    class_to_indices = _pool(20)
    assignment = assign_images(class_to_indices, plan, np.random.default_rng(0))

    for echo_id, entry in plan.echo_table.items():
        owning_task = next(t for t, c in enumerate(plan.task_class_lists) if echo_id in c)
        source_task = next(
            t for t, c in enumerate(plan.task_class_lists) if entry.source_id in c and t != owning_task
        )
        echo_images = set(assignment[owning_task][echo_id])
        source_images = set(assignment[source_task][entry.source_id])
        assert echo_images.isdisjoint(source_images)
        assert echo_images | source_images == set(class_to_indices[plan.class_order[entry.source_id]])


def test_partial_overlap_shares_a_core_between_new_and_source():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[
            RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="partial:0.5")
        ],
    )
    info = DatasetInfo("synthetic", 20)
    plan = resolve(spec, info)
    class_to_indices = _pool(20)
    assignment = assign_images(class_to_indices, plan, np.random.default_rng(0))

    for echo_id, entry in plan.echo_table.items():
        owning_task = next(t for t, c in enumerate(plan.task_class_lists) if echo_id in c)
        source_task = next(
            t for t, c in enumerate(plan.task_class_lists) if entry.source_id in c and t != owning_task
        )
        echo_images = set(assignment[owning_task][echo_id])
        source_images = set(assignment[source_task][entry.source_id])
        shared = echo_images & source_images
        assert shared  # a core is shared under partial overlap
        assert shared != echo_images  # but not fully duplicated
