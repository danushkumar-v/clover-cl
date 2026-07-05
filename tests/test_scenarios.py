"""Golden-plan tests for the 6 core scenarios (SPEC §8), against a small synthetic dataset."""

from __future__ import annotations

import pytest

from clover.core.planner import resolve
from clover.core.spec import DatasetInfo
from clover.scenarios import get_scenario, list_scenarios

CORE_SCENARIOS = {
    "disjoint_baseline",
    "exact_replay",
    "partial_overlap",
    "long_range_revisit",
    "mid_range_revisit",
    "cumulative_drift",
}


def test_all_core_scenarios_registered():
    assert CORE_SCENARIOS <= set(list_scenarios())


def _fixed_task_sizes(task_class_lists, expected_sizes):
    assert [len(t) for t in task_class_lists] == expected_sizes


def test_disjoint_baseline_has_no_revisits():
    info = DatasetInfo("synthetic", 20)
    spec = get_scenario("disjoint_baseline")(info, init_cls=5, increment=5, seed=42)
    plan = resolve(spec, info)
    assert plan.echo_table == {}
    assert plan.revisit_ids == frozenset()
    _fixed_task_sizes(plan.task_class_lists, [5, 5, 5, 5])
    assert plan.head_size_schedule == [5, 10, 15, 20]


def test_exact_replay_appends_pure_echo_task_with_identical_images():
    info = DatasetInfo("synthetic", 20)
    spec = get_scenario("exact_replay")(info, init_cls=5, increment=5, seed=42)
    plan = resolve(spec, info)

    _fixed_task_sizes(plan.task_class_lists, [5, 5, 5, 5])
    last_task = plan.task_class_lists[-1]
    assert set(last_task) == set(plan.echo_table)  # last task is pure echo
    assert {e.source_id for e in plan.echo_table.values()} == set(plan.task_class_lists[0])
    assert all(e.image_relation == "same" for e in plan.echo_table.values())
    # echo ids are contiguous and distinct from any source id
    assert set(plan.echo_table).isdisjoint(set(plan.task_class_lists[0]))


def test_long_range_revisit_uses_disjoint_images():
    info = DatasetInfo("synthetic", 20)
    spec = get_scenario("long_range_revisit")(info, init_cls=5, increment=5, seed=42)
    plan = resolve(spec, info)
    assert all(e.image_relation == "new" for e in plan.echo_table.values())
    assert {e.source_id for e in plan.echo_table.values()} == set(plan.task_class_lists[0])


def test_mid_range_revisit_echoes_the_anchor_task_not_task0():
    info = DatasetInfo("synthetic", 30)
    spec = get_scenario("mid_range_revisit")(info, init_cls=5, increment=5, seed=42, anchor_task=2)
    plan = resolve(spec, info)
    anchor_classes = set(plan.task_class_lists[2]) - set(plan.echo_table)
    assert {e.source_id for e in plan.echo_table.values()} == anchor_classes
    assert anchor_classes != set(plan.task_class_lists[0])


def test_mid_range_revisit_requires_enough_experiences():
    info = DatasetInfo("synthetic", 20)  # only 4 experiences at init_cls=5/increment=5
    with pytest.raises(ValueError, match="needs at least"):
        get_scenario("mid_range_revisit")(info, init_cls=5, increment=5, seed=42, anchor_task=4)


def test_partial_overlap_mixes_fresh_and_echo_in_fixed_size_last_task():
    info = DatasetInfo("synthetic", 20)
    spec = get_scenario("partial_overlap")(info, init_cls=5, increment=5, seed=42, overlap_fraction=0.5)
    plan = resolve(spec, info)

    _fixed_task_sizes(plan.task_class_lists, [5, 5, 5, 5])
    last_task = set(plan.task_class_lists[-1])
    echoes_in_last = last_task & set(plan.echo_table)
    fresh_in_last = last_task - echoes_in_last
    assert echoes_in_last and fresh_in_last  # genuinely mixed, not all-or-nothing
    assert {e.source_id for e in plan.echo_table.values()} == set(range(len(echoes_in_last)))


def test_cumulative_drift_anchors_keep_same_id_every_task():
    info = DatasetInfo("synthetic", 20)
    spec = get_scenario("cumulative_drift")(info, init_cls=5, increment=5, seed=42, n_anchors=3)
    plan = resolve(spec, info)

    anchors = set(range(3))
    assert plan.echo_table == {}  # same-id, no echo ids at all
    assert plan.revisit_ids == anchors
    for cls_list in plan.task_class_lists[1:]:
        assert anchors <= set(cls_list)
    # fixed task size: increment - n_anchors fresh classes per later task
    _fixed_task_sizes(plan.task_class_lists, [5, 5, 5, 5])


@pytest.mark.parametrize("name", sorted(CORE_SCENARIOS))
def test_scenario_plan_has_contiguous_head_sizing(name):
    info = DatasetInfo("synthetic", 30)
    kwargs = {"anchor_task": 2} if name == "mid_range_revisit" else {}
    spec = get_scenario(name)(info, init_cls=5, increment=5, seed=42, **kwargs)
    plan = resolve(spec, info)  # raises internally if contiguity is violated
    assert plan.head_size_schedule == sorted(plan.head_size_schedule)
    all_ids = {c for cls_list in plan.task_class_lists for c in cls_list}
    assert plan.head_size_schedule[-1] == len(all_ids)
