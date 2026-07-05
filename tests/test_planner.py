"""planner.resolve(): seed-1993 byte-equivalence, round-trip, feasibility (SPEC §3.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clover.core.plan import StreamPlan
from clover.core.planner import _pilot_class_order, resolve
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pilot_class_orders.json"


@pytest.fixture(scope="module")
def pilot_fixtures():
    with open(FIXTURE_PATH) as fh:
        return json.load(fh)


@pytest.mark.parametrize(
    "key",
    [
        "cifar100__init10__inc10__seed1993",
        "cifar100__init10__inc10__seed1",
        "cifar100__init20__inc20__seed1993",
        "cub200__init20__inc20__seed1993",
        "imagenet_r__init20__inc20__seed1993",
    ],
)
def test_pilot_class_order_byte_equivalence(pilot_fixtures, key):
    entry = pilot_fixtures[key]
    assert _pilot_class_order(entry["n_classes"], entry["seed"]) == entry["class_order"]


def test_disjoint_plan_matches_fixture_task_class_lists(pilot_fixtures):
    entry = pilot_fixtures["cifar100__init10__inc10__seed1993"]
    spec = StreamSpec(
        dataset="cifar100",
        init_cls=entry["init_cls"],
        increment=entry["increment"],
        shuffle_seed=entry["seed"],
    )
    plan = resolve(spec, DatasetInfo("cifar100", entry["n_classes"]))
    assert plan.task_class_lists == entry["task_class_lists"]
    assert plan.class_order == entry["class_order"]


def test_plan_manifest_round_trip(tmp_path):
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="same")],
    )
    plan = resolve(spec, DatasetInfo("synthetic", 20))

    path = tmp_path / "manifest.json"
    plan.save(str(path))
    restored = StreamPlan.from_manifest(str(path))

    assert restored.task_class_lists == plan.task_class_lists
    assert restored.echo_table == plan.echo_table
    assert restored.revisit_ids == plan.revisit_ids
    assert restored.head_size_schedule == plan.head_size_schedule
    assert restored.class_order == plan.class_order
    assert restored.spec.to_dict() == plan.spec.to_dict()


def test_min_gap_infeasible_raises_actionable_error():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[
            RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="new", min_gap=10)
        ],
    )
    with pytest.raises(ValueError, match="Cannot place 1 revisit\\(s\\) of class 0"):
        resolve(spec, DatasetInfo("synthetic", 20))


def test_class_budget_overflow_raises_actionable_error():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        task_size="fixed",
        revisits=[
            RevisitSpec(
                classes=list(range(5)),
                placement="end_of_stream",
                label="new",
                images="new",
                times=1,
                min_gap=1,
            ),
            RevisitSpec(
                classes=[5, 6, 7],
                placement="end_of_stream",
                label="new",
                images="new",
                times=1,
                min_gap=1,
            ),
        ],
    )
    with pytest.raises(ValueError, match="exceed its class budget"):
        resolve(spec, DatasetInfo("synthetic", 20))


def test_task_size_grow_allows_overflow_beyond_budget():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        task_size="grow",
        revisits=[
            RevisitSpec(classes=list(range(5)), placement="end_of_stream", label="new", images="new")
        ],
    )
    plan = resolve(spec, DatasetInfo("synthetic", 20))
    assert len(plan.task_class_lists[-1]) == 5 + 5  # normal budget + all echoes


def test_out_of_range_revisit_class_id_rejected():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes=[999], placement="end_of_stream", label="new", images="new")],
    )
    with pytest.raises(ValueError, match="out of range"):
        resolve(spec, DatasetInfo("synthetic", 20))


def test_init_cls_exceeding_num_classes_rejected():
    spec = StreamSpec(dataset="synthetic", init_cls=50, increment=5)
    with pytest.raises(ValueError, match="exceeds dataset"):
        resolve(spec, DatasetInfo("synthetic", 20))


@pytest.mark.parametrize("placement", ["random", "spaced", "clustered"])
def test_placement_strategies_respect_min_gap(placement):
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[
            RevisitSpec(
                classes="task0", placement=placement, label="new", images="new", min_gap=1, times=2
            )
        ],
    )
    plan = resolve(spec, DatasetInfo("synthetic", 30))
    # task0's classes each get 2 echo occurrences somewhere after experience 0
    assert len(plan.echo_table) == 5 * 2


def test_random_placement_with_multiple_times_avoids_adjacent_targets():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[
            RevisitSpec(
                classes=[0], placement="random", label="new", images="new", min_gap=2, times=2
            )
        ],
    )
    plan = resolve(spec, DatasetInfo("synthetic", 30))
    owning_tasks = sorted(
        t for t, cls_list in enumerate(plan.task_class_lists) for c in cls_list if c in plan.echo_table
    )
    assert len(owning_tasks) == 2
    assert owning_tasks[1] - owning_tasks[0] >= 2


def test_classes_random_dict_selector_picks_n_distinct_classes():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[
            RevisitSpec(
                classes={"random": 3}, placement="end_of_stream", label="new", images="new", min_gap=1
            )
        ],
    )
    plan = resolve(spec, DatasetInfo("synthetic", 30))
    assert len(plan.echo_table) == 3
    assert len({e.source_id for e in plan.echo_table.values()}) == 3


def test_classes_random_dict_exceeding_num_classes_rejected():
    spec = StreamSpec(
        dataset="synthetic",
        init_cls=5,
        increment=5,
        revisits=[RevisitSpec(classes={"random": 999}, placement="end_of_stream")],
    )
    with pytest.raises(ValueError, match="exceeds dataset size"):
        resolve(spec, DatasetInfo("synthetic", 30))


def test_head_size_schedule_is_cumulative_and_monotonic():
    spec = StreamSpec(dataset="synthetic", init_cls=5, increment=5)
    plan = resolve(spec, DatasetInfo("synthetic", 20))
    assert plan.head_size_schedule == [5, 10, 15, 20]
