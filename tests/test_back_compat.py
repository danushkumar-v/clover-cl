"""Regression guard: v0.1 OverlapDataManager API must remain unchanged."""

from __future__ import annotations

import pytest

from clover import OverlapDataManager
from clover.scenarios import (
    cumulative_drift,
    distribution_shift,
    exact_replay,
    hierarchical,
    long_range_revisit,
    near_miss,
    partial_overlap,
    symmetric_pair,
)

TOTAL = 100
INIT = 10
INC = 10


@pytest.fixture
def baseline(patched_cifar100):
    return OverlapDataManager("cifar100", init_cls=INIT, increment=INC)


def test_baseline_nb_tasks(baseline):
    assert baseline.nb_tasks == 10


def test_baseline_nb_classes(baseline):
    assert baseline.nb_classes == 100


def test_baseline_get_task_size(baseline):
    for t in range(baseline.nb_tasks):
        assert baseline.get_task_size(t) == INC


def test_baseline_get_dataset(baseline):
    from torch.utils.data import Dataset
    ds = baseline.get_dataset(0, source="train", mode="train")
    assert isinstance(ds, Dataset)


def test_baseline_pilot_class_list_api(baseline):
    ds = baseline.get_dataset([0, 1, 2, 3, 4], "train", "test")
    assert ds is not None


def test_baseline_get_dataset_ret_data(baseline):
    result = baseline.get_dataset(0, source="train", mode="test", ret_data=True)
    assert len(result) == 3


def test_baseline_overlap_matrix(baseline):
    import numpy as np
    mat = baseline.get_overlap_matrix()
    # Disjoint → off-diagonal is zero
    T = baseline.nb_tasks
    for i in range(T):
        for j in range(T):
            if i != j:
                assert mat[i, j] == 0
    assert mat.shape == (T, T)


# ---------------------------------------------------------------------------
# All 8 scenarios still work with the v0.1 API
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    ("exact_replay", lambda: exact_replay.build_spec(TOTAL, INIT, INC)),
    ("partial_overlap", lambda: partial_overlap.build_spec(TOTAL, INIT, INC)),
    ("hierarchical", lambda: hierarchical.build_spec(TOTAL, INIT, INC)),
    ("distribution_shift", lambda: distribution_shift.build_spec(TOTAL, INIT, INC)),
    ("near_miss", lambda: near_miss.build_spec(TOTAL, INIT, INC)),
    ("long_range_revisit", lambda: long_range_revisit.build_spec(TOTAL, INIT, INC)),
    ("cumulative_drift", lambda: cumulative_drift.build_spec(TOTAL, INIT, INC, drift_classes=list(range(5)))),
    ("symmetric_pair", lambda: symmetric_pair.build_spec(TOTAL, INIT, INC)),
])
def scenario_spec(patched_cifar100, request):
    name, build_fn = request.param
    return name, build_fn()


def test_scenario_builds_manager(scenario_spec):
    name, spec = scenario_spec
    dm = OverlapDataManager("cifar100", init_cls=INIT, increment=INC, overlap_spec=spec)
    assert dm.nb_tasks >= 1


def test_scenario_get_dataset_works(scenario_spec):
    name, spec = scenario_spec
    dm = OverlapDataManager("cifar100", init_cls=INIT, increment=INC, overlap_spec=spec)
    ds = dm.get_dataset(0, source="train", mode="train")
    assert ds is not None


def test_scenario_overlap_matrix_diagonal_positive(scenario_spec):
    name, spec = scenario_spec
    dm = OverlapDataManager("cifar100", init_cls=INIT, increment=INC, overlap_spec=spec)
    mat = dm.get_overlap_matrix()
    for t in range(dm.nb_tasks):
        assert mat[t, t] > 0
