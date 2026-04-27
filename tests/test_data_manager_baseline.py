"""CRITICAL: PILOT-equivalence test for OverlapDataManager.

When overlap_spec=None and shuffle_seed=1993, the per-task class lists
produced by CLOVER must EXACTLY match PILOT's DataManager with the same
parameters. This is the drop-in-replacement contract.

Golden values are stored in tests/fixtures/pilot_class_orders.json and were
captured by scripts/capture_pilot_fixtures.py using PILOT's exact RNG call:
    np.random.seed(seed); np.random.permutation(n_classes).tolist()
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clover.core.data_manager import OverlapDataManager
from clover.utils.seeding import pilot_class_order

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pilot_class_orders.json"


def _load_fixture(key: str) -> dict:
    with open(FIXTURE_PATH) as fh:
        data = json.load(fh)
    assert key in data, f"Fixture key {key!r} not found. Re-run scripts/capture_pilot_fixtures.py."
    return data[key]


@pytest.fixture
def baseline_manager(patched_cifar100):
    """Baseline CIFAR100 manager with 10-class synthetic data."""
    return OverlapDataManager(
        dataset_name="cifar100",
        init_cls=10,
        increment=10,
        overlap_spec=None,
        shuffle_seed=1993,
    )


# ---------------------------------------------------------------------------
# Class order equivalence — fixture-based golden check
# ---------------------------------------------------------------------------

def test_class_order_matches_pilot_fixture(baseline_manager):
    fixture = _load_fixture("cifar100__init10__inc10__seed1993")
    assert baseline_manager._class_order == fixture["class_order"]


def test_class_order_seed1_matches_fixture(patched_cifar100):
    fixture = _load_fixture("cifar100__init10__inc10__seed1")
    dm = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=1)
    assert dm._class_order == fixture["class_order"]


def test_task_class_lists_match_fixture(baseline_manager):
    fixture = _load_fixture("cifar100__init10__inc10__seed1993")
    assert baseline_manager._task_class_lists == fixture["task_class_lists"]


def test_increments_match_fixture(baseline_manager):
    fixture = _load_fixture("cifar100__init10__inc10__seed1993")
    assert baseline_manager._increments == fixture["increments"]


def test_pilot_class_order_helper_matches_fixture():
    """pilot_class_order() must match the golden fixture exactly."""
    fixture = _load_fixture("cifar100__init10__inc10__seed1993")
    assert pilot_class_order(100, 1993) == fixture["class_order"]


def test_different_seeds_different_orders():
    fix1 = _load_fixture("cifar100__init10__inc10__seed1993")
    fix2 = _load_fixture("cifar100__init10__inc10__seed1")
    assert fix1["class_order"] != fix2["class_order"]


# ---------------------------------------------------------------------------
# Task slicing equivalence
# ---------------------------------------------------------------------------

def test_nb_tasks_matches_pilot(baseline_manager):
    assert baseline_manager.nb_tasks == 10


def test_task_class_lists_are_contiguous_remapped_ranges(baseline_manager):
    """Without overlap, each task should be a disjoint contiguous slice."""
    all_seen = set()
    for t in range(baseline_manager.nb_tasks):
        cls_list = baseline_manager.get_task_classes(t)
        assert len(cls_list) == 10, f"Task {t} should have 10 classes"
        as_set = set(cls_list)
        assert as_set.isdisjoint(all_seen), f"Task {t} overlaps with a previous task"
        all_seen |= as_set

    assert all_seen == set(range(100)), "All 100 remapped class IDs must appear exactly once"


def test_task_zero_is_first_10_remapped_classes(baseline_manager):
    task0 = baseline_manager.get_task_classes(0)
    assert task0 == list(range(0, 10))


def test_task_nine_is_last_10_remapped_classes(baseline_manager):
    task9 = baseline_manager.get_task_classes(9)
    assert task9 == list(range(90, 100))


# ---------------------------------------------------------------------------
# Dataset object basics
# ---------------------------------------------------------------------------

def test_get_dataset_returns_dataset(baseline_manager):
    from torch.utils.data import Dataset

    ds = baseline_manager.get_dataset(0, source="train", mode="train")
    assert isinstance(ds, Dataset)


def test_get_dataset_labels_are_remapped(baseline_manager):
    ds = baseline_manager.get_dataset(0, source="train", mode="test")
    labels = set()
    for _, _, label in ds:
        labels.add(int(label))
    assert labels.issubset(set(range(10)))


def test_pilot_api_indices_list_works(baseline_manager):
    """PILOT-compatible calling: get_dataset(list_of_class_ids, source, mode)."""
    ds = baseline_manager.get_dataset([0, 1, 2], "train", "test")
    from torch.utils.data import Dataset

    assert isinstance(ds, Dataset)


def test_get_dataset_ret_data(baseline_manager):
    result = baseline_manager.get_dataset(0, source="train", mode="test", ret_data=True)
    assert len(result) == 3  # data, targets, dataset


# ---------------------------------------------------------------------------
# nb_classes and get_task_size
# ---------------------------------------------------------------------------

def test_nb_classes(baseline_manager):
    assert baseline_manager.nb_classes == 100


def test_get_task_size(baseline_manager):
    for t in range(baseline_manager.nb_tasks):
        assert baseline_manager.get_task_size(t) == 10
