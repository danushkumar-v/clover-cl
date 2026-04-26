"""CRITICAL: PILOT-equivalence test for OverlapDataManager.

When overlap_spec=None and shuffle_seed=1993, the per-task class lists
produced by CLOVER must EXACTLY match PILOT's DataManager with the same
parameters. This is the drop-in-replacement contract.

Expected class order for CIFAR100 with np.random.seed(1993):
  np.random.seed(1993); np.random.permutation(100).tolist()
"""

from __future__ import annotations

import numpy as np
import pytest

from clover.core.data_manager import OverlapDataManager
from clover.utils.seeding import pilot_class_order


# Pre-computed from PILOT: np.random.seed(1993); np.random.permutation(100).tolist()
# This is the authoritative expected value.
def _expected_class_order() -> list:
    np.random.seed(1993)
    return np.random.permutation(100).tolist()


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
# Class order equivalence
# ---------------------------------------------------------------------------

def test_class_order_matches_pilot(baseline_manager):
    expected = _expected_class_order()
    # Only check the classes that actually exist in the synthetic 10-class dataset
    # (positions 0..9 in the class order are the only valid remapped IDs)
    assert baseline_manager._class_order == expected


def test_pilot_class_order_helper():
    """pilot_class_order() must match legacy np.random.permutation."""
    expected = _expected_class_order()
    got = pilot_class_order(100, 1993)
    assert got == expected


# ---------------------------------------------------------------------------
# Task slicing equivalence
# ---------------------------------------------------------------------------

def test_nb_tasks_matches_pilot(baseline_manager):
    # 100 classes, init_cls=10, increment=10 → 10 tasks
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
    # All labels must be in the remapped class range for task 0 (0..9)
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
