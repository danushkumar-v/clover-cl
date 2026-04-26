"""Tests for all 8 scenario build_spec() functions."""

from __future__ import annotations

import pytest

from clover.core.data_manager import OverlapDataManager
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


# ---------------------------------------------------------------------------
# Each scenario must produce a valid OverlapSpec
# ---------------------------------------------------------------------------

def test_exact_replay_validates():
    spec = exact_replay.build_spec(TOTAL, INIT, INC)
    spec.validate()
    assert spec.mode == "exact_replay"
    assert len(spec.pairs) == 1
    assert spec.pairs[0].tasks == (0, 9)


def test_partial_overlap_validates():
    spec = partial_overlap.build_spec(TOTAL, INIT, INC, pair=(0, 5), overlap_fraction=0.5)
    spec.validate()
    assert spec.mode == "partial"
    assert 5 in [len(spec.pairs[0].shared_classes), 10]  # half of init_cls


def test_hierarchical_validates():
    spec = hierarchical.build_spec(TOTAL, INIT, INC, pair=(0, 1))
    spec.validate()
    assert spec.mode == "hierarchical"


def test_hierarchical_with_hierarchy_map():
    # Classes 0-4 are in task 0 (init_cls=5 here), classes 5-9 in task 1.
    # Parent 99 has children [3, 4, 7, 8] spanning both tasks.
    hmap = {99: [3, 4, 7, 8]}
    spec = hierarchical.build_spec(
        50, 5, 5, pair=(0, 1), hierarchy_map=hmap, parent_ids=[99]
    )
    spec.validate()
    shared = set(spec.pairs[0].shared_classes)
    # Children 3, 4 come from task 0 (classes 0-4); 7, 8 from task 1 (5-9)
    assert 3 in shared and 4 in shared
    assert 7 in shared and 8 in shared


def test_hierarchical_hierarchy_map_no_overlap_raises():
    # All children in task 0 only — no children in task 1
    hmap = {99: [0, 1, 2]}  # all in task 0 classes
    with pytest.raises(ValueError, match="no shared classes"):
        hierarchical.build_spec(50, 5, 5, pair=(0, 1), hierarchy_map=hmap, parent_ids=[99])


def test_distribution_shift_validates():
    spec = distribution_shift.build_spec(TOTAL, INIT, INC, pair=(0, 1))
    spec.validate()
    assert spec.mode == "distribution_shift"
    assert spec.image_split.strategy == "disjoint"


def test_near_miss_validates():
    spec = near_miss.build_spec(TOTAL, INIT, INC, pair=(0, 1))
    spec.validate()
    assert spec.mode == "near_miss"
    assert spec.pairs[0].shared_classes == []


def test_long_range_revisit_validates():
    spec = long_range_revisit.build_spec(TOTAL, INIT, INC)
    spec.validate()
    assert spec.mode == "long_range_revisit"
    assert spec.pairs[0].tasks[0] == 0
    assert spec.pairs[0].tasks[1] == 9  # last task


def test_cumulative_drift_validates():
    spec = cumulative_drift.build_spec(TOTAL, INIT, INC, drift_classes=list(range(5)))
    spec.validate()
    assert spec.mode == "cumulative_drift"
    assert len(spec.pairs) >= 1


def test_symmetric_pair_validates():
    spec = symmetric_pair.build_spec(TOTAL, INIT, INC, pair=(0, 1))
    spec.validate()
    assert spec.mode == "symmetric_pair"
    assert len(spec.pairs[0].shared_classes) == 5  # half of init_cls=10


# ---------------------------------------------------------------------------
# Downstream overlap matrix shape/value checks
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_exact_replay(patched_cifar100):
    spec = exact_replay.build_spec(100, 10, 10, revisit_at_task=9)
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


def test_exact_replay_matrix_pair_nonzero(tiny_exact_replay):
    mat = tiny_exact_replay.get_overlap_matrix()
    assert mat[0, 9] > 0
    assert mat[9, 0] > 0
    # Tasks 1..8 should not share with task 0
    for t in range(1, 9):
        assert mat[0, t] == 0


@pytest.fixture
def tiny_partial(patched_cifar100):
    spec = partial_overlap.build_spec(100, 10, 10, pair=(0, 5), overlap_fraction=0.5)
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


def test_partial_overlap_matrix(tiny_partial):
    mat = tiny_partial.get_overlap_matrix()
    assert mat[0, 5] > 0
    assert mat[5, 0] > 0
    # Unrelated pairs should not overlap
    assert mat[1, 3] == 0
