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
    # Echo-based: one echo task clones task 0 with the SAME images.
    assert len(spec.echoes) == INIT
    assert all(e.image_relation == "same" for e in spec.echoes)
    assert {e.source_id for e in spec.echoes} == set(range(INIT))
    assert len(spec.task_class_lists) == 11  # backbone (10 tasks) + echo task


def test_partial_overlap_validates():
    spec = partial_overlap.build_spec(TOTAL, INIT, INC, overlap_fraction=0.5)
    spec.validate()
    assert spec.mode == "partial_overlap_50"
    # Half the mixed task is returning (echoes with new images), half fresh.
    assert len(spec.echoes) == 5
    assert all(e.image_relation == "new" for e in spec.echoes)


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
    assert len(spec.echoes) == INIT
    assert all(e.image_relation == "new" for e in spec.echoes)
    assert {e.source_id for e in spec.echoes} == set(range(INIT))


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
# Seeded sampling: different seeds → different shared classes; same seed → same
# ---------------------------------------------------------------------------

def test_partial_overlap_echo_sources_deterministic():
    # Returning categories are fixed (task 0's first classes); only the image
    # sampling varies with the seed, so the echo sources match across seeds.
    spec1 = partial_overlap.build_spec(TOTAL, INIT, INC, overlap_fraction=0.5, seed=1)
    spec2 = partial_overlap.build_spec(TOTAL, INIT, INC, overlap_fraction=0.5, seed=2)
    assert sorted(e.source_id for e in spec1.echoes) == \
           sorted(e.source_id for e in spec2.echoes)


def test_partial_overlap_same_seed_reproducible():
    spec1 = partial_overlap.build_spec(TOTAL, INIT, INC, overlap_fraction=0.5, seed=7)
    spec2 = partial_overlap.build_spec(TOTAL, INIT, INC, overlap_fraction=0.5, seed=7)
    assert spec1.task_class_lists == spec2.task_class_lists
    assert [(e.new_id, e.source_id) for e in spec1.echoes] == \
           [(e.new_id, e.source_id) for e in spec2.echoes]


def test_symmetric_pair_different_seeds_different_shared():
    spec1 = symmetric_pair.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=1)
    spec2 = symmetric_pair.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=2)
    assert spec1.pairs[0].shared_classes != spec2.pairs[0].shared_classes


def test_symmetric_pair_same_seed_reproducible():
    spec1 = symmetric_pair.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=99)
    spec2 = symmetric_pair.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=99)
    assert spec1.pairs[0].shared_classes == spec2.pairs[0].shared_classes


def test_hierarchical_fallback_different_seeds_different_shared():
    # No hierarchy_map → uses fallback seeded sampling
    spec1 = hierarchical.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=1)
    spec2 = hierarchical.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=2)
    assert spec1.pairs[0].shared_classes != spec2.pairs[0].shared_classes


def test_hierarchical_fallback_same_seed_reproducible():
    spec1 = hierarchical.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=5)
    spec2 = hierarchical.build_spec(TOTAL, INIT, INC, pair=(0, 1), seed=5)
    assert spec1.pairs[0].shared_classes == spec2.pairs[0].shared_classes


# ---------------------------------------------------------------------------
# Downstream overlap matrix shape/value checks
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_exact_replay(patched_cifar100):
    spec = exact_replay.build_spec(100, 10, 10)
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


def test_exact_replay_echo_task_is_clone(tiny_exact_replay):
    dm = tiny_exact_replay
    # 10-task disjoint backbone plus one echo task (fresh ids, same images).
    assert dm.nb_tasks == 11
    em = dm.get_echo_map()
    assert len(em) == 10
    assert all(rel == "same" for (_s, rel) in em.values())
    # Echo ids are disjoint from task 0's ids (a fresh head block).
    assert set(dm.get_task_classes(10)).isdisjoint(set(dm.get_task_classes(0)))


@pytest.fixture
def tiny_partial(patched_cifar100):
    spec = partial_overlap.build_spec(100, 10, 10, overlap_fraction=0.5)
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


def test_partial_overlap_mixed_task(tiny_partial):
    dm = tiny_partial
    em = dm.get_echo_map()
    assert len(em) == 5
    # Echoes live in the final mixed task alongside fresh classes.
    mixed = set(dm.get_task_classes(dm.nb_tasks - 1))
    assert set(em).issubset(mixed)
    assert len(mixed) == 10
