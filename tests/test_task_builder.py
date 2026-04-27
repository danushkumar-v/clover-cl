"""Tests for clover/core/task_builder.py."""

from __future__ import annotations

import numpy as np
import pytest

from clover.core.overlap_spec import OverlapPair, OverlapSpec
from clover.core.task_builder import build_tasks, compute_increments


def _identity_order(n):
    return list(range(n))


# ---------------------------------------------------------------------------
# Baseline: mode="none" must match PILOT slicing exactly
# ---------------------------------------------------------------------------

def test_none_mode_matches_pilot_slicing():
    total = 10
    init_cls = 3
    increment = 2
    order = _identity_order(total)
    spec = OverlapSpec(mode="none")
    tasks = build_tasks(total, init_cls, increment, order, spec)

    # PILOT: [3, 2, 2, 2, 1]
    assert tasks[0] == [0, 1, 2]
    assert tasks[1] == [3, 4]
    assert tasks[2] == [5, 6]
    assert tasks[3] == [7, 8]
    assert tasks[4] == [9]


def test_none_mode_cifar100_style():
    """init_cls=10, increment=10, total=100 → 10 tasks of 10."""
    order = _identity_order(100)
    spec = OverlapSpec(mode="none")
    tasks = build_tasks(100, 10, 10, order, spec)

    assert len(tasks) == 10
    for i, t in enumerate(tasks):
        assert t == list(range(i * 10, (i + 1) * 10))


def test_none_mode_with_remainder():
    """total=15, init_cls=5, increment=4 → tasks of 5, 4, 4, 2."""
    order = _identity_order(15)
    spec = OverlapSpec(mode="none")
    tasks = build_tasks(15, 5, 4, order, spec)

    assert tasks[0] == [0, 1, 2, 3, 4]
    assert tasks[1] == [5, 6, 7, 8]
    assert tasks[2] == [9, 10, 11, 12]
    assert tasks[3] == [13, 14]


def test_compute_increments_matches_pilot():
    incs = compute_increments(100, 10, 10)
    assert incs == [10] * 10


# ---------------------------------------------------------------------------
# Overlap injection
# ---------------------------------------------------------------------------

def test_exact_replay_classes_injected():
    order = _identity_order(100)
    pair = OverlapPair(tasks=(0, 9), shared_classes=list(range(10)))
    spec = OverlapSpec(mode="exact_replay", pairs=[pair])
    tasks = build_tasks(100, 10, 10, order, spec)

    # Task 9 must contain all of task 0's classes
    for cls in range(10):
        assert cls in tasks[9], f"Class {cls} not in task 9 after exact_replay injection"


def test_partial_injection_adds_to_both_tasks():
    order = _identity_order(100)
    pair = OverlapPair(tasks=(0, 5), shared_classes=[3, 7])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(100, 10, 10, order, spec)

    assert 3 in tasks[0]
    assert 7 in tasks[0]
    assert 3 in tasks[5]
    assert 7 in tasks[5]


def test_injection_out_of_bounds_raises():
    order = _identity_order(20)
    pair = OverlapPair(tasks=(0, 99), shared_classes=[0])
    spec = OverlapSpec(mode="partial", pairs=[pair])

    with pytest.raises(ValueError, match="task index"):
        build_tasks(20, 5, 5, order, spec)


def test_no_duplicate_classes_injected():
    """Classes already in a task should not be duplicated after injection."""
    order = _identity_order(20)
    pair = OverlapPair(tasks=(0, 1), shared_classes=[0])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(20, 5, 5, order, spec)

    assert tasks[0].count(0) == 1


# ---------------------------------------------------------------------------
# preserve_size contract
# ---------------------------------------------------------------------------

def test_preserve_size_true_keeps_non_final_task_sizes():
    """Non-final tasks must not grow when preserve_size=True."""
    order = _identity_order(50)
    # inject classes 0, 1 (from task 0) into task 2
    pair = OverlapPair(tasks=(0, 2), shared_classes=[0, 1])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(50, 10, 10, order, spec, preserve_size=True)

    # total=50, init=10, inc=10 → 5 tasks of [0-9, 10-19, 20-29, 30-39, 40-49]
    # task 0 already has [0,1] — no injection needed, size stays 10
    assert len(tasks[0]) == 10
    # task 2 receives [0,1] → evicts [29,28] → stays at 10
    assert len(tasks[2]) == 10
    assert 0 in tasks[2]
    assert 1 in tasks[2]


def test_preserve_size_displaced_classes_in_final_task():
    """Classes evicted from non-final tasks must appear in the final task."""
    order = _identity_order(50)
    pair = OverlapPair(tasks=(0, 2), shared_classes=[0, 1])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(50, 10, 10, order, spec, preserve_size=True)

    # Evicted from task 2: the two highest non-shared classes (28, 29)
    final = tasks[-1]
    assert 28 in final
    assert 29 in final


def test_preserve_size_false_allows_growth():
    """With preserve_size=False tasks may grow (v0.1 regression guard)."""
    order = _identity_order(50)
    pair = OverlapPair(tasks=(0, 2), shared_classes=[0, 1])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(50, 10, 10, order, spec, preserve_size=False)

    # task 2 grows from 10 → 12 (two new classes injected, none evicted)
    assert len(tasks[2]) == 12
    assert 0 in tasks[2]
    assert 1 in tasks[2]


def test_preserve_size_final_task_grows_for_exact_replay():
    """Injecting into the final task always grows it (no eviction target)."""
    order = _identity_order(100)
    # exact_replay injects task 0's 10 classes into the last task (9)
    pair = OverlapPair(tasks=(0, 9), shared_classes=list(range(10)))
    spec = OverlapSpec(mode="exact_replay", pairs=[pair])
    tasks = build_tasks(100, 10, 10, order, spec, preserve_size=True)

    # Task 9 is the last task — it is allowed to grow
    assert len(tasks[9]) > 10
    for cls in range(10):
        assert cls in tasks[9]


def test_preserve_size_no_information_lost():
    """Every original class must appear in at least one task after injection."""
    order = _identity_order(50)
    pair = OverlapPair(tasks=(0, 2), shared_classes=[0, 1, 2, 3, 4])
    spec = OverlapSpec(mode="partial", pairs=[pair])
    tasks = build_tasks(50, 10, 10, order, spec, preserve_size=True)

    all_classes = set()
    for t in tasks:
        all_classes.update(t)
    assert all_classes == set(range(50))
