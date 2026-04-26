"""Tests for clover/core/image_assigner.py."""

from __future__ import annotations

import numpy as np
import pytest

from clover.core.image_assigner import assign_images
from clover.core.overlap_spec import ImageSplit, OverlapPair
from clover.utils.seeding import get_rng


def _build_c2i(n_classes: int = 5, n_per: int = 20) -> dict:
    return {c: list(range(c * n_per, (c + 1) * n_per)) for c in range(n_classes)}


def _make_tasks(n=5):
    return [[i] for i in range(n)]


# ---------------------------------------------------------------------------
# disjoint strategy
# ---------------------------------------------------------------------------

def test_disjoint_no_overlap():
    c2i = _build_c2i(4, 10)
    tasks = [[0, 1], [1, 2, 3]]  # class 1 shared
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[1])]
    split = ImageSplit(strategy="disjoint", ratio=0.5)
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    imgs_0 = set(result[0].get(1, []))
    imgs_1 = set(result[1].get(1, []))
    assert imgs_0.isdisjoint(imgs_1), "Disjoint strategy must produce non-overlapping sets"


def test_disjoint_ratio():
    c2i = {0: list(range(100))}
    tasks = [[0], [0]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[0])]
    split = ImageSplit(strategy="disjoint", ratio=0.3)
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    assert len(result[0][0]) == 30  # floor(0.3 * 100)
    assert len(result[1][0]) == 70


# ---------------------------------------------------------------------------
# duplicate strategy
# ---------------------------------------------------------------------------

def test_duplicate_identical_sets():
    c2i = _build_c2i(3, 15)
    tasks = [[0, 1], [1, 2]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[1])]
    split = ImageSplit(strategy="duplicate")
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    assert sorted(result[0][1]) == sorted(result[1][1]), \
        "Duplicate strategy must give both tasks the same image list"


# ---------------------------------------------------------------------------
# partial_duplicate strategy
# ---------------------------------------------------------------------------

def test_partial_duplicate_overlap_pct():
    n = 100
    c2i = {0: list(range(n))}
    tasks = [[0], [0]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[0])]
    split = ImageSplit(strategy="partial_duplicate", overlap_pct=0.4)
    result = assign_images(c2i, tasks, pairs, split, get_rng(42))

    shared_core = set(result[0][0]) & set(result[1][0])
    n_shared = int(0.4 * n)
    assert len(shared_core) == n_shared, \
        f"Expected {n_shared} shared images, got {len(shared_core)}"

    unique_0 = set(result[0][0]) - shared_core
    unique_1 = set(result[1][0]) - shared_core
    assert unique_0.isdisjoint(unique_1), "Unique portions must be disjoint"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    c2i = _build_c2i(4, 20)
    tasks = [[0, 1], [1, 2, 3]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[1])]
    split = ImageSplit(strategy="disjoint")

    r1 = assign_images(c2i, tasks, pairs, split, get_rng(7))
    r2 = assign_images(c2i, tasks, pairs, split, get_rng(7))

    for t in range(len(tasks)):
        for cls in tasks[t]:
            assert r1[t].get(cls, []) == r2[t].get(cls, [])


def test_different_seed_different_result():
    c2i = {0: list(range(50))}
    tasks = [[0], [0]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[0])]
    split = ImageSplit(strategy="disjoint", ratio=0.5)

    r1 = assign_images(c2i, tasks, pairs, split, get_rng(1))
    r2 = assign_images(c2i, tasks, pairs, split, get_rng(999))

    # Very unlikely to be identical with different seeds
    assert r1[0][0] != r2[0][0]


# ---------------------------------------------------------------------------
# Non-shared classes
# ---------------------------------------------------------------------------

def test_non_shared_classes_get_all_images():
    c2i = _build_c2i(3, 10)
    tasks = [[0, 1], [1, 2]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[1])]
    split = ImageSplit(strategy="disjoint")
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    # Class 0 is only in task 0 — should get all 10 images
    assert sorted(result[0][0]) == list(range(0, 10))
    # Class 2 is only in task 1 — should get all 10 images
    assert sorted(result[1][2]) == list(range(20, 30))


# ---------------------------------------------------------------------------
# 3-way shared class
# ---------------------------------------------------------------------------

def test_three_way_disjoint():
    c2i = {0: list(range(60))}
    tasks = [[0], [0], [0]]
    pairs = [
        OverlapPair(tasks=(0, 1), shared_classes=[0]),
        OverlapPair(tasks=(1, 2), shared_classes=[0]),
    ]
    split = ImageSplit(strategy="disjoint")
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    all_images = set()
    for t in range(3):
        imgs = set(result[t].get(0, []))
        assert all_images.isdisjoint(imgs), f"Task {t} overlaps with previous tasks"
        all_images |= imgs


def test_three_way_duplicate():
    c2i = {0: list(range(20))}
    tasks = [[0], [0], [0]]
    pairs = [
        OverlapPair(tasks=(0, 1), shared_classes=[0]),
        OverlapPair(tasks=(1, 2), shared_classes=[0]),
    ]
    split = ImageSplit(strategy="duplicate")
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    for t in range(3):
        assert sorted(result[t][0]) == sorted(result[0][0])


def test_three_way_partial_duplicate():
    n = 90
    c2i = {0: list(range(n))}
    tasks = [[0], [0], [0]]
    pairs = [
        OverlapPair(tasks=(0, 1), shared_classes=[0]),
        OverlapPair(tasks=(1, 2), shared_classes=[0]),
    ]
    split = ImageSplit(strategy="partial_duplicate", overlap_pct=0.3)
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    # Each task should have some images
    for t in range(3):
        assert len(result[t][0]) > 0

    # All tasks share the same core
    core = set(result[0][0]) & set(result[1][0]) & set(result[2][0])
    n_shared_core = int(0.3 * n)
    assert len(core) == n_shared_core


def test_empty_pool_shared_class():
    """A shared class with no images should produce empty lists for all tasks."""
    c2i = {0: [], 1: list(range(10))}  # class 0 has no images
    tasks = [[0, 1], [0, 1]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[0])]
    split = ImageSplit(strategy="duplicate")
    result = assign_images(c2i, tasks, pairs, split, get_rng(0))

    assert result[0][0] == []
    assert result[1][0] == []
