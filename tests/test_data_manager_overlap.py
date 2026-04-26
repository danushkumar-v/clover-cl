"""Tests for OverlapDataManager with overlap scenarios."""

from __future__ import annotations

import pytest

from clover.core.data_manager import OverlapDataManager
from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec


@pytest.fixture
def exact_replay_manager(patched_cifar100):
    spec = OverlapSpec(
        mode="exact_replay",
        pairs=[OverlapPair(tasks=(0, 9), shared_classes=list(range(10)))],
        image_split=ImageSplit(strategy="duplicate"),
        seed=42,
    )
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


@pytest.fixture
def partial_manager(patched_cifar100):
    spec = OverlapSpec(
        mode="partial",
        pairs=[OverlapPair(tasks=(0, 5), shared_classes=[3, 7])],
        image_split=ImageSplit(strategy="duplicate"),
        seed=42,
    )
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


# ---------------------------------------------------------------------------
# Task class list tests
# ---------------------------------------------------------------------------

def test_exact_replay_task9_contains_task0_classes(exact_replay_manager):
    task0 = set(exact_replay_manager.get_task_classes(0))
    task9 = set(exact_replay_manager.get_task_classes(9))
    assert task0.issubset(task9), "Task 9 must contain all of task 0's classes"


def test_partial_shared_classes_in_both_tasks(partial_manager):
    task0 = partial_manager.get_task_classes(0)
    task5 = partial_manager.get_task_classes(5)
    assert 3 in task0 and 3 in task5
    assert 7 in task0 and 7 in task5


# ---------------------------------------------------------------------------
# Overlap matrix tests
# ---------------------------------------------------------------------------

def test_overlap_matrix_shape(exact_replay_manager):
    mat = exact_replay_manager.get_overlap_matrix()
    T = exact_replay_manager.nb_tasks
    assert mat.shape == (T, T)


def test_exact_replay_off_diagonal_nonzero(exact_replay_manager):
    mat = exact_replay_manager.get_overlap_matrix()
    assert mat[0, 9] > 0, "Tasks 0 and 9 must share classes in exact_replay"
    assert mat[9, 0] == mat[0, 9], "Overlap matrix must be symmetric"


def test_baseline_matrix_is_diagonal(patched_cifar100):
    dm = OverlapDataManager("cifar100", init_cls=10, increment=10)
    mat = dm.get_overlap_matrix()
    # Off-diagonal entries must be 0 (no class appears in two tasks)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                assert mat[i, j] == 0, f"Baseline matrix has non-zero at ({i},{j})"


# ---------------------------------------------------------------------------
# get_manifest and save_manifest
# ---------------------------------------------------------------------------

def test_get_manifest_structure(exact_replay_manager):
    manifest = exact_replay_manager.get_manifest()
    assert isinstance(manifest, dict)
    for task_key, task_map in manifest.items():
        assert isinstance(task_map, dict)
        for cls_key, idxs in task_map.items():
            assert isinstance(idxs, list)


def test_save_manifest_creates_file(exact_replay_manager, tmp_path):
    path = str(tmp_path / "manifest.json")
    exact_replay_manager.save_manifest(path)
    import json
    import os

    assert os.path.exists(path)
    with open(path) as fh:
        data = json.load(fh)
    assert "_header" in data
    assert data["_header"]["dataset"] == "cifar100"
