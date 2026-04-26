"""Reproducibility tests: identical seeds produce identical manifests."""

from __future__ import annotations

import json

import pytest

from clover.core.data_manager import OverlapDataManager
from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec


@pytest.fixture
def make_manager(patched_cifar100):
    def _factory(seed: int = 42, overlap_seed: int = 42):
        spec = OverlapSpec(
            mode="partial",
            pairs=[OverlapPair(tasks=(0, 5), shared_classes=[0, 1, 2])],
            image_split=ImageSplit(strategy="disjoint"),
            seed=overlap_seed,
        )
        return OverlapDataManager(
            "cifar100",
            init_cls=10,
            increment=10,
            overlap_spec=spec,
            shuffle_seed=seed,
        )

    return _factory


def _manifest_to_json(dm) -> str:
    manifest = dm.get_manifest()
    return json.dumps(manifest, sort_keys=True)


def test_same_seed_identical_manifests(make_manager):
    dm1 = make_manager(seed=1993, overlap_seed=42)
    dm2 = make_manager(seed=1993, overlap_seed=42)
    assert _manifest_to_json(dm1) == _manifest_to_json(dm2)


def test_different_overlap_seed_different_manifest(make_manager):
    dm1 = make_manager(seed=1993, overlap_seed=42)
    dm2 = make_manager(seed=1993, overlap_seed=43)
    # Image assignment should differ with different overlap seeds
    assert _manifest_to_json(dm1) != _manifest_to_json(dm2)


def test_different_shuffle_seed_different_class_order(patched_cifar100):
    dm1 = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=1993)
    dm2 = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=2024)
    assert dm1._class_order != dm2._class_order


def test_baseline_reproducibility(patched_cifar100):
    dm1 = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=1993)
    dm2 = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=1993)
    assert dm1._class_order == dm2._class_order
    for t in range(dm1.nb_tasks):
        assert dm1.get_task_classes(t) == dm2.get_task_classes(t)
