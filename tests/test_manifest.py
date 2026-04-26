"""Tests for manifest serialisation / deserialisation."""

from __future__ import annotations

import json
import os

import pytest

from clover.core.data_manager import OverlapDataManager
from clover.utils.manifest import load_manifest, save_manifest


@pytest.fixture
def baseline_dm(patched_cifar100):
    return OverlapDataManager("cifar100", init_cls=10, increment=10)


def test_save_and_load_round_trip(baseline_dm, tmp_path):
    path = str(tmp_path / "manifest.json")
    baseline_dm.save_manifest(path)
    data = load_manifest(path)

    assert "_header" in data
    assert data["_header"]["dataset"] == "cifar100"
    assert data["_header"]["init_cls"] == 10
    assert data["_header"]["increment"] == 10
    assert "train" in data


def test_manifest_task_keys(baseline_dm, tmp_path):
    path = str(tmp_path / "manifest.json")
    baseline_dm.save_manifest(path)
    data = load_manifest(path)

    train_manifest = data["train"]
    assert len(train_manifest) == baseline_dm.nb_tasks


def test_manifest_class_keys_are_strings(baseline_dm, tmp_path):
    """JSON keys must be strings; check that we can round-trip back to ints."""
    path = str(tmp_path / "manifest.json")
    baseline_dm.save_manifest(path)
    data = load_manifest(path)

    for task_key, task_map in data["train"].items():
        for cls_key in task_map:
            int(cls_key)  # must not raise


def test_load_manifest_validates_header(tmp_path):
    bad_path = str(tmp_path / "bad.json")
    with open(bad_path, "w") as fh:
        json.dump({"_header": {"clover_version": "0.1.0"}}, fh)

    with pytest.raises(ValueError, match="missing required key"):
        load_manifest(bad_path)


def test_save_manifest_flat_dict(tmp_path):
    """save_manifest works with a plain dict (no _header)."""
    path = str(tmp_path / "flat.json")
    flat = {"0": {"5": [1, 2, 3]}}
    save_manifest(flat, path)
    loaded = load_manifest(path)
    assert loaded["0"]["5"] == [1, 2, 3]
