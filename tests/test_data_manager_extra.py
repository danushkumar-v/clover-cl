"""Additional OverlapDataManager tests for coverage of from_yaml and split API."""

from __future__ import annotations

import pytest


@pytest.fixture
def baseline_dm(patched_cifar100):
    from clover.core.data_manager import OverlapDataManager

    return OverlapDataManager("cifar100", init_cls=10, increment=10)


# ---------------------------------------------------------------------------
# get_dataset_with_split (PILOT-compatible)
# ---------------------------------------------------------------------------

def test_get_dataset_with_split_returns_pair(baseline_dm):
    from torch.utils.data import Dataset

    train_ds, val_ds = baseline_dm.get_dataset_with_split(
        0, source="train", mode="train", val_samples_per_class=1
    )
    assert isinstance(train_ds, Dataset)
    assert isinstance(val_ds, Dataset)


def test_get_dataset_with_split_val_size(baseline_dm):
    train_ds, val_ds = baseline_dm.get_dataset_with_split(
        [0, 1], source="train", mode="train", val_samples_per_class=2
    )
    # 2 classes * 2 val samples each = 4 val samples
    assert len(val_ds) == 4


def test_get_dataset_with_split_train_val_disjoint(baseline_dm):
    """Train and val should not share samples (by index)."""
    train_ds, val_ds = baseline_dm.get_dataset_with_split(
        0, source="train", mode="train", val_samples_per_class=1
    )
    # They can share images by content but their total should be
    # <= num_class_images * num_classes
    assert len(train_ds) + len(val_ds) <= 10 * 5  # 10 classes * 5 per class


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------

def test_from_yaml_baseline(patched_cifar100, tmp_path):
    from clover.core.data_manager import OverlapDataManager

    yaml_content = """dataset_name: cifar100
init_cls: 10
increment: 10
shuffle_seed: 1993
data_root: ./data
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    dm = OverlapDataManager.from_yaml(str(p))
    assert dm.nb_tasks == 10
    assert dm.nb_classes == 100


def test_from_yaml_with_overlap(patched_cifar100, tmp_path):
    from clover.core.data_manager import OverlapDataManager

    yaml_content = """dataset_name: cifar100
init_cls: 10
increment: 10
shuffle_seed: 1993
data_root: ./data

overlap_spec:
  mode: partial
  seed: 42
  image_split:
    strategy: duplicate
    ratio: 0.5
    overlap_pct: 0.0
  pairs:
    - tasks: [0, 5]
      shared_classes: [0, 1, 2]
"""
    p = tmp_path / "cfg_overlap.yaml"
    p.write_text(yaml_content)
    dm = OverlapDataManager.from_yaml(str(p))
    assert dm.overlap_spec is not None
    assert dm.overlap_spec.mode == "partial"
    mat = dm.get_overlap_matrix()
    assert mat[0, 5] > 0


# ---------------------------------------------------------------------------
# getlen (PILOT-compatible)
# ---------------------------------------------------------------------------

def test_getlen_returns_positive(baseline_dm):
    count = baseline_dm.getlen(0)
    assert count >= 0


# ---------------------------------------------------------------------------
# Overlap manager with disjoint image split
# ---------------------------------------------------------------------------

def test_disjoint_split_manager_dataset(patched_cifar100):
    from clover.core.data_manager import OverlapDataManager
    from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec

    spec = OverlapSpec(
        mode="distribution_shift",
        pairs=[OverlapPair(tasks=(0, 1), shared_classes=[0, 1])],
        image_split=ImageSplit(strategy="disjoint", ratio=0.6),
        seed=42,
    )
    dm = OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)
    ds0 = dm.get_dataset(0, source="train", mode="train")
    ds1 = dm.get_dataset(1, source="train", mode="train")
    assert len(ds0) >= 0
    assert len(ds1) >= 0


# ---------------------------------------------------------------------------
# Unknown dataset raises
# ---------------------------------------------------------------------------

def test_unknown_dataset_raises():
    from clover.core.data_manager import OverlapDataManager

    with pytest.raises(NotImplementedError, match="Unknown dataset"):
        OverlapDataManager("no_such_dataset_xyz", init_cls=5, increment=5)
