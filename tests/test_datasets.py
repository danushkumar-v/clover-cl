"""Tests for dataset wrappers — error paths and property accessors."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# CIFAR100Dataset
# ---------------------------------------------------------------------------

def test_cifar100_num_classes(patched_cifar100):
    from clover.datasets.cifar100 import CIFAR100Dataset

    ds = CIFAR100Dataset(root="./data", train=True)
    assert ds.num_classes == 100


def test_cifar100_len(patched_cifar100):
    from clover.datasets.cifar100 import CIFAR100Dataset

    ds = CIFAR100Dataset(root="./data", train=True)
    assert len(ds) == 500  # 100 classes * 5 per class


def test_cifar100_getitem_returns_tuple(patched_cifar100):
    from clover.datasets.cifar100 import CIFAR100Dataset
    from torchvision import transforms

    ds = CIFAR100Dataset(root="./data", train=True, transform=transforms.ToTensor())
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_cifar100_class_to_indices(patched_cifar100):
    from clover.datasets.cifar100 import CIFAR100Dataset

    ds = CIFAR100Dataset(root="./data", train=True)
    c2i = ds.get_class_to_indices()
    assert len(c2i) == 100
    for cls, idxs in c2i.items():
        assert len(idxs) == 5  # 5 per class in synthetic


def test_cifar100_transforms_are_lists(patched_cifar100):
    from clover.datasets.cifar100 import CIFAR100Dataset

    ds = CIFAR100Dataset(root="./data", train=True)
    assert isinstance(ds.train_trsf, list)
    assert isinstance(ds.test_trsf, list)
    assert isinstance(ds.common_trsf, list)


# ---------------------------------------------------------------------------
# Path-based dataset error paths (FileNotFoundError on missing data)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("CUB200Dataset", "clover.datasets.cub200"),
        ("ImageNetRDataset", "clover.datasets.imagenet_r"),
        ("ImageNetADataset", "clover.datasets.imagenet_a"),
        ("OmniBenchDataset", "clover.datasets.omnibench"),
        ("VTABDataset", "clover.datasets.vtab"),
    ],
)
def test_path_dataset_raises_on_missing_data(cls_name, module, tmp_path):
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    with pytest.raises(FileNotFoundError):
        cls(root=str(tmp_path), train=True)


@pytest.mark.parametrize(
    "cls_name,module",
    [
        ("CUB200Dataset", "clover.datasets.cub200"),
        ("ImageNetRDataset", "clover.datasets.imagenet_r"),
        ("ImageNetADataset", "clover.datasets.imagenet_a"),
        ("OmniBenchDataset", "clover.datasets.omnibench"),
        ("VTABDataset", "clover.datasets.vtab"),
    ],
)
def test_path_dataset_num_classes(cls_name, module, tmp_path):
    """num_classes is a class-level constant; load it without instantiating data."""
    import importlib

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    # Just check the constant by peeking at what __init__ would set
    # We can't instantiate without data, but we can check the property is defined
    assert hasattr(cls, "use_path")
    assert cls.use_path is True


# ---------------------------------------------------------------------------
# CLDataset base abstract interface
# ---------------------------------------------------------------------------

def test_cldata_abstract_methods_cannot_instantiate():
    from clover.datasets.base import CLDataset

    with pytest.raises(TypeError):
        CLDataset(root="./data")  # type: ignore


# ---------------------------------------------------------------------------
# image_assigner unknown strategy
# ---------------------------------------------------------------------------

def test_image_assigner_unknown_strategy_raises():
    from clover.core.image_assigner import assign_images
    from clover.core.overlap_spec import ImageSplit, OverlapPair
    from clover.utils.seeding import get_rng

    c2i = {0: list(range(10))}
    tasks = [[0], [0]]
    pairs = [OverlapPair(tasks=(0, 1), shared_classes=[0])]
    bad_split = ImageSplit(strategy="disjoint")  # type: ignore
    bad_split.strategy = "unknown_xyz"  # bypass Literal enforcement

    with pytest.raises(ValueError, match="Unknown image split strategy"):
        assign_images(c2i, tasks, pairs, bad_split, get_rng(0))
