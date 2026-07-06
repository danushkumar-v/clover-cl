"""CIFAR-100 dataset wrapper correctness (SPEC §7) -- no real download,
``torchvision.datasets.CIFAR100`` is monkeypatched (see conftest.py)."""

from __future__ import annotations

from torchvision import transforms

from clover.datasets import get_dataset
from clover.datasets.cifar100 import CIFAR100Dataset


def test_registered_as_cifar100():
    assert get_dataset("cifar100") is CIFAR100Dataset


def test_num_classes(patched_cifar100):
    ds = CIFAR100Dataset(root="./data", train=True)
    assert ds.num_classes == 100


def test_len_matches_synthetic_fixture_size(patched_cifar100):
    ds = CIFAR100Dataset(root="./data", train=True)
    assert len(ds) == 500  # 100 classes * 5 per class


def test_class_to_indices_covers_every_class(patched_cifar100):
    ds = CIFAR100Dataset(root="./data", train=True)
    c2i = ds.get_class_to_indices()
    assert len(c2i) == 100
    for _cls, idxs in c2i.items():
        assert len(idxs) == 5


def test_getitem_returns_image_and_label(patched_cifar100):
    ds = CIFAR100Dataset(root="./data", train=True, transform=transforms.ToTensor())
    image, label = ds[0]
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_transform_properties_are_lists(patched_cifar100):
    ds = CIFAR100Dataset(root="./data", train=True)
    assert isinstance(ds.train_trsf, list) and ds.train_trsf
    assert isinstance(ds.test_trsf, list) and ds.test_trsf
    assert isinstance(ds.common_trsf, list) and ds.common_trsf
