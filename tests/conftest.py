"""Shared pytest fixtures (SPEC §7): avoid any real dataset download."""

from __future__ import annotations

import numpy as np
import pytest

_N_CLASSES = 100  # matches real CIFAR-100 so num_classes/remapping logic is exercised faithfully
_N_PER_CLASS = 5  # tiny -- 500 total samples is enough to cover every class id
_IMG_H, _IMG_W = 32, 32


class _SyntheticCIFAR100:
    """Minimal stand-in for ``torchvision.datasets.CIFAR100`` -- same
    ``.data``/``.targets`` shape, no download, no real image content."""

    def __init__(self, root, train=True, download=False):
        rng = np.random.default_rng(0)
        self.data = rng.integers(
            0, 256, size=(_N_CLASSES * _N_PER_CLASS, _IMG_H, _IMG_W, 3), dtype=np.uint8
        )
        self.targets = np.repeat(np.arange(_N_CLASSES), _N_PER_CLASS).tolist()


@pytest.fixture
def patched_cifar100(monkeypatch):
    """Monkeypatch ``torchvision.datasets.CIFAR100`` so
    ``CIFAR100Dataset`` never attempts a real download during tests."""
    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, "CIFAR100", _SyntheticCIFAR100)
    return _N_CLASSES, _N_PER_CLASS
