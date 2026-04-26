"""Shared pytest fixtures and synthetic dataset helpers."""

from __future__ import annotations

import types
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic CIFAR100 factory
# ---------------------------------------------------------------------------

N_CLASSES = 100   # must match real CIFAR-100 so remapping works end-to-end
N_PER_CLASS = 5   # 500 total — tiny but covers all 100 remapped class IDs
IMG_H, IMG_W = 32, 32


def make_synthetic_cifar(n_classes: int = N_CLASSES, n_per_class: int = N_PER_CLASS):
    """Return (data, targets) numpy arrays for a tiny synthetic dataset."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=(n_classes * n_per_class, IMG_H, IMG_W, 3), dtype=np.uint8)
    targets = np.repeat(np.arange(n_classes), n_per_class)
    return data, targets


class _SyntheticCIFAR100:
    """Minimal stand-in for torchvision CIFAR100."""

    def __init__(self, root, train=True, download=False):
        data, targets = make_synthetic_cifar(N_CLASSES, N_PER_CLASS)
        self.data = data
        self.targets = targets.tolist()


@pytest.fixture
def tiny_cifar(monkeypatch):
    """Monkeypatch torchvision CIFAR100 with a 10-class, 200-sample synthetic dataset.

    All tests that import CIFAR100Dataset will use this fixture instead of
    downloading the real dataset.
    """
    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, "CIFAR100", _SyntheticCIFAR100)
    return N_CLASSES, N_PER_CLASS


# ---------------------------------------------------------------------------
# Helpers for building minimal managers without downloading data
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_cifar100(monkeypatch):
    """Patch torchvision CIFAR100 used inside CLOVER datasets."""
    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, "CIFAR100", _SyntheticCIFAR100)
