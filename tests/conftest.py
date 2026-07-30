"""Shared pytest fixtures (SPEC §7): avoid any real dataset download."""

from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
import pytest

from clover.core.experience import Experience, LabelSpaceView

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


def build_experience(
    task_label: int,
    classes: Iterable[int],
    seen_classes: Optional[Set[int]] = None,
    head_size: Optional[int] = None,
    dataset: str = "synthetic",
) -> Experience:
    """A real ``Experience`` for tests that drive a method's lifecycle
    directly instead of through the Trainer.

    Real, not a stand-in: ``LabelSpaceView`` is where every method's
    revisit-safe masking comes from, so a fake one could let a method pass
    a test with logic the real stream would reject. Membership in
    *seen_classes* is what makes a class a revisit -- never index
    arithmetic.
    """
    here = sorted(set(classes))
    seen = set(seen_classes or ())
    revisiting = sorted(set(here) & seen)
    first = sorted(set(here) - seen)
    size = head_size if head_size is not None else len(set(here) | seen)
    return Experience(
        task_label=task_label,
        benchmark_name="lifecycle-test",
        dataset=dataset,
        classes_in_this_experience=here,
        classes_seen_so_far=sorted(set(here) | seen),
        classes_in_future=[],
        total_classes_in_stream=size,
        revisiting_classes=revisiting,
        first_appearance_of=first,
        overlap_with_previous={},
        echo_map={},
        label_space=LabelSpaceView(
            new_classes=frozenset(first),
            seen_classes=frozenset(seen),
            revisiting_classes=frozenset(revisiting),
            head_size=size,
        ),
        image_indices={c: [] for c in here},
        n_samples=len(here),
    )


@pytest.fixture
def make_experience():
    """Factory fixture wrapping :func:`build_experience`."""
    return build_experience
