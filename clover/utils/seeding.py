"""Centralised RNG management for CLOVER.

All stochastic decisions (image assignment, near-miss adjacency sampling, etc.)
must go through :func:`get_rng`. The only exception is class-order shuffling,
which uses ``np.random.seed`` + ``np.random.permutation`` to stay byte-identical
with PILOT's legacy behaviour.
"""

from __future__ import annotations

import numpy as np


def get_rng(seed: int) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` seeded with *seed*.

    Args:
        seed: Integer seed. Deterministic for the same value.

    Returns:
        A ``numpy.random.Generator`` instance (PCG64).
    """
    return np.random.default_rng(seed)


def pilot_class_order(n_classes: int, seed: int) -> list[int]:
    """Reproduce PILOT's class-order shuffle exactly.

    PILOT uses the legacy ``np.random.seed`` + ``np.random.permutation`` API.
    This function mirrors that call so CLOVER produces identical class orderings.

    Args:
        n_classes: Total number of classes in the dataset.
        seed: Shuffle seed (PILOT uses 1993 by default).

    Returns:
        A permutation of ``range(n_classes)`` as a list.
    """
    np.random.seed(seed)
    return np.random.permutation(n_classes).tolist()
