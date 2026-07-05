"""Synthetic dataset metadata smoke test (SPEC §12.2)."""

from __future__ import annotations

import numpy as np
import torch

from clover.datasets import get_dataset
from clover.datasets.synthetic import SyntheticDataset


def test_synthetic_is_registered_and_constructible():
    cls = get_dataset("synthetic")
    assert cls is SyntheticDataset


def test_shapes_and_counts():
    ds = SyntheticDataset(train=True)
    assert ds.num_classes == 20
    assert len(ds) == 20 * 32
    assert ds.input_size == 8

    image, label = ds[0]
    assert image.shape == (8, 8)
    assert isinstance(label, int)
    assert 0 <= label < ds.num_classes


def test_class_to_indices_covers_every_sample_exactly_once():
    ds = SyntheticDataset(train=True)
    class_to_indices = ds.get_class_to_indices()
    assert set(class_to_indices) == set(range(ds.num_classes))

    all_indices = sorted(i for idxs in class_to_indices.values() for i in idxs)
    assert all_indices == list(range(len(ds)))
    for c, idxs in class_to_indices.items():
        assert len(idxs) == ds.SAMPLES_PER_CLASS
        assert all(ds[i][1] == c for i in idxs)


def test_train_and_test_splits_share_class_patterns_but_differ_in_noise():
    train = SyntheticDataset(train=True)
    test = SyntheticDataset(train=False)

    train_images = torch.stack([train[i][0] for i in range(len(train))])
    test_images = torch.stack([test[i][0] for i in range(len(test))])
    assert not torch.equal(train_images, test_images)  # different noise draws

    # Same underlying per-class pattern: nearest-centroid classification
    # (centroids from train, evaluated on test) should be near-perfect --
    # this is the "actually learnable" signal the revisit-safety gate needs.
    centroids = np.stack(
        [train_images[i * 32 : (i + 1) * 32].mean(dim=0).numpy() for i in range(20)]
    )
    correct = 0
    for c in range(20):
        for i in range(32):
            sample = test_images[c * 32 + i].numpy()
            distances = [np.linalg.norm(sample - centroids[k]) for k in range(20)]
            if int(np.argmin(distances)) == c:
                correct += 1
    accuracy = correct / (20 * 32)
    assert accuracy > 0.9  # comfortably above chance (1/20 = 0.05)
