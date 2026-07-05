"""RMatrix + PerClassEvaluator correctness (SPEC §10, this phase's slice)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from clover.core.experience import Experience, LabelSpaceView
from clover.evaluation import PerClassEvaluator, RMatrix


def test_rmatrix_starts_all_nan():
    m = RMatrix(3)
    assert all(math.isnan(v) for v in m.to_array().flatten())


def test_rmatrix_update_and_read_back():
    m = RMatrix(3)
    m.update(task_evaluated=0, task_trained=2, value=0.75)
    assert m.to_array()[0, 2] == 0.75


def test_rmatrix_rejects_lower_triangular_update():
    m = RMatrix(3)
    with pytest.raises(ValueError, match="must be <="):
        m.update(task_evaluated=2, task_trained=0, value=0.5)


def test_rmatrix_round_trip_via_array():
    m = RMatrix(2)
    m.update(0, 0, 0.9)
    m.update(0, 1, 0.8)
    restored = RMatrix.from_array(m.to_array())
    assert np.array_equal(restored.to_array(), m.to_array(), equal_nan=True)


def test_rmatrix_save_and_load(tmp_path):
    m = RMatrix(2)
    m.update(0, 1, 0.5)
    path = str(tmp_path / "R_matrix.npy")
    m.save(path)
    loaded = RMatrix.load(path)
    assert loaded.to_array()[0, 1] == 0.5


class _FakeDataset:
    """Returns a fixed 1-D feature per index so classifier predictions are exact."""

    input_size = 1

    def __init__(self, features):
        self._features = features

    def __getitem__(self, idx):
        return self._features[idx], -1  # label unused by evaluator


class _FixedClassifier(nn.Module):
    """Predicts argmax of the raw feature vector -- deterministic, hand-verifiable."""

    def forward(self, x):
        return x


def _make_experience(image_indices, first_appearance_of):
    here = sorted(image_indices)
    return Experience(
        task_label=0,
        benchmark_name="bench",
        dataset="fake",
        classes_in_this_experience=here,
        classes_seen_so_far=here,
        classes_in_future=[],
        total_classes_in_stream=len(here),
        revisiting_classes=[c for c in here if c not in first_appearance_of],
        first_appearance_of=first_appearance_of,
        overlap_with_previous={},
        echo_map={},
        label_space=LabelSpaceView(
            new_classes=frozenset(first_appearance_of),
            seen_classes=frozenset(),
            revisiting_classes=frozenset(c for c in here if c not in first_appearance_of),
            head_size=len(here),
        ),
        image_indices=image_indices,
        n_samples=sum(len(v) for v in image_indices.values()),
    )


def test_per_class_evaluator_computes_exact_accuracy():
    # 3 classes, one-hot-ish features so argmax always predicts a known class.
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # correctly predicts class 0
            [1.0, 0.0, 0.0],  # correctly predicts class 0
            [0.0, 0.0, 1.0],  # WRONG for class 1 (predicts class 2)
            [0.0, 1.0, 0.0],  # correctly predicts class 1
            [0.0, 0.0, 1.0],  # correctly predicts class 2
        ]
    )
    dataset = _FakeDataset(features)
    exp = _make_experience(image_indices={0: [0, 1], 1: [2, 3], 2: [4]}, first_appearance_of=[0, 1, 2])

    evaluator = PerClassEvaluator(_FixedClassifier(), dataset, torch.device("cpu"))
    per_class_acc = evaluator.evaluate([exp])

    assert per_class_acc[0] == 1.0
    assert per_class_acc[1] == 0.5
    assert per_class_acc[2] == 1.0


def test_per_class_evaluator_skips_classes_with_no_images():
    dataset = _FakeDataset(torch.zeros(1, 2))
    exp = _make_experience(image_indices={0: []}, first_appearance_of=[0])
    evaluator = PerClassEvaluator(_FixedClassifier(), dataset, torch.device("cpu"))
    assert evaluator.evaluate([exp]) == {}
