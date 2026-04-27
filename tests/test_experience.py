"""Tests for clover/core/experience.py."""

from __future__ import annotations

import pytest
import torch.utils.data

from clover.core.experience import Experience


def _make_dataset():
    return torch.utils.data.TensorDataset()


def _make_exp(**overrides):
    defaults = dict(
        task_label=0,
        benchmark_name="test_bench",
        dataset=_make_dataset(),
        classes_in_this_experience=[0, 1, 2, 3, 4],
        classes_seen_so_far=[0, 1, 2, 3, 4],
        classes_in_future=[5, 6, 7],
        total_classes_in_stream=10,
        revisiting_classes=[],
        first_appearance_of=[0, 1, 2, 3, 4],
        overlap_with_previous={},
        image_indices={0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]},
        n_samples=10,
    )
    defaults.update(overrides)
    return Experience(**defaults)


def test_experience_construction():
    exp = _make_exp()
    assert exp.task_label == 0
    assert exp.benchmark_name == "test_bench"
    assert exp.n_samples == 10


def test_partition_invariant_holds():
    exp = _make_exp(
        classes_in_this_experience=[0, 1, 2, 3, 4],
        revisiting_classes=[3, 4],
        first_appearance_of=[0, 1, 2],
    )
    assert set(exp.revisiting_classes) | set(exp.first_appearance_of) == set(
        exp.classes_in_this_experience
    )
    assert not (set(exp.revisiting_classes) & set(exp.first_appearance_of))


def test_partition_union_must_equal_classes_in_experience():
    with pytest.raises(ValueError, match="must equal"):
        _make_exp(
            classes_in_this_experience=[0, 1, 2],
            revisiting_classes=[0],
            first_appearance_of=[1],  # missing class 2
        )


def test_partition_must_be_disjoint():
    with pytest.raises(ValueError, match="disjoint"):
        _make_exp(
            classes_in_this_experience=[0, 1, 2],
            revisiting_classes=[0, 1],
            first_appearance_of=[1, 2],  # class 1 in both
        )


def test_has_class():
    exp = _make_exp(classes_in_this_experience=[0, 1, 2, 3, 4], first_appearance_of=[0, 1, 2, 3, 4])
    assert exp.has_class(3)
    assert not exp.has_class(99)


def test_is_revisit_experience_false_when_no_revisits():
    exp = _make_exp(revisiting_classes=[], first_appearance_of=[0, 1, 2, 3, 4])
    assert not exp.is_revisit_experience()


def test_is_revisit_experience_true_when_revisits():
    exp = _make_exp(
        classes_in_this_experience=[0, 1, 2, 3, 4],
        revisiting_classes=[3],
        first_appearance_of=[0, 1, 2, 4],
    )
    assert exp.is_revisit_experience()


def test_frozen_dataclass():
    exp = _make_exp()
    with pytest.raises((AttributeError, TypeError)):
        exp.task_label = 99  # type: ignore[misc]


def test_overlap_with_previous():
    exp = _make_exp(
        task_label=3,
        overlap_with_previous={0: 2, 1: 1},
    )
    assert exp.overlap_with_previous[0] == 2
    assert exp.overlap_with_previous[1] == 1
