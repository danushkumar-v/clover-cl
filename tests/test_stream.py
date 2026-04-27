"""Tests for clover/core/stream.py."""

from __future__ import annotations

import pytest
import torch.utils.data

from clover.core.experience import Experience
from clover.core.stream import Stream


def _make_dataset():
    return torch.utils.data.TensorDataset()


def _make_exp(task_label, here, revisiting=None, seen=None):
    revisiting = revisiting or []
    first_app = [c for c in here if c not in revisiting]
    seen = seen or here
    return Experience(
        task_label=task_label,
        benchmark_name="test",
        dataset=_make_dataset(),
        classes_in_this_experience=sorted(here),
        classes_seen_so_far=sorted(seen),
        classes_in_future=[],
        total_classes_in_stream=30,
        revisiting_classes=sorted(revisiting),
        first_appearance_of=sorted(first_app),
        overlap_with_previous={},
        image_indices={c: [c * 5] for c in here},
        n_samples=len(here) * 5,
    )


def _build_stream():
    """Build a small 4-experience stream with 2 revisits."""
    exps = [
        _make_exp(0, [0, 1, 2]),
        _make_exp(1, [3, 4, 5]),
        _make_exp(2, [0, 6, 7], revisiting=[0], seen=[0, 1, 2, 3, 4, 5, 6, 7]),
        _make_exp(3, [3, 8, 9], revisiting=[3], seen=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ]
    return Stream(exps, split="train")


def test_len():
    s = _build_stream()
    assert len(s) == 4


def test_iteration_in_order():
    s = _build_stream()
    labels = [exp.task_label for exp in s]
    assert labels == [0, 1, 2, 3]


def test_indexing():
    s = _build_stream()
    assert s[0].task_label == 0
    assert s[2].task_label == 2


def test_split_attribute():
    s = _build_stream()
    assert s.split == "train"


def test_invalid_split_raises():
    with pytest.raises(ValueError):
        Stream([], split="invalid")


def test_revisit_density():
    s = _build_stream()
    # 2 out of 4 experiences have revisits
    assert s.revisit_density() == pytest.approx(0.5)


def test_revisit_density_empty():
    s = Stream([], split="train")
    assert s.revisit_density() == 0.0


def test_class_appearance_count():
    s = _build_stream()
    counts = s.class_appearance_count()
    # class 0 appears in exp 0 and exp 2
    assert counts[0] == 2
    # class 3 appears in exp 1 and exp 3
    assert counts[3] == 2
    # class 1 appears only in exp 0
    assert counts[1] == 1


def test_average_overlap():
    s = _build_stream()
    # Consecutive pairs: (0,1)→0 shared, (1,2)→0 shared, (2,3)→0 shared
    # Wait: exp2 has [0,6,7], exp3 has [3,8,9] → 0 shared
    # But exp1 has [3,4,5], exp2 has [0,6,7] → 0 shared
    assert s.average_overlap() == pytest.approx(0.0)


def test_average_overlap_with_consecutive_sharing():
    exps = [
        _make_exp(0, [0, 1, 2]),
        _make_exp(1, [0, 3, 4], revisiting=[0], seen=[0, 1, 2, 3, 4]),
        _make_exp(2, [0, 5, 6], revisiting=[0], seen=[0, 1, 2, 3, 4, 5, 6]),
    ]
    s = Stream(exps, split="train")
    # (0,1)→1 shared (class 0), (1,2)→1 shared (class 0) → mean=1.0
    assert s.average_overlap() == pytest.approx(1.0)
