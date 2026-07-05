"""Experience invariants (SPEC §4)."""

from __future__ import annotations

import pytest

from clover.core.experience import Experience, LabelSpaceView


def _make(revisiting, first_appearance, here=None):
    here = here if here is not None else sorted(set(revisiting) | set(first_appearance))
    return Experience(
        task_label=0,
        benchmark_name="bench",
        dataset="synthetic",
        classes_in_this_experience=here,
        classes_seen_so_far=here,
        classes_in_future=[],
        total_classes_in_stream=10,
        revisiting_classes=revisiting,
        first_appearance_of=first_appearance,
        overlap_with_previous={},
        echo_map={},
        label_space=LabelSpaceView(
            new_classes=frozenset(first_appearance),
            seen_classes=frozenset(revisiting),
            revisiting_classes=frozenset(revisiting),
            head_size=len(here),
        ),
        image_indices={c: [] for c in here},
        n_samples=0,
    )


def test_valid_partition_constructs():
    exp = _make(revisiting=[0, 1], first_appearance=[2, 3])
    assert exp.has_class(0)
    assert not exp.has_class(99)
    assert exp.is_revisit_experience()


def test_missing_class_from_partition_rejected():
    with pytest.raises(ValueError, match="must equal classes_in_this_experience"):
        _make(revisiting=[0], first_appearance=[2], here=[0, 1, 2])


def test_overlapping_partition_rejected():
    with pytest.raises(ValueError, match="must be disjoint"):
        _make(revisiting=[0, 1], first_appearance=[1, 2], here=[0, 1, 2])


def test_no_revisit_experience_when_all_first_appearance():
    exp = _make(revisiting=[], first_appearance=[0, 1, 2])
    assert not exp.is_revisit_experience()
