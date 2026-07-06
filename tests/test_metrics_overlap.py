"""CLOVER-specific metrics correctness (SPEC §10): hand-computed fixtures
for the echo-aware bookkeeping helpers and the RAG/Repetition-Gain/
Anchor-Retention/Long-Range-Retention metric functions.
"""

from __future__ import annotations

import math

from clover.core.experience import Experience, LabelSpaceView
from clover.core.plan import EchoEntry
from clover.core.stream import Stream
from clover.evaluation.history import PerClassHistory
from clover.evaluation.metrics import (
    anchor_classes,
    anchor_retention,
    classify_task,
    echo_source_ids,
    first_appearance_map,
    image_level_bonus,
    long_range_retention,
    rag_mean,
    rag_per_class,
    repetition_gain,
    revisit_task_map,
)


def _experience(task_label, classes_in_this_experience, revisiting, first_appearance, echo_map=None):
    here = classes_in_this_experience
    return Experience(
        task_label=task_label,
        benchmark_name="bench",
        dataset="fake",
        classes_in_this_experience=here,
        classes_seen_so_far=sorted(set(here)),
        classes_in_future=[],
        total_classes_in_stream=10,
        revisiting_classes=revisiting,
        first_appearance_of=first_appearance,
        overlap_with_previous={},
        echo_map=echo_map or {},
        label_space=LabelSpaceView(
            new_classes=frozenset(first_appearance),
            seen_classes=frozenset(),
            revisiting_classes=frozenset(revisiting),
            head_size=len(here),
        ),
        image_indices={c: [] for c in here},
        n_samples=0,
    )


# Task 0: classes {0,1} first appear.
# Task 1: classes {2,3} first appear; class 0 same-id revisits.
# Task 2: class 4 first appears as an echo of class 1 (image_relation="new").
_EXP0 = _experience(0, [0, 1], revisiting=[], first_appearance=[0, 1])
_EXP1 = _experience(1, [0, 2, 3], revisiting=[0], first_appearance=[2, 3])
_ECHO_TABLE = {4: EchoEntry(source_id=1, image_relation="new")}
_EXP2 = _experience(2, [4], revisiting=[], first_appearance=[4], echo_map=_ECHO_TABLE)

_STREAM = Stream([_EXP0, _EXP1, _EXP2], split="train")


def test_first_appearance_map():
    assert first_appearance_map(_STREAM) == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}


def test_revisit_task_map_records_first_revisit_only():
    assert revisit_task_map(_STREAM) == {0: 1}


def test_echo_source_ids():
    assert echo_source_ids(_ECHO_TABLE) == frozenset({1})


def test_anchor_classes_empty_when_no_class_persists_every_experience():
    assert anchor_classes(_STREAM) == []


def test_anchor_classes_finds_the_common_class():
    exp_a = _experience(0, [0, 1], revisiting=[], first_appearance=[0, 1])
    exp_b = _experience(1, [0, 2], revisiting=[0], first_appearance=[2])
    exp_c = _experience(2, [0, 3], revisiting=[0], first_appearance=[3])
    stream = Stream([exp_a, exp_b, exp_c], split="train")
    assert anchor_classes(stream) == [0]


def test_classify_task_splits_returning_vs_fresh():
    fam = first_appearance_map(_STREAM)
    rtm = revisit_task_map(_STREAM)

    returning, fresh = classify_task(_EXP1, {}, rtm, fam)
    assert returning == [0]
    assert fresh == [2, 3]

    returning2, fresh2 = classify_task(_EXP2, _ECHO_TABLE, rtm, fam)
    assert returning2 == [4]  # echo id counts as "returning", not "fresh"
    assert fresh2 == []


def test_rag_per_class_and_mean():
    history = PerClassHistory()
    history.record(0, {0: 0.9, 1: 0.85})
    history.record(1, {0: 0.7, 1: 0.6, 2: 0.95, 3: 0.92})

    fam = first_appearance_map(_STREAM)
    rtm = revisit_task_map(_STREAM)

    per_class = rag_per_class(history, fam, rtm)
    assert math.isclose(per_class[0], 0.7 - 0.9)
    assert math.isclose(rag_mean(history, fam, rtm), -0.2)


def test_rag_mean_is_nan_when_no_revisiting_classes():
    history = PerClassHistory()
    history.record(0, {0: 0.9})
    assert math.isnan(rag_mean(history, {0: 0}, {}))


def test_repetition_gain_hand_computed():
    history = PerClassHistory()
    history.record(1, {0: 0.7, 2: 0.95, 3: 0.92})
    assert math.isclose(repetition_gain(history, task=1, returning_ids=[0], fresh_ids=[2, 3]), 0.7 - 0.935)


def test_repetition_gain_is_nan_when_either_group_empty():
    history = PerClassHistory()
    history.record(1, {0: 0.7})
    assert math.isnan(repetition_gain(history, task=1, returning_ids=[0], fresh_ids=[]))


def test_anchor_retention_hand_computed():
    history = PerClassHistory()
    history.record(2, {0: 0.8, 5: 0.6})
    assert math.isclose(anchor_retention(history, anchor_ids=[0, 5], final_task=2), 0.7)


def test_anchor_retention_is_nan_for_empty_anchor_list():
    history = PerClassHistory()
    assert math.isnan(anchor_retention(history, anchor_ids=[], final_task=2))


def test_long_range_retention_reads_echo_source_ids_not_echo_ids():
    history = PerClassHistory()
    history.record(2, {1: 0.55, 4: 0.1})  # class 1 = echo source, class 4 = the echo id itself
    retention_ids = sorted(echo_source_ids(_ECHO_TABLE))
    assert retention_ids == [1]
    assert math.isclose(long_range_retention(history, retention_ids, final_task=2), 0.55)


def test_image_level_bonus_is_always_nan_stub():
    assert math.isnan(image_level_bonus())
