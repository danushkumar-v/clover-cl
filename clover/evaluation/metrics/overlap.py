"""CLOVER-specific metrics (SPEC §10): Repetition Gain, RAG (Revisit
Accuracy Gain), Anchor Retention, Long-Range Retention. Ported from the
bench's ``metrics/overlap.py`` (confirmed via read-only research, no code
copied). Unlike the standard metrics, these need a **per-class accuracy
history** (``PerClassHistory``) plus revisit/echo bookkeeping -- not
derivable from the R-matrix alone.

Echo-aware: an "echo" class is a *fresh* label id that is a conceptual
repeat of an earlier class (as opposed to a same-id revisit, where the
identical label id reappears -- e.g. ``cumulative_drift``'s anchors).
Retention on an echoed lineage is read on the echo's *source* id, since
that's where the original category's identity actually lives
(``long_range_retention``'s ``retention_ids`` argument) -- confirmed via
research this is exactly what the bench does and why.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from clover.core.experience import Experience
from clover.core.plan import EchoEntry
from clover.core.stream import Stream
from clover.evaluation.history import PerClassHistory


def first_appearance_map(stream: Stream) -> Dict[int, int]:
    """``{class_id: task}`` for the experience each class first appears in."""
    result: Dict[int, int] = {}
    for exp in stream:
        for c in exp.first_appearance_of:
            result.setdefault(c, exp.task_label)
    return result


def revisit_task_map(stream: Stream) -> Dict[int, int]:
    """``{class_id: task}`` for each class's *first* revisit only -- a
    class revisiting more than once only ever counts its first revisit
    here, matching the bench's own single-revisit-per-class limitation
    (kept as-is, not generalized, for fidelity)."""
    result: Dict[int, int] = {}
    for exp in stream:
        for c in exp.revisiting_classes:
            result.setdefault(c, exp.task_label)
    return result


def anchor_classes(stream: Stream) -> List[int]:
    """Classes present in *every* experience (``cumulative_drift``'s
    anchors are the only core scenario with any)."""
    experiences = list(stream)
    if not experiences:
        return []
    common: Set[int] = set(experiences[0].classes_in_this_experience)
    for exp in experiences[1:]:
        common &= set(exp.classes_in_this_experience)
    return sorted(common)


def echo_source_ids(echo_table: Dict[int, EchoEntry]) -> FrozenSet[int]:
    """The set of *source* ids every echo entry points back to."""
    return frozenset(entry.source_id for entry in echo_table.values())


def classify_task(
    exp: Experience,
    echo_table: Dict[int, EchoEntry],
    revisit_map: Dict[int, int],
    first_appearance: Dict[int, int],
) -> Tuple[List[int], List[int]]:
    """Split *exp*'s classes into ``(returning_ids, fresh_ids)`` for
    Repetition Gain: "returning" = echo ids introduced this experience,
    union classes whose first revisit is this experience; "fresh" =
    first-appearing, non-echo classes that never revisit."""
    returning = [c for c in exp.classes_in_this_experience if c in echo_table]
    returning += [
        c
        for c in exp.classes_in_this_experience
        if c not in echo_table and revisit_map.get(c) == exp.task_label
    ]
    fresh = [
        c
        for c in exp.classes_in_this_experience
        if c not in echo_table
        and first_appearance.get(c) == exp.task_label
        and c not in revisit_map
    ]
    return sorted(returning), sorted(fresh)


def _mean_at(history: PerClassHistory, class_ids: List[int], task: int) -> float:
    values = [history.get(c, task) for c in class_ids]
    valid = [v for v in values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def rag_per_class(
    history: PerClassHistory, first_appearance: Dict[int, int], revisit_map: Dict[int, int]
) -> Dict[int, float]:
    """RAG (Revisit Accuracy Gain) per revisiting class: accuracy at its
    first revisit minus accuracy at its first appearance."""
    result: Dict[int, float] = {}
    for class_id, revisit_task in revisit_map.items():
        first_task = first_appearance.get(class_id)
        if first_task is None:
            continue
        result[class_id] = history.get(class_id, revisit_task) - history.get(class_id, first_task)
    return result


def rag_mean(
    history: PerClassHistory, first_appearance: Dict[int, int], revisit_map: Dict[int, int]
) -> float:
    per_class = rag_per_class(history, first_appearance, revisit_map)
    valid = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def repetition_gain(
    history: PerClassHistory, task: int, returning_ids: List[int], fresh_ids: List[int]
) -> float:
    """Mean accuracy of returning classes minus mean accuracy of fresh
    classes, both measured at *task*. NaN unless both groups are
    non-empty and have at least one valid recorded accuracy."""
    acc_returning = _mean_at(history, returning_ids, task)
    acc_fresh = _mean_at(history, fresh_ids, task)
    if np.isnan(acc_returning) or np.isnan(acc_fresh):
        return float("nan")
    return acc_returning - acc_fresh


def anchor_retention(history: PerClassHistory, anchor_ids: List[int], final_task: int) -> float:
    """Mean accuracy of anchor classes (present in every experience) at
    the final task."""
    return _mean_at(history, anchor_ids, final_task)


def long_range_retention(
    history: PerClassHistory, retention_ids: List[int], final_task: int
) -> float:
    """Mean accuracy, at the final task, of the ids whose *original*
    category identity should still be recognized -- echo *source* ids
    when any echoes exist, else the plain first-appearance ids (the
    caller decides which set to pass; see ``echo_source_ids``)."""
    return _mean_at(history, retention_ids, final_task)


def image_level_bonus() -> float:
    """Reserved: needs a paired split-half control run (same classes,
    disjoint images) to isolate an image-level (vs. concept-level) memory
    effect. Not implemented -- the bench itself never wired this up
    either (always NaN there too); no fixture exists to port faithfully
    against."""
    return float("nan")
