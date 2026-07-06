"""Standard + CLOVER-specific metrics (SPEC §10), ported from the bench's
``metrics/standard.py``/``metrics/overlap.py`` (no code copied).
"""

from __future__ import annotations

from clover.evaluation.metrics.overlap import (
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
from clover.evaluation.metrics.standard import (
    aggregate_accuracy,
    average_incremental_accuracy,
    backward_transfer,
    forgetting,
    forward_transfer,
)

__all__ = [
    "aggregate_accuracy",
    "average_incremental_accuracy",
    "backward_transfer",
    "forgetting",
    "forward_transfer",
    "first_appearance_map",
    "revisit_task_map",
    "anchor_classes",
    "echo_source_ids",
    "classify_task",
    "rag_per_class",
    "rag_mean",
    "repetition_gain",
    "anchor_retention",
    "long_range_retention",
    "image_level_bonus",
]
