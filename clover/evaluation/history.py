"""Per-class accuracy history (SPEC §10): the CLOVER-specific metrics
(RAG, Repetition Gain, Anchor/Long-Range Retention) need per-class
accuracy at arbitrary ``(class_id, task)`` pairs, not just the R-matrix's
task-level reduction. ``Trainer.run()`` already computes exact per-class
accuracy every experience via ``PerClassEvaluator.evaluate`` -- this class
just persists what was already computed instead of discarding it after
reducing it to the R-matrix's per-experience mean.
"""

from __future__ import annotations

import math
from typing import Any, Dict


class PerClassHistory:
    """``{class_id: {task: accuracy}}``, built up one experience at a time."""

    def __init__(self) -> None:
        self._data: Dict[int, Dict[int, float]] = {}

    def record(self, task: int, per_class_acc: Dict[int, float]) -> None:
        """Record every ``(class_id, accuracy)`` pair evaluated at *task*."""
        for class_id, acc in per_class_acc.items():
            self._data.setdefault(class_id, {})[task] = acc

    def get(self, class_id: int, task: int) -> float:
        """Accuracy for *class_id* at *task*, or ``NaN`` if never recorded."""
        return self._data.get(class_id, {}).get(task, math.nan)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Plain-dict form for checkpointing (``torch.save`` doesn't need
        string keys, but JSON-style round-tripping elsewhere in this
        codebase does -- keeping this consistently stringified avoids two
        different conventions for the same shape)."""
        return {
            str(class_id): {str(task): acc for task, acc in tasks.items()}
            for class_id, tasks in self._data.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[Any, Dict[Any, float]]) -> "PerClassHistory":
        obj = cls()
        obj._data = {
            int(class_id): {int(task): acc for task, acc in tasks.items()}
            for class_id, tasks in data.items()
        }
        return obj
