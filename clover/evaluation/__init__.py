"""Per-class evaluator + R-matrix + per-class history (SPEC §10).

Standard + CLOVER metrics built on top of these live in
``clover.evaluation.metrics``.
"""

from __future__ import annotations

from clover.evaluation.evaluator import PerClassEvaluator, RMatrix
from clover.evaluation.history import PerClassHistory

__all__ = ["PerClassEvaluator", "RMatrix", "PerClassHistory"]
