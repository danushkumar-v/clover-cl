"""Per-class evaluator + R-matrix (SPEC §10, this phase's slice).

Standard + CLOVER metrics on top of the R-matrix are P7 scope.
"""

from __future__ import annotations

from clover.evaluation.evaluator import PerClassEvaluator, RMatrix

__all__ = ["PerClassEvaluator", "RMatrix"]
