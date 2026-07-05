"""Per-class evaluator + R-matrix (SPEC §10, this phase's slice only).

Per-class accuracy is the evaluator primitive (bench-proven design): R-matrix
cells are the mean of per-class accuracy over each experience's
first-appearance class set, which is exact for non-uniform/revisit streams
where a coarse "block accuracy" can't represent things correctly. Standard
(A_t/AIA/BWT/FWT/Forgetting) and CLOVER (RAG/Repetition Gain/...) metrics
built on top of this are explicitly P7 scope, not built here.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

from clover.core.experience import Experience
from clover.datasets.base import CLDataset


class RMatrix:
    """``[n, n]`` accuracy matrix: ``R[task_evaluated, task_trained]``.

    Upper-triangular by construction (an experience can only be evaluated
    once training has reached at least that far).
    """

    def __init__(self, n: int) -> None:
        self.n = n
        self._data = np.full((n, n), np.nan, dtype=float)

    def update(self, task_evaluated: int, task_trained: int, value: float) -> None:
        if task_evaluated > task_trained:
            raise ValueError(
                f"task_evaluated ({task_evaluated}) must be <= task_trained ({task_trained})."
            )
        self._data[task_evaluated, task_trained] = value

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    @classmethod
    def from_array(cls, array: np.ndarray) -> "RMatrix":
        obj = cls(array.shape[0])
        obj._data = np.array(array, dtype=float, copy=True)
        return obj

    def save(self, path: str) -> None:
        np.save(path, self._data)

    @classmethod
    def load(cls, path: str) -> "RMatrix":
        return cls.from_array(np.load(path))


class PerClassEvaluator:
    """Runs a classifier over test experiences, returning per-class accuracy."""

    def __init__(self, classifier: nn.Module, test_dataset: CLDataset, device: torch.device) -> None:
        self.classifier = classifier
        self.test_dataset = test_dataset
        self.device = device

    def evaluate(self, experiences: Iterable[Experience]) -> Dict[int, float]:
        """Return ``{class_id: accuracy}`` pooled over every image in *experiences*."""
        self.classifier.eval()
        correct: Dict[int, int] = {}
        total: Dict[int, int] = {}
        with torch.no_grad():
            for exp in experiences:
                for class_id, indices in exp.image_indices.items():
                    if not indices:
                        continue
                    images = torch.stack([self.test_dataset[i][0] for i in indices]).to(self.device)
                    targets = torch.full((len(indices),), class_id, dtype=torch.long)
                    preds = self.classifier(images).argmax(dim=-1).cpu()
                    correct[class_id] = correct.get(class_id, 0) + int((preds == targets).sum().item())
                    total[class_id] = total.get(class_id, 0) + len(indices)
        return {c: correct[c] / total[c] for c in total}
