"""Stream — an ordered sequence of Experience objects."""

from __future__ import annotations

from typing import Dict, Iterator, List

from clover.core.experience import Experience


class Stream:
    """An ordered sequence of :class:`~clover.core.experience.Experience` objects.

    Supports iteration, indexing, and stream-level metrics.

    Args:
        experiences: Ordered list of Experience objects.
        split: ``"train"`` or ``"test"``.
    """

    def __init__(self, experiences: List[Experience], split: str) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
        self._experiences = list(experiences)
        self.split = split

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._experiences)

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(self, k: int) -> Experience:
        return self._experiences[k]

    # ------------------------------------------------------------------
    # Stream-level metrics
    # ------------------------------------------------------------------

    def revisit_density(self) -> float:
        """Fraction of experiences that contain at least one revisiting class."""
        if not self._experiences:
            return 0.0
        revisit_count = sum(1 for e in self._experiences if e.is_revisit_experience())
        return revisit_count / len(self._experiences)

    def class_appearance_count(self) -> Dict[int, int]:
        """Return ``{class_id: number_of_experiences_it_appears_in}``."""
        counts: Dict[int, int] = {}
        for exp in self._experiences:
            for c in exp.classes_in_this_experience:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def average_overlap(self) -> float:
        """Mean number of shared classes between each pair of consecutive experiences."""
        if len(self._experiences) < 2:
            return 0.0
        total = 0
        for i in range(len(self._experiences) - 1):
            a = set(self._experiences[i].classes_in_this_experience)
            b = set(self._experiences[i + 1].classes_in_this_experience)
            total += len(a & b)
        return total / (len(self._experiences) - 1)
