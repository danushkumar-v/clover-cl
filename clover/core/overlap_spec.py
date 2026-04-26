"""Dataclasses that describe how class overlap is configured across tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import yaml


@dataclass
class OverlapPair:
    """Declares which two tasks share a set of classes.

    Args:
        tasks: Indices of the two tasks that share classes (zero-based).
        shared_classes: Remapped class IDs (after shuffle) that appear in both tasks.
    """

    tasks: Tuple[int, int]
    shared_classes: List[int]

    def __post_init__(self) -> None:
        self.tasks = tuple(self.tasks)  # type: ignore[assignment]
        self.shared_classes = list(self.shared_classes)


@dataclass
class ImageSplit:
    """Controls how images of shared classes are divided between tasks.

    Args:
        strategy: How images are partitioned.
            ``disjoint``          – images split by ``ratio``; no image in both tasks.
            ``duplicate``         – all images appear in both tasks.
            ``partial_duplicate`` – ``overlap_pct`` fraction of images appear in both;
                                    the remainder is split equally and uniquely.
        ratio: For ``disjoint``, fraction of images assigned to the *first* task in
               the pair. Must be in [0, 1].
        overlap_pct: For ``partial_duplicate``, fraction of images shared by both
                     tasks. Must be in [0, 1].
    """

    strategy: Literal["disjoint", "duplicate", "partial_duplicate"] = "disjoint"
    ratio: float = 0.5
    overlap_pct: float = 0.0


_VALID_MODES = frozenset(
    [
        "none",
        "exact_replay",
        "partial",
        "hierarchical",
        "distribution_shift",
        "near_miss",
        "long_range_revisit",
        "cumulative_drift",
        "symmetric_pair",
    ]
)


@dataclass
class OverlapSpec:
    """Top-level configuration for a CLOVER overlap scenario.

    Args:
        mode: Named overlap regime. Use ``"none"`` for strict PILOT-equivalent
              disjoint partitioning.
        pairs: List of :class:`OverlapPair` declarations.
        image_split: How images are distributed for shared classes.
        seed: RNG seed used by :mod:`clover.utils.seeding` for all stochastic
              image-assignment decisions.
    """

    mode: Literal[
        "none",
        "exact_replay",
        "partial",
        "hierarchical",
        "distribution_shift",
        "near_miss",
        "long_range_revisit",
        "cumulative_drift",
        "symmetric_pair",
    ] = "none"
    pairs: List[OverlapPair] = field(default_factory=list)
    image_split: ImageSplit = field(default_factory=ImageSplit)
    seed: int = 42

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the spec is internally inconsistent."""
        if self.mode not in _VALID_MODES:
            raise ValueError(f"Unknown mode {self.mode!r}. Valid: {sorted(_VALID_MODES)}")

        if self.mode == "none" and self.pairs:
            raise ValueError("mode='none' must have an empty pairs list.")

        for i, pair in enumerate(self.pairs):
            if any(t < 0 for t in pair.tasks):
                raise ValueError(f"pairs[{i}].tasks contains a negative task index.")
            if len(pair.tasks) != 2:
                raise ValueError(f"pairs[{i}].tasks must have exactly 2 elements.")
            if self.mode != "near_miss" and len(pair.shared_classes) == 0:
                raise ValueError(
                    f"pairs[{i}].shared_classes is empty for mode={self.mode!r}."
                )

        split = self.image_split
        if not (0.0 <= split.ratio <= 1.0):
            raise ValueError(
                f"ImageSplit.ratio={split.ratio} is outside [0, 1]."
            )
        if not (0.0 <= split.overlap_pct <= 1.0):
            raise ValueError(
                f"ImageSplit.overlap_pct={split.overlap_pct} is outside [0, 1]."
            )

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "OverlapSpec":
        """Load an :class:`OverlapSpec` from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            A validated :class:`OverlapSpec` instance.
        """
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        overlap_raw = raw.get("overlap_spec", raw)
        pairs = [
            OverlapPair(
                tasks=tuple(p["tasks"]),
                shared_classes=p.get("shared_classes", []),
            )
            for p in overlap_raw.get("pairs", [])
        ]
        split_raw = overlap_raw.get("image_split", {})
        image_split = ImageSplit(
            strategy=split_raw.get("strategy", "disjoint"),
            ratio=float(split_raw.get("ratio", 0.5)),
            overlap_pct=float(split_raw.get("overlap_pct", 0.0)),
        )
        spec = cls(
            mode=overlap_raw.get("mode", "none"),
            pairs=pairs,
            image_split=image_split,
            seed=int(overlap_raw.get("seed", 42)),
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for manifest headers)."""
        return {
            "mode": self.mode,
            "seed": self.seed,
            "image_split": {
                "strategy": self.image_split.strategy,
                "ratio": self.image_split.ratio,
                "overlap_pct": self.image_split.overlap_pct,
            },
            "pairs": [
                {"tasks": list(p.tasks), "shared_classes": p.shared_classes}
                for p in self.pairs
            ],
        }
