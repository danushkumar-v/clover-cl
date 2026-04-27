"""Declarative stream specification — describes WHAT, not WHERE."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Literal, Union

import yaml


@dataclass
class RevisitSpec:
    """Describes a class-revisit pattern within a stream.

    Attributes:
        classes: Explicit list of remapped class IDs to revisit, or an integer
            ``N`` meaning "pick N classes at random using the stream_seed".
        times: How many times each class revisits (default 1).
        placement: Strategy for distributing revisits across the stream:
            ``"random"`` — sample uniformly subject to ``min_gap``.
            ``"spaced"`` — distribute as evenly as possible.
            ``"end_of_stream"`` — place all revisits in the final experiences.
            ``"clustered"`` — place revisits in a tight consecutive window.
        min_gap: Minimum number of experiences between the first occurrence and
            any revisit (and between consecutive revisits).
    """

    classes: Union[List[int], int]
    times: int = 1
    placement: Literal["random", "spaced", "end_of_stream", "clustered"] = "random"
    min_gap: int = 1

    def __post_init__(self) -> None:
        if self.times < 1:
            raise ValueError(f"times must be >= 1, got {self.times}.")
        if self.min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {self.min_gap}.")
        valid = {"random", "spaced", "end_of_stream", "clustered"}
        if self.placement not in valid:
            raise ValueError(f"placement must be one of {valid}, got {self.placement!r}.")


@dataclass
class StreamSpec:
    """Declarative stream description.

    The user describes the *statistical structure* of the stream — which classes
    revisit, how many times, and with what placement strategy.  The framework
    resolves this into concrete task-pair assignments, seeded by ``stream_seed``.
    Different ``stream_seed`` values yield different concrete streams with the
    same statistical structure, enabling proper multi-seed error bars.

    Attributes:
        dataset: Dataset name, e.g. ``"cifar100"``.
        init_cls: Number of classes in the first experience.
        increment: Number of new classes added per subsequent experience.
        revisits: List of :class:`RevisitSpec` objects describing each revisit
            pattern.  An empty list produces a disjoint (PILOT-equivalent) stream.
        image_strategy: How images are distributed for shared classes.
            ``"disjoint"`` — each task gets a unique slice.
            ``"duplicate"`` — all images available to every task.
            ``"partial_duplicate"`` — unique slices plus a shared core.
        image_overlap_pct: For ``"partial_duplicate"``, fraction of images shared.
        image_split_ratio: For ``"disjoint"``, fraction allocated to the first task.
        shuffle_seed: Seed for PILOT-compatible class-order permutation (default 1993).
        stream_seed: Seed used for revisit placement and random class selection.
        data_root: Root directory for dataset files.
        preserve_task_size: When ``True`` (default), inject overlap without growing
            non-final tasks.
    """

    dataset: str
    init_cls: int
    increment: int
    revisits: List[RevisitSpec] = field(default_factory=list)
    image_strategy: Literal["disjoint", "duplicate", "partial_duplicate"] = "disjoint"
    image_overlap_pct: float = 0.0
    image_split_ratio: float = 0.5
    shuffle_seed: int = 1993
    stream_seed: int = 42
    data_root: str = "./data"
    preserve_task_size: bool = True

    def validate(self) -> None:
        """Raise :class:`ValueError` if the spec is self-contradictory."""
        if self.init_cls < 1:
            raise ValueError(f"init_cls must be >= 1, got {self.init_cls}.")
        if self.increment < 1:
            raise ValueError(f"increment must be >= 1, got {self.increment}.")
        if not 0.0 <= self.image_overlap_pct <= 1.0:
            raise ValueError(
                f"image_overlap_pct must be in [0, 1], got {self.image_overlap_pct}."
            )
        if not 0.0 < self.image_split_ratio < 1.0:
            raise ValueError(
                f"image_split_ratio must be in (0, 1), got {self.image_split_ratio}."
            )
        for rv in self.revisits:
            rv.__post_init__()

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary (JSON/YAML compatible)."""
        revisits = []
        for rv in self.revisits:
            revisits.append(
                {
                    "classes": rv.classes,
                    "times": rv.times,
                    "placement": rv.placement,
                    "min_gap": rv.min_gap,
                }
            )
        return {
            "dataset": self.dataset,
            "init_cls": self.init_cls,
            "increment": self.increment,
            "revisits": revisits,
            "image_strategy": self.image_strategy,
            "image_overlap_pct": self.image_overlap_pct,
            "image_split_ratio": self.image_split_ratio,
            "shuffle_seed": self.shuffle_seed,
            "stream_seed": self.stream_seed,
            "data_root": self.data_root,
            "preserve_task_size": self.preserve_task_size,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StreamSpec":
        """Deserialise from a plain dictionary."""
        revisits = []
        for rv in raw.get("revisits", []):
            revisits.append(
                RevisitSpec(
                    classes=rv["classes"],
                    times=int(rv.get("times", 1)),
                    placement=rv.get("placement", "random"),
                    min_gap=int(rv.get("min_gap", 1)),
                )
            )
        return cls(
            dataset=raw["dataset"],
            init_cls=int(raw["init_cls"]),
            increment=int(raw["increment"]),
            revisits=revisits,
            image_strategy=raw.get("image_strategy", "disjoint"),
            image_overlap_pct=float(raw.get("image_overlap_pct", 0.0)),
            image_split_ratio=float(raw.get("image_split_ratio", 0.5)),
            shuffle_seed=int(raw.get("shuffle_seed", 1993)),
            stream_seed=int(raw.get("stream_seed", 42)),
            data_root=str(raw.get("data_root", "./data")),
            preserve_task_size=bool(raw.get("preserve_task_size", True)),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "StreamSpec":
        """Load a :class:`StreamSpec` from a YAML config file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: str) -> "StreamSpec":
        """Load a :class:`StreamSpec` from a JSON file."""
        with open(path) as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)
