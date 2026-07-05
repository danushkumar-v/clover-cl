"""StreamSpec v2 + RevisitSpec + validation (SPEC §3.1-§3.2).

One spec unifies v1's two drifting overlap mechanisms (same-id ``OverlapPair``
vs. echo classes): ``label`` is an axis (``new`` = fresh echo id, ``same`` =
original id kept), not a mechanism choice.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Union

_PLACEMENTS = frozenset({"random", "spaced", "end_of_stream", "clustered", "every_task"})
_LABELS = frozenset({"new", "same"})
_TASK_SIZES = frozenset({"fixed", "grow"})
_PARTIAL_RE = re.compile(r"^partial:(0(?:\.\d+)?|1(?:\.0+)?)$")


@dataclass(frozen=True)
class DatasetInfo:
    """Minimal dataset metadata a scenario factory/planner needs.

    Deliberately not the full ``CLDataset`` ABC (added in P7/P8) — keeps
    ``clover/core`` free of I/O and lets P1's planner/scenarios be tested
    against a small synthetic ``num_classes`` with no real dataset.
    """

    name: str
    num_classes: int


@dataclass(frozen=True)
class ImageRelation:
    """Parsed form of a revisit's ``images: same|new|partial:<pct>`` field."""

    kind: str
    overlap_pct: float | None = None

    @classmethod
    def parse(cls, raw: str) -> "ImageRelation":
        if raw == "same":
            return cls("same")
        if raw == "new":
            return cls("new")
        match = _PARTIAL_RE.match(raw)
        if match:
            return cls("partial", float(match.group(1)))
        raise ValueError(
            f"images must be 'same', 'new', or 'partial:<pct>' (0<=pct<=1), got {raw!r}."
        )


def _unknown_key_error(unknown: set[str], allowed: frozenset[str], kind: str) -> str:
    parts = []
    for key in sorted(unknown):
        suggestion = difflib.get_close_matches(key, allowed, n=1)
        hint = f" did you mean {suggestion[0]!r}?" if suggestion else ""
        parts.append(f"unknown {kind} spec key {key!r}.{hint}")
    return " ".join(parts) + f" allowed keys: {sorted(allowed)}"


@dataclass
class RevisitSpec:
    """One class-revisit pattern within a stream.

    Attributes:
        classes: ``"task0"``, an explicit list of class ids, or
            ``{"random": N}`` (pick N classes using the stream_seed rng).
        placement: ``random | spaced | end_of_stream | clustered | every_task``.
        label: ``new`` (fresh echo id) or ``same`` (keeps its original id).
        images: ``same | new | partial:<pct>``.
        min_gap: Minimum experiences between the first occurrence and any
            revisit (and between consecutive revisits).
        times: How many times each class revisits.
    """

    classes: Union[str, list[int], dict[str, int]]
    placement: str = "random"
    label: str = "new"
    images: str = "new"
    min_gap: int = 1
    times: int = 1

    _ALLOWED_KEYS = frozenset({"classes", "placement", "label", "images", "min_gap", "times"})

    def validate(self) -> None:
        if self.placement not in _PLACEMENTS:
            raise ValueError(
                f"placement must be one of {sorted(_PLACEMENTS)}, got {self.placement!r}."
            )
        if self.label not in _LABELS:
            raise ValueError(f"label must be one of {sorted(_LABELS)}, got {self.label!r}.")
        ImageRelation.parse(self.images)
        if self.min_gap < 1:
            raise ValueError(f"min_gap must be >= 1, got {self.min_gap}.")
        if self.times < 1:
            raise ValueError(f"times must be >= 1, got {self.times}.")

        if isinstance(self.classes, str):
            if self.classes != "task0":
                raise ValueError(f"classes string shorthand must be 'task0', got {self.classes!r}.")
        elif isinstance(self.classes, dict):
            keys = set(self.classes)
            if keys != {"random"}:
                raise ValueError(
                    f"classes dict must be exactly {{'random': N}}, got keys {sorted(keys)}."
                )
            n_pick = self.classes["random"]
            if not isinstance(n_pick, int) or n_pick < 1:
                raise ValueError(f"classes.random must be a positive int, got {n_pick!r}.")
        elif isinstance(self.classes, list):
            if not self.classes:
                raise ValueError("classes list must not be empty.")
        else:
            raise ValueError(
                f"classes must be 'task0', a list of ids, or {{'random': N}}, got {self.classes!r}."
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RevisitSpec":
        unknown = set(raw) - cls._ALLOWED_KEYS
        if unknown:
            raise ValueError(_unknown_key_error(unknown, cls._ALLOWED_KEYS, "revisit"))
        rv = cls(
            classes=raw["classes"],
            placement=raw.get("placement", "random"),
            label=raw.get("label", "new"),
            images=raw.get("images", "new"),
            min_gap=int(raw.get("min_gap", 1)),
            times=int(raw.get("times", 1)),
        )
        rv.validate()
        return rv

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "placement": self.placement,
            "label": self.label,
            "images": self.images,
            "min_gap": self.min_gap,
            "times": self.times,
        }


@dataclass
class StreamSpec:
    """The only user-facing stream spec (SPEC §3.1)."""

    dataset: str
    init_cls: int
    increment: int
    task_size: str = "fixed"
    revisits: list[RevisitSpec] = field(default_factory=list)
    shuffle_seed: int = 1993
    stream_seed: int = 42
    data_root: str = "./data"

    _ALLOWED_KEYS = frozenset(
        {
            "dataset",
            "init_cls",
            "increment",
            "task_size",
            "revisits",
            "shuffle_seed",
            "stream_seed",
            "data_root",
        }
    )

    def validate(self) -> None:
        """Structural validation only — no dataset lookup.

        Plan-time feasibility (min_gap vs. stream length, class-budget
        overflow, contiguous first-appearance) is checked by
        ``planner.resolve()``, where the concrete class count exists.
        """
        if self.init_cls < 1:
            raise ValueError(f"init_cls must be >= 1, got {self.init_cls}.")
        if self.increment < 1:
            raise ValueError(f"increment must be >= 1, got {self.increment}.")
        if self.task_size not in _TASK_SIZES:
            raise ValueError(
                f"task_size must be one of {sorted(_TASK_SIZES)}, got {self.task_size!r}."
            )
        for rv in self.revisits:
            rv.validate()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StreamSpec":
        unknown = set(raw) - cls._ALLOWED_KEYS
        if unknown:
            raise ValueError(_unknown_key_error(unknown, cls._ALLOWED_KEYS, "stream"))
        revisits = [RevisitSpec.from_dict(r) for r in raw.get("revisits", [])]
        spec = cls(
            dataset=raw["dataset"],
            init_cls=int(raw["init_cls"]),
            increment=int(raw["increment"]),
            task_size=raw.get("task_size", "fixed"),
            revisits=revisits,
            shuffle_seed=int(raw.get("shuffle_seed", 1993)),
            stream_seed=int(raw.get("stream_seed", 42)),
            data_root=str(raw.get("data_root", "./data")),
        )
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "init_cls": self.init_cls,
            "increment": self.increment,
            "task_size": self.task_size,
            "revisits": [rv.to_dict() for rv in self.revisits],
            "shuffle_seed": self.shuffle_seed,
            "stream_seed": self.stream_seed,
            "data_root": self.data_root,
        }
