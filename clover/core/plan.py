"""StreamPlan: resolved, concrete, serializable (SPEC §3.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import clover
from clover.core.manifest import load_manifest, save_manifest
from clover.core.spec import StreamSpec


@dataclass(frozen=True)
class EchoEntry:
    """One echo-id table entry: ``echo_id -> (source_id, image_relation)``."""

    source_id: int
    image_relation: str
    overlap_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "image_relation": self.image_relation,
            "overlap_pct": self.overlap_pct,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EchoEntry":
        return cls(
            source_id=raw["source_id"],
            image_relation=raw["image_relation"],
            overlap_pct=raw.get("overlap_pct"),
        )


@dataclass(frozen=True)
class StreamPlan:
    """A fully concrete, seed-resolved, JSON-serializable stream plan.

    Deterministic given ``(spec, dataset_info)``. The manifest **is** the
    serialized plan + header (version, spec, seeds) — ``from_manifest``
    reconstructs a run's data exactly.
    """

    spec: StreamSpec
    num_classes: int
    class_order: list[int]
    task_class_lists: list[list[int]]
    echo_table: dict[int, EchoEntry]
    revisit_ids: frozenset[int]
    head_size_schedule: list[int]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "_header": {
                "clover_version": clover.__version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": self.spec.dataset,
                "init_cls": self.spec.init_cls,
                "increment": self.spec.increment,
            },
            "spec": self.spec.to_dict(),
            "num_classes": self.num_classes,
            "class_order": self.class_order,
            "task_class_lists": self.task_class_lists,
            "echo_table": {str(k): v.to_dict() for k, v in self.echo_table.items()},
            "revisit_ids": sorted(self.revisit_ids),
            "head_size_schedule": self.head_size_schedule,
        }

    def save(self, path: str) -> None:
        save_manifest(self.to_manifest(), path)

    @classmethod
    def from_manifest(cls, path: str) -> "StreamPlan":
        data = load_manifest(path)
        return cls(
            spec=StreamSpec.from_dict(data["spec"]),
            num_classes=data["num_classes"],
            class_order=list(data["class_order"]),
            task_class_lists=[list(t) for t in data["task_class_lists"]],
            echo_table={int(k): EchoEntry.from_dict(v) for k, v in data["echo_table"].items()},
            revisit_ids=frozenset(data["revisit_ids"]),
            head_size_schedule=list(data["head_size_schedule"]),
        )
