"""Typed config sections (SPEC §9): plain dataclasses, strict unknown-key
rejection. Layered-defaults + stream resolution happen in ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clover.training.trainer import OPTIMIZER_FACTORIES
from clover.utils.strict_dict import reject_unknown_keys

_AMP_VALUES = frozenset({"none", "bf16", "fp16"})
_TASK_SIZES = frozenset({"fixed", "grow"})


@dataclass
class RunSection:
    name: Optional[str] = None
    seed: int = 1993
    output_dir: str = "runs"

    _ALLOWED_KEYS = frozenset({"name", "seed", "output_dir"})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RunSection":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "run config")
        return cls(
            name=raw.get("name"),
            seed=int(raw.get("seed", 1993)),
            output_dir=str(raw.get("output_dir", "runs")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "seed": self.seed, "output_dir": self.output_dir}


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3

    _ALLOWED_KEYS = frozenset({"name", "lr"})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OptimizerConfig":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "optimizer config")
        name = str(raw.get("name", "adam"))
        if name not in OPTIMIZER_FACTORIES:
            raise ValueError(
                f"optimizer name must be one of {sorted(OPTIMIZER_FACTORIES)}, got {name!r}."
            )
        return cls(name=name, lr=float(raw.get("lr", 1e-3)))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lr": self.lr}


@dataclass
class TrainingSection:
    """``amp`` is validated but only collapsed to a bool for
    ``TrainContext.amp`` right now -- no autocast wrapping exists anywhere
    yet (no gradient-trained method exists), so finer bf16-vs-fp16 dtype
    handling is deferred to whichever phase adds one.
    """

    epochs: int = 1
    batch_size: int = 32
    optimizer: Optional[OptimizerConfig] = None
    amp: str = "none"

    _ALLOWED_KEYS = frozenset({"epochs", "batch_size", "optimizer", "amp"})

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.amp not in _AMP_VALUES:
            raise ValueError(f"amp must be one of {sorted(_AMP_VALUES)}, got {self.amp!r}.")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainingSection":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "training config")
        optimizer_raw = raw.get("optimizer")
        optimizer = OptimizerConfig.from_dict(optimizer_raw) if optimizer_raw is not None else None
        section = cls(
            epochs=int(raw.get("epochs", 1)),
            batch_size=int(raw.get("batch_size", 32)),
            optimizer=optimizer,
            amp=str(raw.get("amp", "none")),
        )
        section.validate()
        return section

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer.to_dict() if self.optimizer else None,
            "amp": self.amp,
        }


@dataclass
class StreamSection:
    dataset: str
    init_cls: int
    increment: int
    scenario: Optional[str] = None
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    revisits: Optional[List[Dict[str, Any]]] = None
    stream_seed: int = 42
    shuffle_seed: int = 1993
    task_size: str = "fixed"
    data_root: str = "./data"

    _ALLOWED_KEYS = frozenset(
        {
            "dataset",
            "init_cls",
            "increment",
            "scenario",
            "scenario_params",
            "revisits",
            "stream_seed",
            "shuffle_seed",
            "task_size",
            "data_root",
        }
    )

    def validate(self) -> None:
        if self.init_cls < 1:
            raise ValueError(f"init_cls must be >= 1, got {self.init_cls}.")
        if self.increment < 1:
            raise ValueError(f"increment must be >= 1, got {self.increment}.")
        if self.task_size not in _TASK_SIZES:
            raise ValueError(
                f"task_size must be one of {sorted(_TASK_SIZES)}, got {self.task_size!r}."
            )
        if self.scenario is not None and self.revisits is not None:
            raise ValueError(
                "stream: cannot set both 'scenario' and 'revisits' -- choose one "
                f"(got scenario={self.scenario!r} and revisits=<{len(self.revisits)} entries>)."
            )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "StreamSection":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "stream config")
        if "dataset" not in raw:
            raise ValueError("stream config is missing required key 'dataset'.")
        if "init_cls" not in raw:
            raise ValueError("stream config is missing required key 'init_cls'.")
        if "increment" not in raw:
            raise ValueError("stream config is missing required key 'increment'.")
        section = cls(
            dataset=str(raw["dataset"]),
            init_cls=int(raw["init_cls"]),
            increment=int(raw["increment"]),
            scenario=raw.get("scenario"),
            scenario_params=dict(raw.get("scenario_params", {})),
            revisits=raw.get("revisits"),
            stream_seed=int(raw.get("stream_seed", 42)),
            shuffle_seed=int(raw.get("shuffle_seed", 1993)),
            task_size=raw.get("task_size", "fixed"),
            data_root=str(raw.get("data_root", "./data")),
        )
        section.validate()
        return section

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "init_cls": self.init_cls,
            "increment": self.increment,
            "scenario": self.scenario,
            "scenario_params": self.scenario_params,
            "revisits": self.revisits,
            "stream_seed": self.stream_seed,
            "shuffle_seed": self.shuffle_seed,
            "task_size": self.task_size,
            "data_root": self.data_root,
        }


@dataclass
class MatrixSection:
    """``run-matrix`` sweep config (SPEC §11.1): a methods × scenarios ×
    datasets × seeds grid. clover-cl configs are single-file/self
    -contained (unlike the bench's per-method YAML directory), so
    per-method hyperparameter differences are layered via
    ``method_overrides`` on top of one shared ``training`` block, rather
    than a separate config file per method.
    """

    methods: List[str]
    scenarios: List[str]
    datasets: List[str]
    seeds: List[int]
    stream: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    method_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_dir: str = "runs"

    _ALLOWED_KEYS = frozenset(
        {
            "methods",
            "scenarios",
            "datasets",
            "seeds",
            "stream",
            "training",
            "method_overrides",
            "output_dir",
        }
    )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MatrixSection":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "matrix config")
        for required in ("methods", "scenarios", "datasets", "seeds"):
            if required not in raw:
                raise ValueError(f"matrix config is missing required top-level key {required!r}.")
        return cls(
            methods=list(raw["methods"]),
            scenarios=list(raw["scenarios"]),
            datasets=list(raw["datasets"]),
            seeds=[int(s) for s in raw["seeds"]],
            stream=dict(raw.get("stream", {})),
            training=dict(raw.get("training", {})),
            method_overrides={k: dict(v) for k, v in raw.get("method_overrides", {}).items()},
            output_dir=str(raw.get("output_dir", "runs")),
        )


@dataclass
class MethodSection:
    """``extra`` passes through untyped to ``CLMethod.build(stream_info,
    cfg)`` -- per-method typed schemas (SPEC's "each method declares a typed
    schema") are deferred until a method other than SimpleCIL actually has
    config knobs to validate.
    """

    name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MethodSection":
        if "name" not in raw:
            raise ValueError("method config is missing required key 'name'.")
        extra = {k: v for k, v in raw.items() if k != "name"}
        return cls(name=str(raw["name"]), extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, **self.extra}
