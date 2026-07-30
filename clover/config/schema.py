"""Typed config sections (SPEC §9): plain dataclasses, strict unknown-key
rejection. Layered-defaults + stream resolution happen in ``loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from clover.training.trainer import OPTIMIZER_FACTORIES
from clover.utils.strict_dict import reject_unknown_keys

_AMP_VALUES = frozenset({"none", "bf16", "fp16"})

#: Per-epoch learning-rate schedules (P11-C). ``constant`` means no
#: scheduler at all; ``cosine`` is what 5 of the 9 published method configs
#: use. Kept deliberately small -- a name here must be honoured by
#: ``Trainer``'s scheduler factory, so adding one is a code change, not a
#: config change.
_SCHEDULER_VALUES = frozenset({"constant", "cosine"})
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
    """``weight_decay`` (P11-C) is carried because the published method
    configs disagree on it by two orders of magnitude -- 0.0 for the prompt
    family, 5e-4 for most of the adapter family, 0.05 for SimpleCIL -- so a
    single shared value cannot represent a 9-method comparison."""

    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

    _ALLOWED_KEYS = frozenset({"name", "lr", "weight_decay"})

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "OptimizerConfig":
        reject_unknown_keys(raw, cls._ALLOWED_KEYS, "optimizer config")
        name = str(raw.get("name", "adam"))
        if name not in OPTIMIZER_FACTORIES:
            raise ValueError(
                f"optimizer name must be one of {sorted(OPTIMIZER_FACTORIES)}, got {name!r}."
            )
        weight_decay = float(raw.get("weight_decay", 0.0))
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}.")
        return cls(name=name, lr=float(raw.get("lr", 1e-3)), weight_decay=weight_decay)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "lr": self.lr, "weight_decay": self.weight_decay}


@dataclass
class TrainingSection:
    """``amp`` is validated but only collapsed to a bool for
    ``TrainContext.amp`` right now -- no autocast wrapping exists anywhere
    yet (still true as of P9, even though L2P/DualPrompt/CODA-Prompt are
    real gradient-trained methods since P5: nothing in ``Trainer``/any
    method actually wraps a forward/backward pass in ``torch.autocast``),
    so finer bf16-vs-fp16 dtype handling is deferred to whichever phase
    adds one. ``cudnn_benchmark`` (SPEC §11) is real and set on the
    torch backend at run start (``clover/training/trainer.py
    :_seed_everything``) -- unlike ``amp``, it isn't a no-op.
    """

    epochs: int = 1
    batch_size: int = 32
    optimizer: Optional[OptimizerConfig] = None
    amp: str = "none"
    cudnn_benchmark: bool = False
    scheduler: str = "constant"
    min_lr: float = 0.0

    _ALLOWED_KEYS = frozenset(
        {"epochs", "batch_size", "optimizer", "amp", "cudnn_benchmark", "scheduler", "min_lr"}
    )

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.amp not in _AMP_VALUES:
            raise ValueError(f"amp must be one of {sorted(_AMP_VALUES)}, got {self.amp!r}.")
        if self.scheduler not in _SCHEDULER_VALUES:
            raise ValueError(
                f"scheduler must be one of {sorted(_SCHEDULER_VALUES)}, got {self.scheduler!r}."
            )
        if self.min_lr < 0.0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}.")

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
            cudnn_benchmark=bool(raw.get("cudnn_benchmark", False)),
            scheduler=str(raw.get("scheduler", "constant")),
            min_lr=float(raw.get("min_lr", 0.0)),
        )
        section.validate()
        return section

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer.to_dict() if self.optimizer else None,
            "amp": self.amp,
            "cudnn_benchmark": self.cudnn_benchmark,
            "scheduler": self.scheduler,
            "min_lr": self.min_lr,
        }


#: Keys allowed inside an inline ``dataset: {type: ..., root: ..., num_classes: ...}``
#: mapping (SPEC §7's config-only path, e.g. for ``image_folder``).
_INLINE_DATASET_KEYS = frozenset({"type", "root", "num_classes"})


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
    #: Required for config-only datasets (SPEC §7: ``dataset: {type:
    #: image_folder, root: ..., num_classes: ...}``) whose class count
    #: isn't a hardcoded registry constant. Optional for a named built-in,
    #: which knows its own ``num_classes`` -- there it declares the count
    #: so a config can be *validated offline*, on a machine where that
    #: dataset isn't staged. It never overrides a staged dataset: ``clover
    #: run`` cross-checks it against the data it actually loads.
    dataset_num_classes: Optional[int] = None

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
            "dataset_num_classes",
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

        raw_dataset = raw["dataset"]
        data_root = str(raw.get("data_root", "./data"))
        dataset_num_classes: Optional[int] = None
        if isinstance(raw_dataset, dict):
            reject_unknown_keys(raw_dataset, _INLINE_DATASET_KEYS, "inline dataset config")
            if "type" not in raw_dataset:
                raise ValueError(
                    "inline dataset config is missing required key 'type' "
                    "(e.g. {type: image_folder, root: ..., num_classes: ...})."
                )
            dataset_name = str(raw_dataset["type"])
            if "root" in raw_dataset:
                data_root = str(raw_dataset["root"])
            if "num_classes" in raw_dataset:
                dataset_num_classes = int(raw_dataset["num_classes"])
        else:
            dataset_name = str(raw_dataset)
            # Round-tripping a resolved config: StreamSpec.to_dict() writes
            # dataset_num_classes as a plain top-level key (the inline
            # {type/root/num_classes} mapping is already gone by then).
            if raw.get("dataset_num_classes") is not None:
                dataset_num_classes = int(raw["dataset_num_classes"])

        section = cls(
            dataset=dataset_name,
            init_cls=int(raw["init_cls"]),
            increment=int(raw["increment"]),
            scenario=raw.get("scenario"),
            scenario_params=dict(raw.get("scenario_params", {})),
            revisits=raw.get("revisits"),
            stream_seed=int(raw.get("stream_seed", 42)),
            shuffle_seed=int(raw.get("shuffle_seed", 1993)),
            task_size=raw.get("task_size", "fixed"),
            data_root=data_root,
            dataset_num_classes=dataset_num_classes,
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
            "dataset_num_classes": self.dataset_num_classes,
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
    #: Per-method ``training`` layering (P11-C). ``method_overrides`` reaches
    #: only the *method* section, but the 9 published configs also disagree
    #: on optimizer/lr/batch/schedule -- without this a matrix would run
    #: every method at one shared learning rate, which is not a fair
    #: comparison. Keys are validated per cell by ``TrainingSection``.
    training_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
            "training_overrides",
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
            training_overrides={
                k: dict(v) for k, v in raw.get("training_overrides", {}).items()
            },
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
