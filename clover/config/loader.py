"""Config loading + resolution (SPEC §9).

One run = one file. Unknown top-level keys are errors; the fully resolved
config (after scenario resolution) is what gets written to the run dir as
``config_resolved.yaml`` so a run can be reproduced exactly from it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import yaml

from clover.config.schema import MethodSection, RunSection, StreamSection, TrainingSection
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.datasets import get_dataset
from clover.scenarios import get_scenario
from clover.utils.strict_dict import reject_unknown_keys

_TOP_LEVEL_KEYS = frozenset({"run", "stream", "method", "training"})

#: Smoke profile (SPEC §9): CPU-only, minutes, keeps the chosen method and
#: scenario mechanism, only the dataset and the concrete numbers change.
_SMOKE_DATASET = "synthetic"
_SMOKE_INIT_CLS = 4
_SMOKE_INCREMENT = 4
_SMOKE_EPOCHS = 1
_SMOKE_BATCH_SIZE = 8


@dataclass
class ResolvedConfig:
    run: RunSection
    stream_spec: StreamSpec
    method_name: str
    method_cfg: Dict[str, Any]
    training: TrainingSection

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "stream": self.stream_spec.to_dict(),
            "method": {"name": self.method_name, **self.method_cfg},
            "training": self.training.to_dict(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}.")
    return data


def resolve_dataset_num_classes(
    dataset: str, data_root: str = "./data", declared: Optional[int] = None
) -> int:
    """Class count for a configured dataset, without requiring its data.

    ``stream.dataset_num_classes`` was introduced for config-only datasets
    (SPEC §7: ``image_folder``), which can't be constructed at all without
    it. It doubles as the **offline-validation** path for named datasets:
    resolving a config otherwise instantiates the dataset purely to read
    ``num_classes``, so a config naming an unstaged dataset (``imagenet_r``
    on a laptop that only submits cluster jobs) couldn't be validated at
    all. Declaring the count keeps ``clover preflight``/``inspect`` usable.

    The declaration stands in for absent data; it is never an override.
    ``clover run`` re-reads the count from the dataset it actually loads
    and rejects a mismatch, so a stale declaration can't silently reshape
    a real run's stream.

    Args:
        dataset: Registered dataset name.
        data_root: Where the data would be staged.
        declared: ``stream.dataset_num_classes``, if the config set it.

    Returns:
        The number of classes in the dataset.

    Raises:
        FileNotFoundError: Data isn't staged and no count was declared.
        ValueError: The dataset needs a declared count and none was given.
    """
    if declared is not None:
        return int(declared)
    dataset_cls = get_dataset(dataset)
    try:
        return int(dataset_cls(root=data_root).num_classes)
    except (FileNotFoundError, ValueError) as exc:
        raise type(exc)(
            f"{exc} To validate this config without the data staged locally, "
            "declare the class count as stream.dataset_num_classes."
        ) from None


def _resolve_stream_spec(stream: StreamSection) -> StreamSpec:
    num_classes = resolve_dataset_num_classes(
        stream.dataset, stream.data_root, stream.dataset_num_classes
    )
    info = DatasetInfo(stream.dataset, num_classes)

    if stream.scenario is not None:
        factory = get_scenario(stream.scenario)
        spec = factory(
            info, stream.init_cls, stream.increment, stream.stream_seed, **stream.scenario_params
        )
        # Scenario factories don't know about data_root/dataset_num_classes
        # (they only see dataset name + class count via DatasetInfo) --
        # apply the user's actual config on top of whatever they returned.
        return replace(spec, data_root=stream.data_root, dataset_num_classes=stream.dataset_num_classes)

    revisits = [RevisitSpec.from_dict(r) for r in (stream.revisits or [])]
    return StreamSpec(
        dataset=stream.dataset,
        init_cls=stream.init_cls,
        increment=stream.increment,
        task_size=stream.task_size,
        revisits=revisits,
        shuffle_seed=stream.shuffle_seed,
        stream_seed=stream.stream_seed,
        data_root=stream.data_root,
        dataset_num_classes=stream.dataset_num_classes,
    )


def resolve_config(raw: Dict[str, Any], smoke: bool = False) -> ResolvedConfig:
    """Parse + validate every section and resolve the stream into a
    concrete ``StreamSpec``.

    Args:
        raw: The raw config dict (as loaded from YAML).
        smoke: If ``True``, override onto the built-in synthetic dataset
            with a minimal training budget (SPEC §9) -- applied *before*
            stream resolution so a named ``scenario`` is re-invoked with
            the smoke-sized numbers rather than reusing stale resolved
            revisits computed for the original dataset.
    """
    reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "top-level config")
    for required in ("stream", "method"):
        if required not in raw:
            raise ValueError(f"config is missing required top-level section {required!r}.")

    run = RunSection.from_dict(raw.get("run", {}))
    stream = StreamSection.from_dict(raw["stream"])
    method = MethodSection.from_dict(raw["method"])
    training = TrainingSection.from_dict(raw.get("training", {}))

    if smoke:
        stream = replace(
            stream,
            dataset=_SMOKE_DATASET,
            init_cls=_SMOKE_INIT_CLS,
            increment=_SMOKE_INCREMENT,
            # The declared count described the *replaced* dataset. Carrying
            # it over would describe synthetic with imagenet_r's 200 classes.
            dataset_num_classes=None,
        )
        training = replace(training, epochs=_SMOKE_EPOCHS, batch_size=_SMOKE_BATCH_SIZE)

    stream_spec = _resolve_stream_spec(stream)

    return ResolvedConfig(
        run=run,
        stream_spec=stream_spec,
        method_name=method.name,
        method_cfg=method.extra,
        training=training,
    )
