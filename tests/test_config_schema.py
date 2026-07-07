"""Config section schemas (SPEC §9): unknown-key rejection, defaults."""

from __future__ import annotations

import pytest

from clover.config.schema import (
    MethodSection,
    OptimizerConfig,
    RunSection,
    StreamSection,
    TrainingSection,
)


def test_run_section_defaults():
    section = RunSection.from_dict({})
    assert section.name is None
    assert section.seed == 1993
    assert section.output_dir == "runs"


def test_run_section_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown run config key 'seeed'.*did you mean 'seed'"):
        RunSection.from_dict({"seeed": 42})


def test_optimizer_config_defaults_and_round_trip():
    section = OptimizerConfig.from_dict({})
    assert section.name == "adam"
    assert section.lr == 1e-3
    assert OptimizerConfig.from_dict(section.to_dict()) == section


def test_optimizer_config_rejects_unknown_name():
    with pytest.raises(ValueError, match="optimizer name must be one of"):
        OptimizerConfig.from_dict({"name": "bogus"})


def test_training_section_defaults():
    section = TrainingSection.from_dict({})
    assert section.epochs == 1
    assert section.batch_size == 32
    assert section.optimizer is None
    assert section.amp == "none"
    assert section.cudnn_benchmark is False


def test_training_section_cudnn_benchmark_round_trips():
    section = TrainingSection.from_dict({"cudnn_benchmark": True})
    assert section.cudnn_benchmark is True
    assert TrainingSection.from_dict(section.to_dict()) == section


def test_training_section_rejects_bad_amp():
    with pytest.raises(ValueError, match="amp must be one of"):
        TrainingSection.from_dict({"amp": "fp64"})


@pytest.mark.parametrize("overrides", [{"epochs": 0}, {"batch_size": 0}])
def test_training_section_rejects_non_positive_fields(overrides):
    with pytest.raises(ValueError):
        TrainingSection.from_dict(overrides)


def test_training_section_nested_optimizer():
    section = TrainingSection.from_dict({"optimizer": {"name": "sgd", "lr": 0.01}})
    assert section.optimizer == OptimizerConfig(name="sgd", lr=0.01)


def test_stream_section_requires_dataset_init_cls_increment():
    with pytest.raises(ValueError, match="missing required key 'dataset'"):
        StreamSection.from_dict({"init_cls": 5, "increment": 5})


def test_stream_section_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown stream config key 'scenaryo'.*did you mean 'scenario'"):
        StreamSection.from_dict({"dataset": "synthetic", "init_cls": 5, "increment": 5, "scenaryo": "x"})


def test_stream_section_rejects_scenario_and_revisits_together():
    with pytest.raises(ValueError, match="cannot set both 'scenario' and 'revisits'"):
        StreamSection.from_dict(
            {
                "dataset": "synthetic",
                "init_cls": 5,
                "increment": 5,
                "scenario": "exact_replay",
                "revisits": [{"classes": "task0"}],
            }
        )


def test_stream_section_rejects_bad_task_size():
    with pytest.raises(ValueError, match="task_size must be one of"):
        StreamSection.from_dict(
            {"dataset": "synthetic", "init_cls": 5, "increment": 5, "task_size": "shrink"}
        )


def test_method_section_requires_name():
    with pytest.raises(ValueError, match="missing required key 'name'"):
        MethodSection.from_dict({})


def test_method_section_passes_extra_keys_through_untyped():
    section = MethodSection.from_dict({"name": "simplecil", "backbone": "tiny_mlp", "foo": 1})
    assert section.name == "simplecil"
    assert section.extra == {"backbone": "tiny_mlp", "foo": 1}
