"""Full config resolution: scenario vs. inline revisits, smoke profile, round-trip."""

from __future__ import annotations

import pytest

from clover.config.loader import _SMOKE_EPOCHS, _SMOKE_INCREMENT, _SMOKE_INIT_CLS, resolve_config


def _base_config(**stream_overrides):
    stream = {"dataset": "synthetic", "init_cls": 5, "increment": 5}
    stream.update(stream_overrides)
    return {"run": {"seed": 1}, "stream": stream, "method": {"name": "simplecil"}, "training": {}}


def test_scenario_based_stream_resolves_via_registered_factory():
    resolved = resolve_config(_base_config(scenario="exact_replay"))
    assert resolved.stream_spec.dataset == "synthetic"
    assert len(resolved.stream_spec.revisits) == 1
    assert resolved.stream_spec.revisits[0].label == "new"


def test_inline_revisits_stream_resolves_directly():
    raw = _base_config(revisits=[{"classes": "task0", "placement": "end_of_stream", "label": "same"}])
    resolved = resolve_config(raw)
    assert len(resolved.stream_spec.revisits) == 1
    assert resolved.stream_spec.revisits[0].label == "same"


def test_missing_required_top_level_section_rejected():
    raw = _base_config()
    del raw["method"]
    with pytest.raises(ValueError, match="missing required top-level section 'method'"):
        resolve_config(raw)


def test_unknown_top_level_key_rejected():
    raw = _base_config()
    raw["extra_section"] = {}
    with pytest.raises(ValueError, match="unknown top-level config key 'extra_section'"):
        resolve_config(raw)


def test_unknown_scenario_name_raises_actionable_error():
    with pytest.raises(KeyError, match="unknown scenario"):
        resolve_config(_base_config(scenario="exact_repaly"))


def test_unknown_dataset_name_raises_actionable_error():
    raw = _base_config()
    raw["stream"]["dataset"] = "cifar101"
    with pytest.raises(KeyError, match="unknown dataset"):
        resolve_config(raw)


def test_smoke_profile_overrides_dataset_and_keeps_scenario_mechanism():
    raw = _base_config(dataset="synthetic", init_cls=20, increment=20, scenario="exact_replay")
    resolved = resolve_config(raw, smoke=True)

    assert resolved.stream_spec.dataset == "synthetic"
    assert resolved.stream_spec.init_cls == _SMOKE_INIT_CLS
    assert resolved.stream_spec.increment == _SMOKE_INCREMENT
    assert resolved.training.epochs == _SMOKE_EPOCHS
    # scenario mechanism preserved: still an exact_replay-shaped echo revisit
    assert resolved.stream_spec.revisits[0].label == "new"
    assert resolved.stream_spec.revisits[0].images == "same"


def test_smoke_profile_overrides_dataset_even_when_unregistered():
    """A config naming a dataset that doesn't exist should still smoke-run,
    since the override replaces it before any dataset lookup happens."""
    raw = _base_config(dataset="cifar100_does_not_exist_yet")
    resolved = resolve_config(raw, smoke=True)
    assert resolved.stream_spec.dataset == "synthetic"


def test_resolved_config_round_trips_through_save(tmp_path):
    resolved = resolve_config(_base_config(scenario="cumulative_drift"))
    path = tmp_path / "config_resolved.yaml"
    resolved.save(str(path))

    import yaml

    with open(path) as fh:
        reloaded_raw = yaml.safe_load(fh)

    # config_resolved.yaml already has a *resolved* stream (revisits spelled
    # out, no 'scenario' key) -- re-resolving it must reproduce the same spec.
    from clover.config.loader import resolve_config as resolve_again

    reloaded = resolve_again(reloaded_raw)
    assert reloaded.stream_spec.to_dict() == resolved.stream_spec.to_dict()
    assert reloaded.method_name == resolved.method_name
    assert reloaded.training.to_dict() == resolved.training.to_dict()


# --- offline validation of an unstaged dataset (P11-A) ------------------


def _unstaged_config(tmp_path, **stream_overrides):
    """A config naming a real dataset that is definitely not staged here."""
    stream = {
        "dataset": "imagenet_r",
        "init_cls": 20,
        "increment": 20,
        "data_root": str(tmp_path / "nothing-here"),
    }
    stream.update(stream_overrides)
    return {"stream": stream, "method": {"name": "l2p"}, "training": {}}


def test_declared_class_count_resolves_an_unstaged_dataset_offline(tmp_path):
    """Resolution instantiates the dataset purely to read ``num_classes``,
    so before this a config naming an unstaged dataset couldn't be validated
    at all on a machine without that data -- i.e. on every GPU-less dev box,
    which is exactly where configs get written."""
    resolved = resolve_config(_unstaged_config(tmp_path, dataset_num_classes=200))
    assert resolved.stream_spec.dataset == "imagenet_r"
    assert resolved.stream_spec.dataset_num_classes == 200


def test_unstaged_dataset_without_a_declared_count_says_how_to_validate_offline(tmp_path):
    with pytest.raises(FileNotFoundError, match="stream.dataset_num_classes"):
        resolve_config(_unstaged_config(tmp_path))


def test_declaring_the_true_count_changes_nothing_when_the_data_is_present():
    """The offline path must be inert whenever the dataset can actually be
    constructed -- otherwise it would be a second, drifting source of truth
    for a number that lives on the dataset class."""
    without = resolve_config(_base_config())
    with_declaration = resolve_config(_base_config(dataset_num_classes=20))
    assert without.stream_spec.init_cls == with_declaration.stream_spec.init_cls
    assert without.stream_spec.revisits == with_declaration.stream_spec.revisits
    assert without.stream_spec.dataset == with_declaration.stream_spec.dataset


def test_smoke_profile_drops_a_declared_count_for_the_dataset_it_replaced(tmp_path):
    """The smoke profile substitutes the synthetic dataset wholesale, so a
    count declared for the original one no longer describes anything."""
    resolved = resolve_config(_unstaged_config(tmp_path, dataset_num_classes=200), smoke=True)
    assert resolved.stream_spec.dataset == "synthetic"
    assert resolved.stream_spec.dataset_num_classes is None
