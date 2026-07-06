"""P0 gate: the v2 skeleton imports cleanly and registries behave correctly."""

from __future__ import annotations

import pytest

import clover
from clover.backbones import get_backbone, list_backbones, register_backbone
from clover.datasets import get_dataset, list_datasets, register_dataset
from clover.methods import get_method, list_methods, register_method
from clover.scenarios import get_scenario, list_scenarios, register_scenario
from clover.utils.registry import Registry


def test_package_imports() -> None:
    assert clover.__version__ == "2.0.0.dev0"


def test_methods_registry_has_all_9_methods() -> None:
    assert set(list_methods()) == {
        "simplecil",
        "l2p",
        "dualprompt",
        "coda_prompt",
        "aper_adapter",
        "ranpac",
        "ease",
        "mos",
        "tuna",
    }


def test_scenario_registry_has_the_expected_p1_p8_scenarios() -> None:
    assert set(list_scenarios()) == {
        "disjoint_baseline",
        "exact_replay",
        "partial_overlap",
        "long_range_revisit",
        "mid_range_revisit",
        "cumulative_drift",
        "distribution_shift",
    }


def test_dataset_registry_has_the_expected_p2_p8_builtins() -> None:
    assert set(list_datasets()) == {
        "synthetic",
        "cifar100",
        "cub200",
        "imagenet_r",
        "imagenet_a",
        "omnibenchmark",
        "vtab",
        "image_folder",
    }


def test_backbone_registry_has_the_expected_p2_p6_builtins() -> None:
    assert set(list_backbones()) == {
        "tiny_mlp",
        "tiny_vit",
        "vit_prompt_pool",
        "vit_dual_prompt",
        "vit_coda_prompt",
        "vit_adapter",
        "vit_adapter_ease",
        "vit_adapter_mos",
        "vit_adapter_tuna",
    }


@pytest.mark.parametrize(
    ("register_fn", "get_fn"),
    [
        (register_dataset, get_dataset),
        (register_scenario, get_scenario),
        (register_method, get_method),
        (register_backbone, get_backbone),
    ],
)
def test_registry_round_trip(register_fn, get_fn) -> None:
    @register_fn("dummy")
    class Dummy:
        pass

    assert get_fn("dummy") is Dummy


def test_registry_unknown_name_is_actionable() -> None:
    registry: Registry = Registry("widget")
    registry.register("gizmo")(object())

    with pytest.raises(KeyError, match="unknown widget 'gizmo2'.*did you mean 'gizmo'"):
        registry.get("gizmo2")


def test_registry_duplicate_name_rejected() -> None:
    registry: Registry = Registry("widget")
    registry.register("gizmo")(object())

    with pytest.raises(ValueError, match="widget 'gizmo' is already registered"):
        registry.register("gizmo")(object())
