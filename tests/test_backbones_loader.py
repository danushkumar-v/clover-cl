"""Base-model resolution (SPEC §6.4): registry-first, timm fallback, user class."""

from __future__ import annotations

import pytest
import torch.nn as nn

from clover.backbones.loader import resolve_base_model
from clover.backbones.vit import TinyViT


def test_resolves_registered_backbone_by_name():
    model = resolve_base_model("tiny_vit", input_size=8)
    assert isinstance(model, TinyViT)


def test_resolves_timm_model_when_not_in_registry():
    model = resolve_base_model("vit_tiny_patch16_224")
    assert isinstance(model, nn.Module)
    assert hasattr(model, "blocks")  # standard timm ViT structure


def test_forced_timm_source_bypasses_registry():
    # "tiny_vit" IS registered, but source="timm" must not consult the registry.
    with pytest.raises(Exception):
        resolve_base_model("tiny_vit", source="timm")


def test_class_source_imports_and_constructs():
    model = resolve_base_model("clover.backbones.vit:TinyViT", source="class", input_size=8)
    assert isinstance(model, TinyViT)


def test_class_source_requires_colon_separated_path():
    with pytest.raises(ValueError, match="module.path:ClassName"):
        resolve_base_model("clover.backbones.vit.TinyViT", source="class")


def test_rejects_unknown_source():
    with pytest.raises(ValueError, match="source must be"):
        resolve_base_model("tiny_vit", source="bogus")
