"""Scale-derived adapter defaults (P11-B2, Task 1): published PILOT
hyperparameter *values* (``bench/configs/methods/*.yaml``, read-only
reference, never imported/copied -- CLAUDE.md) recovered exactly at
ViT-B/16 scale (``depth=12``, ``feature_dim=768``), with ``TinyViT``-scale
behaviour (``depth=2``, ``feature_dim=16``) numerically unchanged from
before this rule existed, so the existing safety gate stays green without
retuning.

Every rule here is "derive from the base's own ``feature_dim``, floored at
the previous TinyViT-tuned literal" -- see
``clover/backbones/adapter.py:default_bottleneck_dim`` and
``clover/methods/ranpac.py:_default_projection_dim`` for the derivation
itself; this file only pins the resulting numbers so a future change to
either function has to consciously break a test, not silently drift.
"""

from __future__ import annotations

import pytest

from clover.backbones import get_backbone
from clover.backbones.adapter import (
    APER_EASE_RANPAC_BOTTLENECK_RATIO,
    MOS_TUNA_BOTTLENECK_RATIO,
    AdapterViT,
    default_bottleneck_dim,
)
from clover.backbones.adapter_ease import EaseAdapterViT
from clover.backbones.adapter_mos import MOSAdapterViT
from clover.backbones.adapter_tuna import TunaAdapterViT
from clover.backbones.loader import resolve_base_model
from clover.backbones.vit import TinyViT
from clover.methods import get_method
from clover.methods.base import StreamInfo
from clover.methods.ranpac import _default_projection_dim

# vit_base_patch16_224 architecture-only construction takes a few seconds;
# built once per module and reused (read-only structural checks) across
# every ViT-B/16-scale test below rather than once per test.
_VIT_B16_NAME = "vit_base_patch16_224"


@pytest.fixture(scope="module")
def real_vit_b16():
    return resolve_base_model(_VIT_B16_NAME, pretrained=False)


def _tiny_base(depth: int = 2) -> TinyViT:
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


# --- default_bottleneck_dim: the rule itself ----------------------------


def test_bottleneck_dim_matches_published_value_at_vitb16_scale():
    # aper_adapter/ranpac/ease: PILOT ffn_num=64 at feature_dim=768
    # (bench/configs/methods/{aper_adapter,ranpac,ease}.yaml).
    assert default_bottleneck_dim(768, APER_EASE_RANPAC_BOTTLENECK_RATIO) == 64
    # mos/tuna: PILOT ffn_num=16 at feature_dim=768 (mos.yaml; tuna's
    # bottleneck is hardcoded to 16 upstream -- see docs/methods.md).
    assert default_bottleneck_dim(768, MOS_TUNA_BOTTLENECK_RATIO) == 16


def test_bottleneck_dim_unchanged_at_tinyvit_scale():
    # Both ratios floor to the same literal (8) this project's TinyViT
    # safety gate was already tuned against -- zero behavior change for the
    # existing 6-scenario accuracy gate.
    assert default_bottleneck_dim(16, APER_EASE_RANPAC_BOTTLENECK_RATIO) == 8
    assert default_bottleneck_dim(16, MOS_TUNA_BOTTLENECK_RATIO) == 8


# --- end-to-end: each backbone wrapper wires the default through -------


def test_aper_ranpac_backbone_bottleneck_at_both_scales(real_vit_b16):
    tiny = AdapterViT(_tiny_base())
    assert tiny.adapters[0].down_proj.out_features == 8

    real = AdapterViT(real_vit_b16)
    assert real.adapters[0].down_proj.out_features == 64
    assert len(real.adapters) == 12  # one adapter per transformer block


def test_ease_backbone_bottleneck_at_both_scales(real_vit_b16):
    tiny = EaseAdapterViT(_tiny_base())
    tiny.grow()
    assert tiny.cur_adapter[0].down_proj.out_features == 8

    real = EaseAdapterViT(real_vit_b16)
    real.grow()
    assert real.cur_adapter[0].down_proj.out_features == 64
    assert len(real.cur_adapter) == 12


def test_mos_backbone_bottleneck_at_both_scales(real_vit_b16):
    tiny = MOSAdapterViT(_tiny_base())
    assert tiny.cur_adapter[0].down_proj.out_features == 8

    real = MOSAdapterViT(real_vit_b16)
    assert real.cur_adapter[0].down_proj.out_features == 16
    assert len(real.cur_adapter) == 12


def test_tuna_backbone_bottleneck_at_both_scales(real_vit_b16):
    tiny = TunaAdapterViT(_tiny_base())
    tiny.grow()
    assert tiny.cur_adapter[0].down_proj.out_features == 8

    real = TunaAdapterViT(real_vit_b16)
    real.grow()
    assert real.cur_adapter[0].down_proj.out_features == 16
    assert len(real.cur_adapter) == 12


def test_vit_adapter_factory_default_is_scale_derived():
    """The registered ``vit_adapter`` factory (used by APER-Adapter/RanPAC
    via ``default_backbone``) forwards ``bottleneck_dim=None`` through to
    ``AdapterViT`` rather than hardcoding a literal."""
    tiny = get_backbone("vit_adapter")(base_model="tiny_vit", input_size=8, depth=2)
    assert tiny.adapters[0].down_proj.out_features == 8


# --- RanPAC projection width --------------------------------------------


def test_ranpac_projection_dim_default_at_tinyvit_scale():
    method = get_method("ranpac")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8, channels=1)
    method.build(info, {"depth": 2})
    assert method.projection_dim == 256  # unchanged from before this rule existed


def test_ranpac_projection_dim_default_at_vitb16_scale():
    method = get_method("ranpac")()
    info = StreamInfo(
        dataset="synthetic", nb_experiences=2, total_classes=4, input_size=224, channels=3
    )
    method.build(info, {"base_model": _VIT_B16_NAME, "pretrained": False})
    assert method.projection_dim == 10000  # ranpac.yaml: M=10000


def test_ranpac_projection_dim_explicit_value_is_not_overridden():
    method = get_method("ranpac")(projection_dim=64)
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8, channels=1)
    method.build(info, {"depth": 2})
    assert method.projection_dim == 64


def test_default_projection_dim_formula():
    assert _default_projection_dim(768) == 10000
    assert _default_projection_dim(16) == 256
