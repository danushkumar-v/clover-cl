"""EaseAdapterViT correctness (SPEC §6.5): growing adapter-set list, only
the most recent set stays trainable, concatenated feature width grows.
"""

from __future__ import annotations

import pytest
import torch

from clover.backbones import get_backbone
from clover.backbones.adapter_ease import EaseAdapterViT
from clover.backbones.vit import TinyViT


def _base(depth=2):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_forward_before_grow_raises():
    wrapped = EaseAdapterViT(_base(), bottleneck_dim=4)
    with pytest.raises(RuntimeError, match="grow"):
        wrapped(torch.randn(2, 8, 8))


def test_grow_appends_and_freezes_previous():
    wrapped = EaseAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    assert wrapped.num_blocks == 1
    assert all(p.requires_grad for p in wrapped.cur_adapter.parameters())

    first_adapter = wrapped.cur_adapter
    wrapped.grow()
    assert wrapped.num_blocks == 2
    assert all(not p.requires_grad for p in first_adapter.parameters())
    assert all(p.requires_grad for p in wrapped.cur_adapter.parameters())
    assert wrapped.cur_adapter is not first_adapter


def test_base_always_frozen():
    wrapped = EaseAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    wrapped.grow()
    assert all(not p.requires_grad for p in wrapped.base.parameters())


def test_forward_shape_grows_with_num_blocks():
    wrapped = get_backbone("vit_adapter_ease")(base_model="tiny_vit", input_size=8, depth=2)
    images = torch.randn(3, 8, 8)

    wrapped.grow()
    out1 = wrapped(images)
    assert out1.shape == (3, 1, wrapped.feature_dim)

    wrapped.grow()
    out2 = wrapped(images)
    assert out2.shape == (3, 2, wrapped.feature_dim)


def test_forward_current_uses_only_the_last_adapter_set():
    wrapped = EaseAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    images = torch.randn(2, 8, 8)
    with torch.no_grad():
        current = wrapped.forward_current(images)
        full = wrapped(images)
    assert torch.allclose(current, full[:, 0])
