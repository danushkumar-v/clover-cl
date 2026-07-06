"""Adapter / AdapterViT correctness (SPEC §6.5): zero-init no-op at
construction, frozen base + trainable adapters, same-weights plain vs.
adapted pass.
"""

from __future__ import annotations

import torch

from clover.backbones import get_backbone
from clover.backbones.adapter import Adapter, AdapterViT
from clover.backbones.vit import TinyViT


def _base(depth=2):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_adapter_is_zero_init_no_op():
    adapter = Adapter(embed_dim=16, bottleneck_dim=4)
    x = torch.randn(3, 5, 16)
    out = adapter(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_base_is_frozen_adapters_are_not():
    wrapped = AdapterViT(_base(), bottleneck_dim=4)
    assert all(not p.requires_grad for p in wrapped.base.parameters())
    assert all(p.requires_grad for p in wrapped.adapters.parameters())


def test_forward_shape():
    wrapped = get_backbone("vit_adapter")(base_model="tiny_vit", input_size=8, depth=2)
    images = torch.randn(4, 8, 8)
    out = wrapped(images)
    assert out.shape == (4, wrapped.feature_dim)


def test_adapter_pass_diverges_from_plain_pass_once_trained():
    wrapped = AdapterViT(_base(), bottleneck_dim=4)
    images = torch.randn(3, 8, 8)
    with torch.no_grad():
        plain = wrapped.base(images)
        adapted_before_training = wrapped(images)
    # Zero-initialized up-proj -> adapter branch starts as a true no-op:
    # the adapted pass matches the plain pass exactly until trained.
    assert torch.allclose(plain, adapted_before_training)

    with torch.no_grad():
        for module in wrapped.adapters:
            # Perturb the bias with per-channel-varying noise, not a single
            # scalar added to every element: TinyViT's final LayerNorm
            # mean-centers each token, which would near-exactly cancel a
            # channel-uniform shift (every element getting the *same*
            # constant) regardless of how many residual blocks it passed
            # through -- an early version of this test added a uniform
            # scalar and got a false negative for exactly this reason.
            module.up_proj.bias.add_(torch.randn_like(module.up_proj.bias))
    with torch.no_grad():
        adapted_after_training = wrapped(images)
    assert not torch.allclose(plain, adapted_after_training)
