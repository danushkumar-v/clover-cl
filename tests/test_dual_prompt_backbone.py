"""DualPromptViT correctness (SPEC §6.5): general/expert prefix-KV injection
at the right layers, frozen base, disjoint layer validation.
"""

from __future__ import annotations

import pytest
import torch

from clover.backbones import get_backbone
from clover.backbones.dual_prompt import DualPromptViT
from clover.backbones.vit import TinyViT


def _base(depth=3):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_rejects_overlapping_g_and_e_layers():
    with pytest.raises(ValueError, match="must be disjoint"):
        DualPromptViT(_base(), g_layers=(0, 1), e_layers=(1,))


def test_rejects_layer_index_beyond_base_depth():
    with pytest.raises(ValueError, match="only has"):
        DualPromptViT(_base(depth=2), g_layers=(0,), e_layers=(5,))


def test_base_is_frozen_prompts_are_not():
    wrapped = DualPromptViT(_base(), g_layers=(0,), e_layers=(1,))
    assert all(not p.requires_grad for p in wrapped.base.parameters())
    assert wrapped.g_prompt.requires_grad
    assert wrapped.e_pool.requires_grad
    assert wrapped.e_key.requires_grad


def test_forward_shape():
    wrapped = get_backbone("vit_dual_prompt")(base_model="tiny_vit", input_size=8, depth=3)
    images = torch.randn(4, 8, 8)
    out = wrapped(images)
    assert out.shape == (4, wrapped.feature_dim)


def test_general_prompt_actually_changes_output_when_perturbed():
    wrapped = DualPromptViT(_base(), g_layers=(0,), e_layers=(1,), top_k=1)
    images = torch.randn(3, 8, 8)
    with torch.no_grad():
        out_before = wrapped(images)
        wrapped.g_prompt.add_(1.0)
        out_after = wrapped(images)
    assert not torch.allclose(out_before, out_after)


def test_expert_selection_depends_on_input():
    wrapped = DualPromptViT(_base(), g_layers=(0,), e_layers=(1,), pool_size=8, top_k=2)
    query_a = wrapped.base.query_features(torch.randn(1, 8, 8))
    query_b = wrapped.base.query_features(torch.randn(1, 8, 8) * 5 + 3)
    experts_a = wrapped._select_experts(query_a)
    experts_b = wrapped._select_experts(query_b)
    assert not torch.allclose(experts_a, experts_b)


def test_to_prefix_reshape_matches_attention_expectations():
    wrapped = DualPromptViT(_base(), g_layers=(0,), e_layers=(1,))
    x = torch.randn(2, 5, wrapped.feature_dim)
    prefixed = wrapped._to_prefix(x)
    assert prefixed.shape == (2, wrapped.num_heads, 5, wrapped.head_dim)
