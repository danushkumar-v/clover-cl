"""TinyViT correctness (SPEC §6.4): shapes, and the prefix-KV hook actually
changes attention output (unused by L2P but built for DualPrompt/CODA-Prompt).
"""

from __future__ import annotations

import pytest
import torch

from clover.backbones import get_backbone
from clover.backbones.vit import Attention, TinyViT


def test_registered_as_tiny_vit():
    assert get_backbone("tiny_vit") is TinyViT


def test_forward_shape_matches_feature_dim():
    vit = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=2, num_heads=2)
    images = torch.randn(3, 8, 8)
    out = vit(images)
    assert out.shape == (3, 16)


def test_patch_tokens_count():
    vit = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=1, num_heads=2)
    images = torch.randn(2, 8, 8)
    tokens = vit.patch_tokens(images)
    assert tokens.shape == (2, 4, 16)  # (8/4)^2 = 4 patches


def test_query_features_no_grad_and_correct_shape():
    vit = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=1, num_heads=2)
    images = torch.randn(2, 8, 8, requires_grad=True)
    query = vit.query_features(images)
    assert query.shape == (2, 16)
    assert not query.requires_grad


def test_prompt_tokens_prepended_shifts_cls_index():
    vit = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=1, num_heads=2)
    images = torch.randn(2, 8, 8)
    prompt_tokens = torch.randn(2, 3, 16)
    out_with_prompt = vit(images, prompt_tokens=prompt_tokens)
    out_without = vit(images)
    assert out_with_prompt.shape == out_without.shape == (2, 16)
    # different token sequence (prompts shift everything through self-attention) -> different output
    assert not torch.allclose(out_with_prompt, out_without)


def test_prefix_kv_hook_changes_attention_output():
    embed_dim, num_heads = 16, 2
    head_dim = embed_dim // num_heads
    attn = Attention(embed_dim, num_heads)
    x = torch.randn(2, 5, embed_dim)

    out_no_prefix = attn(x)

    prefix_len = 3
    prefix_k = torch.randn(2, num_heads, prefix_len, head_dim)
    prefix_v = torch.randn(2, num_heads, prefix_len, head_dim)
    out_with_prefix = attn(x, prefix_kv=(prefix_k, prefix_v))

    assert out_with_prefix.shape == out_no_prefix.shape
    assert not torch.allclose(out_with_prefix, out_no_prefix)


def test_forward_tokens_prefix_kv_routes_to_correct_block():
    vit = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=2, num_heads=2)
    images = torch.randn(2, 8, 8)
    tokens = vit.patch_tokens(images)
    cls = vit.cls_token.expand(2, -1, -1)
    seq = torch.cat([cls, tokens], dim=1) + vit.pos_embed

    out_plain = vit.forward_tokens(seq)

    head_dim = vit.blocks[0].attn.head_dim
    num_heads = vit.blocks[0].attn.num_heads
    prefix_k = torch.randn(2, num_heads, 2, head_dim)
    prefix_v = torch.randn(2, num_heads, 2, head_dim)
    out_block0_prefixed = vit.forward_tokens(seq, prefix_kv={0: (prefix_k, prefix_v)})

    assert not torch.allclose(out_plain, out_block0_prefixed)


def test_rejects_input_size_not_divisible_by_patch_size():
    with pytest.raises(ValueError, match="must be divisible by"):
        TinyViT(input_size=8, patch_size=3)
