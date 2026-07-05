"""PromptPool / PromptPoolViT correctness (SPEC §6.5, L2P's mechanism)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from clover.backbones import get_backbone
from clover.backbones.prompt_pool import PromptPool, PromptPoolViT
from clover.backbones.vit import TinyViT


def test_select_picks_the_closest_keys_by_cosine_similarity():
    pool = PromptPool(pool_size=4, prompt_length=2, embed_dim=3)
    with torch.no_grad():
        pool.key.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
            )
        )
        pool.prompt.copy_(torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3))

    query = torch.tensor([[1.0, 0.0, 0.0]])  # closest to key 0, then key 1 or 2 (tie), then key 3 is opposite
    selected, similarity = pool.select(query, top_k=2)

    assert selected.shape == (1, 2 * 2, 3)  # top_k * prompt_length tokens
    assert similarity[0, 0].item() == 1.0  # exact match with key 0, sorted first
    # key 0's prompt (index 0) must be the first block in the selection
    assert torch.equal(selected[0, :2], pool.prompt[0])


def test_prompt_pool_vit_freezes_base_and_leaves_pool_trainable():
    base = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=1, num_heads=2)
    wrapped = PromptPoolViT(base, pool_size=6, prompt_length=3, top_k=2)

    assert all(not p.requires_grad for p in wrapped.base.parameters())
    assert all(p.requires_grad for p in wrapped.pool.parameters())


def test_prompt_pool_vit_forward_shape():
    wrapped = get_backbone("vit_prompt_pool")(base_model="tiny_vit", input_size=8, pool_size=6, top_k=2)
    images = torch.randn(4, 8, 8)
    out = wrapped(images)
    assert out.shape == (4, wrapped.feature_dim)


def test_prompt_pool_vit_selection_depends_on_input():
    wrapped = get_backbone("vit_prompt_pool")(base_model="tiny_vit", input_size=8, pool_size=8, top_k=3)
    images_a = torch.randn(1, 8, 8)
    images_b = torch.randn(1, 8, 8) * 5 + 3  # a very different input distribution

    query_a = wrapped.base.query_features(images_a)
    query_b = wrapped.base.query_features(images_b)
    sim_a = F.normalize(query_a, dim=-1) @ F.normalize(wrapped.pool.key, dim=-1).t()
    sim_b = F.normalize(query_b, dim=-1) @ F.normalize(wrapped.pool.key, dim=-1).t()
    assert not torch.allclose(sim_a, sim_b)
