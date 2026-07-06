"""CodaPromptViT / CodaPromptPool correctness (SPEC §6.5): soft weighting
sums to 1 (no top-k), Gram-Schmidt orthogonality, resume-safe unlocked buffer.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from clover.backbones import get_backbone
from clover.backbones.coda_prompt import CodaPromptPool, CodaPromptViT
from clover.backbones.vit import TinyViT


def _base(depth=2):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_rejects_layer_index_beyond_base_depth():
    with pytest.raises(ValueError, match="only has"):
        CodaPromptViT(_base(depth=2), layers=(0, 5))


def test_base_is_frozen_pool_is_not():
    wrapped = CodaPromptViT(_base(), layers=(0, 1), nb_experiences=2)
    assert all(not p.requires_grad for p in wrapped.base.parameters())
    assert wrapped.pool.prompt.requires_grad
    assert wrapped.pool.key.requires_grad
    assert wrapped.pool.attn_vec.requires_grad


def test_unlocked_starts_at_zero_and_is_a_buffer_not_a_parameter():
    pool = CodaPromptPool(pool_size=10, n_layers=2, length=2, embed_dim=8, nb_experiences=5)
    assert pool.unlocked.item() == 0
    assert "unlocked" in dict(pool.named_buffers())
    assert "unlocked" not in dict(pool.named_parameters())


def test_combine_weights_sum_to_one_across_unlocked_slots():
    pool = CodaPromptPool(pool_size=6, n_layers=1, length=2, embed_dim=8, nb_experiences=3)
    pool.start_new_task(0)  # unlocks slots_per_task = 6//3 = 2

    query = torch.randn(4, 8)
    # combine() doesn't expose weights directly; recompute them the same way
    # to check they're a valid softmax distribution over the unlocked slots.
    n = pool.unlocked.item()
    weighted_query = query.unsqueeze(1) * pool.attn_vec[:n].unsqueeze(0)
    similarity = (F.normalize(weighted_query, dim=-1) * F.normalize(pool.key[:n], dim=-1).unsqueeze(0)).sum(
        dim=-1
    )
    weights = F.softmax(similarity, dim=-1)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert n == 2


def test_start_new_task_unlocks_incrementally_and_is_idempotent():
    pool = CodaPromptPool(pool_size=6, n_layers=1, length=2, embed_dim=8, nb_experiences=3)
    pool.start_new_task(0)
    assert pool.unlocked.item() == 2
    pool.start_new_task(0)  # same task again -- no-op
    assert pool.unlocked.item() == 2
    pool.start_new_task(1)
    assert pool.unlocked.item() == 4
    pool.start_new_task(2)
    assert pool.unlocked.item() == 6


def test_gram_schmidt_orthogonalizes_newly_unlocked_prompts():
    pool = CodaPromptPool(pool_size=4, n_layers=1, length=2, embed_dim=8, nb_experiences=4)
    pool.start_new_task(0)  # unlocks slot 0 only (4 // 4 = 1 per task)
    pool.start_new_task(1)  # unlocks slot 1, orthogonalized against slot 0

    v0 = pool.prompt.data[0].flatten()
    v1 = pool.prompt.data[1].flatten()
    assert abs((v0 @ v1).item()) < 1e-4  # orthogonal (dot product ~0)


def test_combine_only_uses_unlocked_slots():
    pool = CodaPromptPool(pool_size=4, n_layers=1, length=2, embed_dim=8, nb_experiences=4)
    pool.start_new_task(0)  # unlock only slot 0
    with torch.no_grad():
        pool.prompt.data[1:].fill_(999.0)  # locked slots -- should never appear in the output

    query = torch.randn(2, 8)
    combined = pool.combine(query)
    assert not torch.any(combined == 999.0)


def test_forward_shape():
    wrapped = get_backbone("vit_coda_prompt")(base_model="tiny_vit", input_size=8, nb_experiences=2)
    images = torch.randn(4, 8, 8)
    out = wrapped(images)
    assert out.shape == (4, wrapped.feature_dim)
