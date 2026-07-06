"""TunaAdapterViT / EMR-merge correctness (SPEC §6.5)."""

from __future__ import annotations

import torch

from clover.backbones import get_backbone
from clover.backbones.adapter_tuna import TunaAdapterViT, _emr_merge
from clover.backbones.vit import TinyViT


def _base(depth=2):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_emr_merge_single_task_is_identity():
    stacked = torch.tensor([[1.0, -2.0, 3.0, 0.0]])
    assert torch.allclose(_emr_merge(stacked), stacked[0])


def test_emr_merge_keeps_max_magnitude_among_agreeing_signs():
    # All three agree on sign (+) for element 0 -> keep the max magnitude (3.0),
    # rescaled; element 1 has a 2-vs-1 majority (+) -> disagreeing -1 is zeroed.
    stacked = torch.tensor([[1.0, 2.0], [3.0, 4.0], [2.0, -1.0]])
    merged = _emr_merge(stacked)
    assert merged[0] > 0
    assert merged[1] > 0  # majority sign wins, disagreeing entry excluded

    mean_abs = stacked.abs().mean()
    assert torch.isclose(merged.abs().mean(), mean_abs, atol=1e-5)


def test_emr_merge_zero_vector_stays_zero():
    stacked = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])  # perfect sign cancellation
    merged = _emr_merge(stacked)
    # sign_sum == 0 everywhere -> falls back to mean's sign, which is also 0
    # here -> elected_sign == 0 -> agree is False for both entries -> merged == 0.
    assert torch.allclose(merged, torch.zeros(2))


def test_grow_freezes_previous_and_allocates_fresh():
    wrapped = TunaAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    assert len(wrapped.adapter_list) == 1
    assert all(p.requires_grad for p in wrapped.cur_adapter.parameters())

    first_adapter = wrapped.cur_adapter
    wrapped.grow()
    assert len(wrapped.adapter_list) == 2
    assert all(not p.requires_grad for p in first_adapter.parameters())
    assert all(p.requires_grad for p in wrapped.cur_adapter.parameters())
    assert wrapped.cur_adapter is not first_adapter


def test_base_always_frozen():
    wrapped = TunaAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    assert all(not p.requires_grad for p in wrapped.base.parameters())


def test_forward_uses_merged_adapter_not_current():
    wrapped = TunaAdapterViT(_base(), bottleneck_dim=4)
    wrapped.grow()
    with torch.no_grad():
        for adapter in wrapped.cur_adapter:
            adapter.up_proj.bias.add_(torch.randn_like(adapter.up_proj.bias))

    images = torch.randn(2, 8, 8)
    with torch.no_grad():
        current_out = wrapped.forward_current(images)
        merged_out = wrapped(images)
    # merged_adapter is still all-zero (recompute_merge hasn't been called
    # yet) -- forward() must not silently fall back to cur_adapter.
    assert not torch.allclose(current_out, merged_out)


def test_recompute_merge_updates_merged_adapter_and_forward_shape():
    wrapped = get_backbone("vit_adapter_tuna")(base_model="tiny_vit", input_size=8, depth=2)
    images = torch.randn(3, 8, 8)

    wrapped.grow()
    with torch.no_grad():
        for adapter in wrapped.cur_adapter:
            adapter.up_proj.bias.add_(torch.randn_like(adapter.up_proj.bias))
    wrapped.recompute_merge()

    out = wrapped(images)
    assert out.shape == (3, wrapped.feature_dim)

    merged_state = {k: v.clone() for k, v in wrapped.merged_adapter.state_dict().items()}
    assert any(not torch.equal(v, torch.zeros_like(v)) for v in merged_state.values())
