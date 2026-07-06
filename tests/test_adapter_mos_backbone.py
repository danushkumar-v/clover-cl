"""MOSAdapterViT correctness (SPEC §6.5): one continuously-trained
adapter, EMA-pulled toward the mean of its own frozen history.
"""

from __future__ import annotations

import torch

from clover.backbones import get_backbone
from clover.backbones.adapter_mos import MOSAdapterViT
from clover.backbones.vit import TinyViT


def _base(depth=2):
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


def test_no_snapshot_yet_merge_step_is_a_no_op():
    wrapped = MOSAdapterViT(_base(), bottleneck_dim=4, momentum=0.5)
    before = {k: v.clone() for k, v in wrapped.cur_adapter.state_dict().items()}
    wrapped.merge_step()
    after = wrapped.cur_adapter.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_snapshot_appends_frozen_copy_and_does_not_alter_cur_adapter():
    wrapped = MOSAdapterViT(_base(), bottleneck_dim=4)
    with torch.no_grad():
        for adapter in wrapped.cur_adapter:
            adapter.up_proj.bias.add_(1.0)
    before_cur = {k: v.clone() for k, v in wrapped.cur_adapter.state_dict().items()}

    wrapped.snapshot()
    assert len(wrapped.adapter_list) == 1
    assert all(not p.requires_grad for p in wrapped.adapter_list[0].parameters())
    after_cur = wrapped.cur_adapter.state_dict()
    for key in before_cur:
        assert torch.equal(before_cur[key], after_cur[key])


def test_merge_step_pulls_cur_adapter_toward_snapshot_mean():
    wrapped = MOSAdapterViT(_base(), bottleneck_dim=4, momentum=1.0)  # momentum=1 -> fully snaps to mean
    with torch.no_grad():
        for adapter in wrapped.cur_adapter:
            adapter.up_proj.bias.fill_(5.0)
    wrapped.snapshot()  # adapter_list[0] now has up_proj.bias == 5.0 everywhere

    with torch.no_grad():
        for adapter in wrapped.cur_adapter:
            adapter.up_proj.bias.fill_(-5.0)  # diverge before merging
    wrapped.merge_step()

    for adapter in wrapped.cur_adapter:
        assert torch.allclose(adapter.up_proj.bias, torch.full_like(adapter.up_proj.bias, 5.0))


def test_forward_shape_stacks_history_plus_current():
    wrapped = get_backbone("vit_adapter_mos")(base_model="tiny_vit", input_size=8, depth=2)
    images = torch.randn(3, 8, 8)

    out0 = wrapped(images)
    assert out0.shape == (1, 3, wrapped.feature_dim)  # only cur_adapter so far

    wrapped.snapshot()
    out1 = wrapped(images)
    assert out1.shape == (2, 3, wrapped.feature_dim)  # 1 snapshot + cur_adapter


def test_base_always_frozen():
    wrapped = MOSAdapterViT(_base(), bottleneck_dim=4)
    wrapped.snapshot()
    assert all(not p.requires_grad for p in wrapped.base.parameters())
