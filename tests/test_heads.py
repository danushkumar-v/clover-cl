"""IncrementalHead correctness (SPEC §6.2)."""

from __future__ import annotations

import pytest
import torch

from clover.methods.heads import IncrementalHead


def test_expand_to_grows_width_and_preserves_old_rows():
    head = IncrementalHead(feature_dim=4, num_classes=2)
    with torch.no_grad():
        head.weight[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        head.weight[1] = torch.tensor([5.0, 6.0, 7.0, 8.0])

    head.expand_to(5)

    assert head.num_classes == 5
    assert torch.equal(head.weight[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(head.weight[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))


def test_expand_to_is_idempotent_when_already_wide_enough():
    head = IncrementalHead(feature_dim=4, num_classes=5)
    original = head.weight.clone()
    head.expand_to(3)  # smaller -- no-op
    head.expand_to(5)  # equal -- no-op
    assert torch.equal(head.weight, original)


def test_set_prototype_assigns_row_exactly():
    head = IncrementalHead(feature_dim=3, num_classes=2)
    vector = torch.tensor([1.0, 2.0, 3.0])
    head.set_prototype(1, vector)
    assert torch.equal(head.weight[1], vector)
    assert torch.equal(head.weight[0], torch.zeros(3))


def test_set_prototype_rejects_out_of_range_class_id():
    head = IncrementalHead(feature_dim=3, num_classes=2)
    with pytest.raises(ValueError, match="exceeds current head width"):
        head.set_prototype(5, torch.zeros(3))


def test_cosine_forward_normalizes_features_and_weights():
    head = IncrementalHead(feature_dim=2, cosine=True, num_classes=1, scale=1.0)
    head.set_prototype(0, torch.tensor([3.0, 4.0]))  # norm 5, direction (0.6, 0.8)

    features = torch.tensor([[0.0, 10.0]])  # normalizes to (0, 1)
    logits = head(features)
    expected = torch.tensor([[0.8]])  # cos sim of (0,1) and (0.6,0.8) = 0.8
    assert torch.allclose(logits, expected, atol=1e-5)


def test_plain_linear_forward_is_a_dot_product():
    head = IncrementalHead(feature_dim=2, cosine=False, num_classes=1)
    head.set_prototype(0, torch.tensor([2.0, 3.0]))
    features = torch.tensor([[1.0, 1.0]])
    logits = head(features)
    assert torch.allclose(logits, torch.tensor([[5.0]]))
