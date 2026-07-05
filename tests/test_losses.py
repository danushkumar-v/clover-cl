"""Loss policy correctness (SPEC §5.2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from clover.core.experience import LabelSpaceView
from clover.methods.losses import masked_logits, new_class_ce, seen_class_ce


def _view(new_classes, seen_classes):
    return LabelSpaceView(
        new_classes=frozenset(new_classes),
        seen_classes=frozenset(seen_classes),
        revisiting_classes=frozenset(),
        head_size=len(set(new_classes) | set(seen_classes)),
    )


def test_masked_logits_pairs_column_mask_with_matching_sample_mask():
    view = _view(new_classes=[2, 3], seen_classes=[0, 1])
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 2, 1, 3])

    masked, sample_mask = masked_logits(logits, targets, view, "new")
    assert sample_mask.tolist() == [False, True, False, True]
    assert torch.isinf(masked[:, :2]).all()  # old columns invalidated
    assert not torch.isinf(masked[:, 2:]).any()

    masked, sample_mask = masked_logits(logits, targets, view, "seen")
    assert sample_mask.tolist() == [True, False, True, False]


def test_new_class_ce_matches_hand_computed_ce_on_new_samples_only():
    view = _view(new_classes=[2, 3], seen_classes=[0, 1])
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 2, 1, 3])

    loss = new_class_ce(logits, targets, view)

    new_mask = torch.tensor([False, True, False, True])
    new_columns_only = torch.tensor([False, False, True, True])  # view.new_classes = {2, 3}
    masked = logits.masked_fill(~new_columns_only, float("-inf"))
    expected = F.cross_entropy(masked[new_mask], targets[new_mask])
    assert torch.allclose(loss, expected)


def test_seen_class_ce_matches_hand_computed_ce_on_seen_samples_only():
    view = _view(new_classes=[2, 3], seen_classes=[0, 1])
    torch.manual_seed(0)
    logits = torch.randn(4, 4)
    targets = torch.tensor([0, 2, 1, 3])

    loss = seen_class_ce(logits, targets, view)

    old_mask = torch.tensor([True, False, True, False])
    seen_columns_only = torch.tensor([True, True, False, False])  # view.seen_classes = {0, 1}
    masked = logits.masked_fill(~seen_columns_only, float("-inf"))
    expected = F.cross_entropy(masked[old_mask], targets[old_mask])
    assert torch.allclose(loss, expected)


def test_new_class_ce_is_zero_and_graph_connected_when_no_new_samples():
    view = _view(new_classes=[2, 3], seen_classes=[0, 1])
    logits = torch.randn(4, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1])  # no new-class samples at all

    loss = new_class_ce(logits, targets, view)
    assert loss.item() == 0.0
    loss.backward()  # must not raise -- connected to the graph
    assert logits.grad is not None


def test_seen_class_ce_is_zero_and_graph_connected_when_no_seen_samples():
    view = _view(new_classes=[0, 1], seen_classes=[])
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1])

    loss = seen_class_ce(logits, targets, view)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None


def test_masked_logits_rejects_unknown_kind():
    view = _view(new_classes=[0], seen_classes=[])
    with pytest.raises(ValueError, match="kind must be"):
        masked_logits(torch.randn(2, 1), torch.tensor([0, 0]), view, "bogus")
