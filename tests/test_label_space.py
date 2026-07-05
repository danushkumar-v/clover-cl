"""LabelSpaceView mask correctness (SPEC §5.1): set membership, not range arithmetic."""

from __future__ import annotations

import torch

from clover.core.experience import LabelSpaceView


def _view(new_classes, seen_classes, revisiting_classes=(), head_size=None):
    return LabelSpaceView(
        new_classes=frozenset(new_classes),
        seen_classes=frozenset(seen_classes),
        revisiting_classes=frozenset(revisiting_classes),
        head_size=head_size if head_size is not None else len(set(new_classes) | set(seen_classes)),
    )


def test_new_mask_flags_only_new_class_samples():
    view = _view(new_classes=[5, 6], seen_classes=[0, 1, 2])
    targets = torch.tensor([0, 5, 1, 6, 2])
    assert view.new_mask(targets).tolist() == [False, True, False, True, False]


def test_old_mask_flags_only_seen_class_samples():
    view = _view(new_classes=[5, 6], seen_classes=[0, 1, 2])
    targets = torch.tensor([0, 5, 1, 6, 2])
    assert view.old_mask(targets).tolist() == [True, False, True, False, True]


def test_logit_mask_covers_seen_and_new_only():
    view = _view(new_classes=[5, 6], seen_classes=[0, 1, 2])
    mask = view.logit_mask(width=10)
    assert mask.tolist() == [True, True, True, False, False, True, True, False, False, False]


def test_masks_use_set_membership_not_numeric_range():
    """A same-id revisit re-presents a *low* id in a later batch: a naive
    ``targets >= known_classes`` check would misclassify it. Set-membership
    must get it right regardless of numeric magnitude.
    """
    # Class 0 is an anchor (cumulative_drift-style): it was seen long ago,
    # but keeps recurring later mixed with much higher-numbered new classes.
    view = _view(new_classes=[17, 18], seen_classes=[0, 1, 2, 3, 4], revisiting_classes=[0])
    targets = torch.tensor([0, 17, 3, 18, 0])

    assert view.new_mask(targets).tolist() == [False, True, False, True, False]
    assert view.old_mask(targets).tolist() == [True, False, True, False, True]


def test_empty_new_classes_masks_nothing():
    view = _view(new_classes=[], seen_classes=[0, 1, 2])
    targets = torch.tensor([0, 1, 2])
    assert not view.new_mask(targets).any()
    assert view.old_mask(targets).all()


def test_logit_mask_ignores_ids_beyond_width():
    view = _view(new_classes=[5], seen_classes=[0, 1], head_size=6)
    mask = view.logit_mask(width=4)  # narrower than head_size on purpose
    assert mask.tolist() == [True, True, False, False]
