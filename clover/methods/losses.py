"""Revisit-safe loss policies (SPEC §5.2): the only place loss masking lives.

Built on ``LabelSpaceView`` (``clover/core/experience.py``) and set-membership
masks — never range arithmetic (``targets >= known_classes``,
``logits[:, :k] = -inf``). No method module may hand-roll its own masking.

``masked_logits`` returns the masked logits *and* the matching sample mask
from one call, keyed by the same ``kind`` — the two cannot be obtained
separately, which is what makes the v1 bug (mask logit columns for the
whole batch, but keep old-labelled samples in the same CE call) impossible
to reproduce through this API.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F

from clover.core.experience import LabelSpaceView, ids_to_column_mask

Kind = Literal["new", "seen"]


def masked_logits(
    logits: torch.Tensor, targets: torch.Tensor, view: LabelSpaceView, kind: Kind
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logits with invalid columns at -inf, matching sample mask)``.

    Both the logit-column mask and the sample mask are derived from the
    *same* ``kind`` — ``"new"`` restricts to new-class samples AND new-class
    logit columns; ``"seen"`` restricts to seen-class samples AND seen-class
    columns. This differs from ``view.logit_mask()`` (which covers
    seen | new, for eval-time prediction): a loss must never see a sample
    whose true-label column it just masked to ``-inf``.

    Args:
        logits: ``[batch, width]`` raw classifier output.
        targets: ``[batch]`` integer labels.
        view: The experience's label-space view.
        kind: ``"new"`` or ``"seen"``.
    """
    if kind == "new":
        sample_mask = view.new_mask(targets)
        logit_mask = ids_to_column_mask(view.new_classes, logits.shape[-1])
    elif kind == "seen":
        sample_mask = view.old_mask(targets)
        logit_mask = ids_to_column_mask(view.seen_classes, logits.shape[-1])
    else:
        raise ValueError(f"kind must be 'new' or 'seen', got {kind!r}.")

    masked = logits.masked_fill(~logit_mask, float("-inf"))
    return masked, sample_mask


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, view: LabelSpaceView, kind: Kind) -> torch.Tensor:
    masked, sample_mask = masked_logits(logits, targets, view, kind)
    if not sample_mask.any():
        # Zero loss connected to the graph -- no new/seen samples in this
        # batch is not an error, just nothing to learn from right now.
        return logits.sum() * 0.0
    return F.cross_entropy(masked[sample_mask], targets[sample_mask])


def new_class_ce(logits: torch.Tensor, targets: torch.Tensor, view: LabelSpaceView) -> torch.Tensor:
    """CE restricted to new-class samples AND new-class logit columns.

    The generalization of the EASE/TUNA/MOS patches and the correct form of
    the L2P/DualPrompt/CODA "mask old logits" intent.
    """
    return _masked_ce(logits, targets, view, "new")


def seen_class_ce(logits: torch.Tensor, targets: torch.Tensor, view: LabelSpaceView) -> torch.Tensor:
    """CE over all seen classes (methods that legitimately train on revisits,
    e.g. drift anchors)."""
    return _masked_ce(logits, targets, view, "seen")
