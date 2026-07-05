"""Experience + LabelSpaceView (SPEC §5.1).

This phase keeps ``Experience`` pure/serializable — unlike v1, it does not
hold a live ``torch.utils.data.Dataset`` (that required a real, loaded
dataset and coupled ``clover/core`` to torch); wrapping ``image_indices``
into an actual per-experience Dataset is a P3/dataset-layer concern once
real data exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

import torch

from clover.core.plan import EchoEntry


def _ids_to_sample_mask(targets: torch.Tensor, ids: FrozenSet[int]) -> torch.Tensor:
    """Bool mask, set membership only — never range arithmetic (SPEC §5.1)."""
    if not ids:
        return torch.zeros_like(targets, dtype=torch.bool)
    id_tensor = torch.tensor(sorted(ids), device=targets.device, dtype=targets.dtype)
    return torch.isin(targets, id_tensor)


def ids_to_column_mask(ids: FrozenSet[int], width: int) -> torch.Tensor:
    """Bool ``[width]`` mask, set membership only: column ``i`` is True iff ``i in ids``.

    Public so ``methods/losses.py`` can build kind-specific (new-only /
    seen-only) logit-column masks without duplicating set-membership logic.
    """
    mask = torch.zeros(width, dtype=torch.bool)
    valid = [i for i in ids if 0 <= i < width]
    if valid:
        mask[torch.tensor(valid, dtype=torch.long)] = True
    return mask


@dataclass(frozen=True)
class LabelSpaceView:
    """Set-membership view of an experience's label space (SPEC §5.1).

    Every mask here is built from a materialized id set — never from
    ``targets >= known_classes`` range arithmetic, which breaks the moment
    a same-id revisit re-presents an "old" (numerically small) id in a
    later batch.
    """

    new_classes: FrozenSet[int]
    seen_classes: FrozenSet[int]
    revisiting_classes: FrozenSet[int]
    head_size: int

    def new_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Bool ``[batch]``: ``targets[i] in new_classes``."""
        return _ids_to_sample_mask(targets, self.new_classes)

    def old_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """Bool ``[batch]``: ``targets[i] in seen_classes``."""
        return _ids_to_sample_mask(targets, self.seen_classes)

    def logit_mask(self, width: int) -> torch.Tensor:
        """Bool ``[width]``: which logit columns are valid *now* (seen | new).

        For eval-time prediction (restrict logits to every class introduced
        so far). Loss policies use kind-specific masks instead — see
        ``methods/losses.py``.
        """
        return ids_to_column_mask(self.seen_classes | self.new_classes, width)


@dataclass(frozen=True)
class Experience:
    """One step in a continual learning stream.

    Attributes:
        task_label: Zero-based step index within the stream.
        benchmark_name: Identifier string for the parent benchmark.
        dataset: Dataset name (e.g. ``"cifar100"``).
        classes_in_this_experience: Class ids present in this experience.
        classes_seen_so_far: Union of class ids over experiences ``0..task_label``.
        classes_in_future: Class ids appearing after this experience.
            Diagnostic/evaluation use only — never consume during training.
        total_classes_in_stream: Total unique class ids across the whole stream.
        revisiting_classes: Subset of ``classes_in_this_experience`` seen before.
        first_appearance_of: Subset of ``classes_in_this_experience`` not seen before.
        overlap_with_previous: ``{prev_task_label: n_shared_classes}``.
        echo_map: Echo ids first introduced in this experience.
        label_space: Set-membership view for revisit-safe loss policies.
        image_indices: ``{class_id: [image_indices]}`` for this experience.
        n_samples: Total number of samples in this experience.
    """

    task_label: int
    benchmark_name: str
    dataset: str

    classes_in_this_experience: List[int]
    classes_seen_so_far: List[int]
    classes_in_future: List[int]
    total_classes_in_stream: int

    revisiting_classes: List[int]
    first_appearance_of: List[int]
    overlap_with_previous: Dict[int, int]
    echo_map: Dict[int, EchoEntry]
    label_space: LabelSpaceView

    image_indices: Dict[int, List[int]]
    n_samples: int

    def __post_init__(self) -> None:
        here = set(self.classes_in_this_experience)
        revisiting = set(self.revisiting_classes)
        first = set(self.first_appearance_of)
        if revisiting | first != here:
            raise ValueError(
                "revisiting_classes | first_appearance_of must equal "
                "classes_in_this_experience."
            )
        if revisiting & first:
            raise ValueError("revisiting_classes and first_appearance_of must be disjoint.")

    def has_class(self, c: int) -> bool:
        """Return True if class *c* appears in this experience."""
        return c in self.classes_in_this_experience

    def is_revisit_experience(self) -> bool:
        """Return True if at least one class is revisiting from an earlier experience."""
        return bool(self.revisiting_classes)
