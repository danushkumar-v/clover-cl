"""Experience — one step in a continual learning stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import torch.utils.data


@dataclass(frozen=True)
class Experience:
    """One step in a continual learning stream.

    An Experience is a Dataset plus self-describing metadata about its
    position and content within the stream.  Everything a CL training loop or
    evaluation routine needs to know about "where am I and what am I seeing"
    is on this object — no external bookkeeping required.

    Attributes:
        task_label: Zero-based step index within the stream.
        benchmark_name: Identifier string for the parent benchmark.
        dataset: PyTorch Dataset yielding ``(sample_idx, image, remapped_label)``.
        classes_in_this_experience: Remapped class IDs present in this experience.
        classes_seen_so_far: Union of class IDs over experiences ``0..task_label``.
        classes_in_future: Class IDs that appear in experiences after this one.
            For diagnostic / evaluation use only — never consume during training.
        total_classes_in_stream: Total unique class IDs across the whole stream.
        revisiting_classes: Subset of ``classes_in_this_experience`` that appeared
            in at least one earlier experience.
        first_appearance_of: Subset of ``classes_in_this_experience`` that have
            not been seen before this experience.
        overlap_with_previous: ``{prev_task_label: n_shared_classes}`` for all
            earlier experiences that share at least one class with this one.
        image_indices: ``{class_id: [image_indices]}`` for this experience,
            suitable for manifest serialisation.
        n_samples: Total number of samples in ``dataset``.
    """

    # Identity
    task_label: int
    benchmark_name: str

    # Data
    dataset: "torch.utils.data.Dataset"

    # Class membership
    classes_in_this_experience: List[int]
    classes_seen_so_far: List[int]
    classes_in_future: List[int]
    total_classes_in_stream: int

    # Revisit metadata
    revisiting_classes: List[int]
    first_appearance_of: List[int]
    overlap_with_previous: Dict[int, int]

    # Reproducibility
    image_indices: Dict[int, List[int]]
    n_samples: int

    def __post_init__(self) -> None:
        # Validate partition invariant: revisiting ∪ first_appearance == classes_in_this_experience
        here = set(self.classes_in_this_experience)
        revisiting = set(self.revisiting_classes)
        first = set(self.first_appearance_of)
        if revisiting | first != here:
            raise ValueError(
                "revisiting_classes ∪ first_appearance_of must equal "
                "classes_in_this_experience."
            )
        if revisiting & first:
            raise ValueError(
                "revisiting_classes and first_appearance_of must be disjoint."
            )

    def has_class(self, c: int) -> bool:
        """Return True if class *c* appears in this experience."""
        return c in self.classes_in_this_experience

    def is_revisit_experience(self) -> bool:
        """Return True if at least one class is revisiting from an earlier experience."""
        return bool(self.revisiting_classes)
