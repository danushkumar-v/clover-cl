"""Abstract base class for all CLOVER per-dataset wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class CLDataset(ABC):
    """Abstract wrapper that every dataset must implement.

    Subclasses load the raw data once and expose a ``get_class_to_indices``
    method so that :class:`~clover.core.image_assigner` can build task
    assignments without touching actual images.

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        train: If ``True``, use the training split; otherwise the test split.
        transform: Optional torchvision transform applied in ``__getitem__``.
    """

    #: Set by subclass: True when data items are file paths, False when arrays.
    use_path: bool = False

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Total number of classes in this dataset split."""

    @abstractmethod
    def get_class_to_indices(self) -> Dict[int, List[int]]:
        """Return a mapping from *original* class ID to list of sample indices.

        Indices refer to positions in the flat data/targets arrays returned by
        ``__getitem__``.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return ``(image, original_class_id)`` for sample *idx*."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in this split."""

    # ------------------------------------------------------------------
    # Transform helpers (mirrors PILOT's iData structure)
    # ------------------------------------------------------------------

    @property
    def train_trsf(self) -> List[Any]:
        """Training-time augmentation transforms."""
        return []

    @property
    def test_trsf(self) -> List[Any]:
        """Evaluation-time transforms."""
        return []

    @property
    def common_trsf(self) -> List[Any]:
        """Transforms applied after both train and test transforms."""
        return []
