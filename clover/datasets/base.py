"""``CLDataset`` ABC (SPEC §7): the contract every dataset plugin implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class CLDataset(ABC):
    """Abstract wrapper every dataset must implement.

    Args:
        root: Root directory where the dataset is stored or will be downloaded.
        train: If ``True``, use the training split; otherwise the test split.
        transform: Optional transform applied in ``__getitem__``.
    """

    #: Native input resolution this dataset provides (e.g. 224 for ViT-ready
    #: datasets); backbones/config read this instead of hard-coding a size.
    input_size: int = 32
    #: True when items are file paths, False when arrays/tensors are held in memory.
    use_path: bool = False

    def __init__(self, root: str, train: bool = True, transform: Optional[Any] = None) -> None:
        self.root = root
        self.train = train
        self.transform = transform

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Total number of classes in this dataset."""

    @abstractmethod
    def get_class_to_indices(self) -> Dict[int, List[int]]:
        """Return ``{original_class_id: [sample_indices]}`` for this split."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return ``(image, original_class_id)`` for sample *idx*."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in this split."""

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
