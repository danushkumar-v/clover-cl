"""CIFAR-100 dataset wrapper for CLOVER.

Mirrors the preprocessing from PILOT's ``iCIFAR100`` exactly so that
CLOVER + TOSCA-baseline produces numerically identical features to PILOT + TOSCA.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from clover.datasets.base import CLDataset


class CIFAR100Dataset(CLDataset):
    """CIFAR-100 wrapper.

    Automatically downloads the dataset to *root* if not already present.

    Args:
        root: Directory for the CIFAR-100 data.
        train: Training split when ``True``, test split otherwise.
        transform: Override the default transform pipeline.
    """

    use_path: bool = False

    _train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    _test_trsf = [transforms.ToTensor()]
    _common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__(root, train, transform)
        ds = datasets.CIFAR100(root, train=train, download=True)
        self._data: np.ndarray = ds.data          # (N, 32, 32, 3) uint8
        self._targets: np.ndarray = np.array(ds.targets)
        self._class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self._targets):
            self._class_to_indices.setdefault(int(label), []).append(idx)

    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return 100

    def get_class_to_indices(self) -> Dict[int, List[int]]:
        return dict(self._class_to_indices)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img = Image.fromarray(self._data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self._targets[idx])

    # ------------------------------------------------------------------
    # Transform properties (PILOT-compatible names)
    # ------------------------------------------------------------------

    @property
    def train_trsf(self) -> List[Any]:
        return list(self._train_trsf)

    @property
    def test_trsf(self) -> List[Any]:
        return list(self._test_trsf)

    @property
    def common_trsf(self) -> List[Any]:
        return list(self._common_trsf)
