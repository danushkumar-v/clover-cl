"""CIFAR-100 dataset wrapper (SPEC §7). Ported from this repo's own `v1`
(``main`` branch) -- torchvision owns the actual download/checksum/
extraction; this wrapper only adapts the result to ``CLDataset``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from clover.datasets import register_dataset
from clover.datasets.base import CLDataset

_TRAIN_TRSF: List[Any] = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=63 / 255),
    transforms.ToTensor(),
]
_TEST_TRSF: List[Any] = [transforms.ToTensor()]
_COMMON_TRSF: List[Any] = [
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
]


@register_dataset("cifar100")
class CIFAR100Dataset(CLDataset):
    """CIFAR-100 wrapper. Downloads to *root* via torchvision if not
    already present."""

    use_path: bool = False
    input_size: int = 32

    def __init__(self, root: str = "./data", train: bool = True, transform: Optional[Any] = None) -> None:
        super().__init__(root, train, transform)
        ds = datasets.CIFAR100(root, train=train, download=True)
        self._data: np.ndarray = ds.data  # (N, 32, 32, 3) uint8
        self._targets: np.ndarray = np.array(ds.targets)
        self._class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self._targets):
            self._class_to_indices.setdefault(int(label), []).append(idx)

    @property
    def num_classes(self) -> int:
        return 100

    def get_class_to_indices(self) -> Dict[int, List[int]]:
        return dict(self._class_to_indices)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image = Image.fromarray(self._data[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self._targets[idx])

    @property
    def train_trsf(self) -> List[Any]:
        return list(_TRAIN_TRSF)

    @property
    def test_trsf(self) -> List[Any]:
        return list(_TEST_TRSF)

    @property
    def common_trsf(self) -> List[Any]:
        return list(_COMMON_TRSF)
