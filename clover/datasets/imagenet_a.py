"""ImageNet-A dataset wrapper for CLOVER.

Download: https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p
Expected layout:
    <root>/imagenet-a/train/<class_id>/...
    <root>/imagenet-a/test/<class_id>/...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from clover.datasets.base import CLDataset

_DOWNLOAD_URL = "https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p"

_train_trsf = [
    transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
]
_test_trsf = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]
_common_trsf: List[Any] = []


class ImageNetADataset(CLDataset):
    """ImageNet-A (200-class natural adversarial examples) wrapper.

    Args:
        root: Parent directory containing the ``imagenet-a/`` subdirectory.
        train: Use the training split when ``True``.
        transform: Override the default transform pipeline.

    Raises:
        FileNotFoundError: If the dataset directory is not found.  Download
            from ``https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p``.
    """

    use_path: bool = True

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Any] = None,
    ) -> None:
        super().__init__(root, train, transform)
        split = "train" if train else "test"
        data_dir = f"{root}/imagenet-a/{split}"
        try:
            ds = datasets.ImageFolder(data_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"ImageNet-A data not found at {data_dir!r}. "
                f"Download from {_DOWNLOAD_URL}."
            )
        self._paths = np.array([p for p, _ in ds.imgs])
        self._targets = np.array([t for _, t in ds.imgs])
        self._class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self._targets):
            self._class_to_indices.setdefault(int(label), []).append(idx)

    @property
    def num_classes(self) -> int:
        return 200

    def get_class_to_indices(self) -> Dict[int, List[int]]:
        return dict(self._class_to_indices)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        with open(self._paths[idx], "rb") as fh:
            img = Image.open(fh).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self._targets[idx])

    @property
    def train_trsf(self) -> List[Any]:
        return list(_train_trsf)

    @property
    def test_trsf(self) -> List[Any]:
        return list(_test_trsf)

    @property
    def common_trsf(self) -> List[Any]:
        return list(_common_trsf)
