"""Shared base for CLOVER's ``torchvision.datasets.ImageFolder``-backed
datasets (SPEC §7): CUB-200, ImageNet-R, ImageNet-A, OmniBenchmark, VTAB.

Ported from this repo's own `v1` (``main`` branch, read via ``git show``,
no code checked out) -- confirmed via direct comparison that all five of
v1's per-dataset wrapper classes are structurally identical (same
``ImageFolder`` load, same ``FileNotFoundError`` staging message, same
train/test transform pipeline, same ``class_to_indices`` construction),
differing only in the on-disk subdirectory name, class count, and download
URL. v1 duplicated this five times; this base class exists once instead --
a genuine, confirmed (not hypothetical) case for extracting a shared base,
since all five concrete uses already existed and matched exactly.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from clover.datasets.base import CLDataset

_TRAIN_TRSF: List[Any] = [
    transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
]
_TEST_TRSF: List[Any] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
]


class ImageFolderCLDataset(CLDataset):
    """Common ``ImageFolder``-backed dataset shape: ``<root>/<dataset_dir>/
    {train,test}/<class>/...`` on disk, staged manually (no auto-download
    -- these datasets aren't distributed via a stable, licensable direct
    URL the way CIFAR-100 is through torchvision).

    Subclasses set three class attributes and register themselves:

    .. code-block:: python

        @register_dataset("cub200")
        class CUB200Dataset(ImageFolderCLDataset):
            dataset_dir = "cub"
            _num_classes = 200
            download_url = "https://..."
    """

    use_path: bool = True
    input_size: int = 224

    #: Set by each subclass.
    dataset_dir: ClassVar[str]
    _num_classes: ClassVar[int]
    download_url: ClassVar[str]

    def __init__(self, root: str = "./data", train: bool = True, transform: Optional[Any] = None) -> None:
        super().__init__(root, train, transform)
        split = "train" if train else "test"
        data_dir = f"{root}/{self.dataset_dir}/{split}"
        try:
            image_folder = datasets.ImageFolder(data_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{type(self).__name__} data not found at {data_dir!r}. Download "
                f"from {self.download_url} and extract so that {data_dir!r} exists "
                "(see docs/concepts.md for dataset staging)."
            ) from None
        self._paths = np.array([p for p, _ in image_folder.imgs])
        self._targets = np.array([t for _, t in image_folder.imgs])
        self._class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self._targets):
            self._class_to_indices.setdefault(int(label), []).append(idx)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def get_class_to_indices(self) -> Dict[int, List[int]]:
        return dict(self._class_to_indices)

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        with open(self._paths[idx], "rb") as fh:
            image = Image.open(fh).convert("RGB")
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
        return []
