"""CUB-200-2011 dataset wrapper (SPEC §7): 200 bird species.

Staging: download from the URL below and extract so that
``<root>/cub/train/`` and ``<root>/cub/test/`` each contain one
subdirectory per class (standard ``ImageFolder`` layout).
"""

from __future__ import annotations

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("cub200")
class CUB200Dataset(ImageFolderCLDataset):
    dataset_dir = "cub"
    _num_classes = 200
    download_url = "https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb"
