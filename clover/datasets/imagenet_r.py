"""ImageNet-R dataset wrapper (SPEC §7): 200-class artistic renditions.

Staging: download from the URL below and extract so that
``<root>/imagenet-r/train/`` and ``<root>/imagenet-r/test/`` each contain
one subdirectory per class (standard ``ImageFolder`` layout).
"""

from __future__ import annotations

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("imagenet_r")
class ImageNetRDataset(ImageFolderCLDataset):
    dataset_dir = "imagenet-r"
    _num_classes = 200
    download_url = "https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR"
