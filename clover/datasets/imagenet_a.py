"""ImageNet-A dataset wrapper (SPEC §7): 200-class natural adversarial
examples.

Staging: download from the URL below and extract so that
``<root>/imagenet-a/train/`` and ``<root>/imagenet-a/test/`` each contain
one subdirectory per class (standard ``ImageFolder`` layout) -- ImageNet-A
has no official train/test split, so this split must be created manually
during staging.
"""

from __future__ import annotations

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("imagenet_a")
class ImageNetADataset(ImageFolderCLDataset):
    dataset_dir = "imagenet-a"
    _num_classes = 200
    download_url = "https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p"
