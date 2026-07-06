"""OmniBenchmark dataset wrapper (SPEC §7): 300-class diverse benchmark.

Staging: download from the URL below and extract so that
``<root>/omnibenchmark/train/`` and ``<root>/omnibenchmark/test/`` each
contain one subdirectory per class (standard ``ImageFolder`` layout).
"""

from __future__ import annotations

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("omnibenchmark")
class OmniBenchDataset(ImageFolderCLDataset):
    dataset_dir = "omnibenchmark"
    _num_classes = 300
    download_url = "https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u"
