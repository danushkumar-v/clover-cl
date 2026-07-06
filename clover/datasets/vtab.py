"""VTAB-CIL dataset wrapper (SPEC §7): 50-class Visual Task Adaptation
Benchmark subset.

Staging: download from the URL below and extract so that
``<root>/vtab-cil/vtab/train/`` and ``<root>/vtab-cil/vtab/test/`` each
contain one subdirectory per class (standard ``ImageFolder`` layout).
"""

from __future__ import annotations

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("vtab")
class VTABDataset(ImageFolderCLDataset):
    dataset_dir = "vtab-cil/vtab"
    _num_classes = 50
    download_url = "https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_"
