"""Config-only ``image_folder`` dataset (SPEC §7): ``dataset: {type:
image_folder, root: ..., num_classes: ...}``. Unlike the 5 named
``ImageFolder``-backed built-ins (which each know their own fixed class
count and a documented download URL), this one has neither -- both the
root and the class count come straight from the user's config, so a
custom dataset can be benchmarked without touching clover source at all
(SPEC §7's acceptance criterion).
"""

from __future__ import annotations

from typing import Any, Optional

from clover.datasets import register_dataset
from clover.datasets.image_folder_base import ImageFolderCLDataset


@register_dataset("image_folder")
class ImageFolderDataset(ImageFolderCLDataset):
    """A user's own dataset, laid out as ``<root>/{train,test}/<class>/...``
    -- no named subdirectory nesting, unlike the 5 built-ins (whose root
    is a *parent* directory containing one fixed subdirectory name).
    """

    dataset_dir = ""
    download_url = ""

    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[Any] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        if num_classes is None:
            raise ValueError(
                "image_folder requires an explicit num_classes -- set it in the "
                "config's inline dataset block, e.g. "
                "dataset: {type: image_folder, root: ..., num_classes: ...}."
            )
        self._num_classes = num_classes
        super().__init__(root, train, transform)

        actual = len({int(t) for t in self._targets})
        if actual != num_classes:
            raise ValueError(
                f"image_folder declared num_classes={num_classes} but found {actual} "
                f"class subdirectories on disk under {root!r} -- check for a config "
                "typo or a staging mistake."
            )

    def _missing_data_hint(self) -> str:
        return (
            "Point the config's dataset.root at a directory with 'train/'/'test/' "
            "subfolders, one per class."
        )
