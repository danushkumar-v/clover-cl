"""``image_folder`` config-only dataset (SPEC §7): a user's own dataset,
laid out as ``<root>/{train,test}/<class>/...``, benchmarked without
touching clover source. No real download -- a tiny hand-built fixture
directory exercises the real loading path end-to-end."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from clover.datasets import get_dataset
from clover.datasets.image_folder_config import ImageFolderDataset


def test_registered_as_image_folder():
    assert get_dataset("image_folder") is ImageFolderDataset


def _make_fake_image_folder(root, split, classes_and_counts):
    split_dir = root / split
    for class_name, count in classes_and_counts.items():
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
            img.save(class_dir / f"{i}.png")


def test_requires_explicit_num_classes(tmp_path):
    _make_fake_image_folder(tmp_path, "train", {"class_a": 2, "class_b": 3})
    with pytest.raises(ValueError, match="requires an explicit num_classes"):
        ImageFolderDataset(root=str(tmp_path), train=True)


def test_raises_on_num_classes_mismatch(tmp_path):
    _make_fake_image_folder(tmp_path, "train", {"class_a": 2, "class_b": 3})
    with pytest.raises(ValueError, match=r"declared num_classes=5 but found 2"):
        ImageFolderDataset(root=str(tmp_path), train=True, num_classes=5)


def test_raises_actionable_error_on_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset.root"):
        ImageFolderDataset(root=str(tmp_path), train=True, num_classes=2)


def test_loads_a_real_tiny_fixture_directory_end_to_end(tmp_path):
    _make_fake_image_folder(tmp_path, "train", {"class_a": 2, "class_b": 3})

    ds = ImageFolderDataset(root=str(tmp_path), train=True, num_classes=2)
    assert len(ds) == 5
    assert ds.num_classes == 2
    c2i = ds.get_class_to_indices()
    assert sorted(len(idxs) for idxs in c2i.values()) == [2, 3]

    image, label = ds[0]
    assert isinstance(label, int)
    assert image.size == (8, 8)  # PIL Image, no transform applied

    assert isinstance(ds.train_trsf, list) and ds.train_trsf
    assert isinstance(ds.test_trsf, list) and ds.test_trsf
    assert ds.use_path is True
