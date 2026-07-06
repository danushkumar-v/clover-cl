"""ImageFolder-backed dataset wrappers correctness (SPEC §7): CUB-200,
ImageNet-R, ImageNet-A, OmniBenchmark, VTAB. No real dataset download --
either the ``FileNotFoundError`` staging-error path, or a tiny hand-built
``ImageFolder``-shaped fixture directory exercising the real loading path
end-to-end.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from PIL import Image

from clover.datasets import get_dataset

_CASES = [
    ("cub200", "clover.datasets.cub200", "CUB200Dataset", "cub"),
    ("imagenet_r", "clover.datasets.imagenet_r", "ImageNetRDataset", "imagenet-r"),
    ("imagenet_a", "clover.datasets.imagenet_a", "ImageNetADataset", "imagenet-a"),
    ("omnibenchmark", "clover.datasets.omnibench", "OmniBenchDataset", "omnibenchmark"),
    ("vtab", "clover.datasets.vtab", "VTABDataset", "vtab-cil/vtab"),
]


@pytest.mark.parametrize("registry_name,module,cls_name,dataset_dir", _CASES)
def test_registered_correctly(registry_name, module, cls_name, dataset_dir):
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    assert get_dataset(registry_name) is cls


@pytest.mark.parametrize("registry_name,module,cls_name,dataset_dir", _CASES)
def test_raises_actionable_error_on_missing_data(registry_name, module, cls_name, dataset_dir, tmp_path):
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    with pytest.raises(FileNotFoundError, match="Download from"):
        cls(root=str(tmp_path), train=True)


def _make_fake_image_folder(root, dataset_dir, split, classes_and_counts):
    split_dir = root / dataset_dir / split
    for class_name, count in classes_and_counts.items():
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
            img.save(class_dir / f"{i}.png")


@pytest.mark.parametrize("registry_name,module,cls_name,dataset_dir", _CASES)
def test_loads_a_real_tiny_fixture_directory_end_to_end(
    registry_name, module, cls_name, dataset_dir, tmp_path
):
    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)

    _make_fake_image_folder(tmp_path, dataset_dir, "train", {"class_a": 2, "class_b": 3})

    ds = cls(root=str(tmp_path), train=True)
    assert len(ds) == 5
    c2i = ds.get_class_to_indices()
    assert sorted(len(idxs) for idxs in c2i.values()) == [2, 3]

    image, label = ds[0]
    assert isinstance(label, int)
    assert image.size == (8, 8)  # PIL Image, no transform applied

    assert isinstance(ds.train_trsf, list) and ds.train_trsf
    assert isinstance(ds.test_trsf, list) and ds.test_trsf
    assert ds.use_path is True
