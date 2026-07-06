"""``image_folder`` proven end-to-end against a real tiny fixture directory
(SPEC §7's literal acceptance criterion: a custom dataset benchmarked via
an inline ``dataset: {type: image_folder, ...}`` config, no clover source
touched).

``clover inspect``/``clover preflight`` exercise the full config-resolution
path (inline mapping -> ``StreamSpec`` -> scenario/plan resolution) without
training. The dataloader test goes one step further and proves the actual
``Trainer``-used ``ExperienceDataset`` + ``DataLoader`` machinery produces
correctly shaped, transformed batches from real files on disk -- the full
extent of the dataset layer's responsibility.

Training the batches through a real method/backbone is out of scope here:
every backbone currently registered (``tiny_mlp``/``tiny_vit``) is a
synthetic-only stand-in sized for 1-channel 8x8 input (see PLAN.md's P5
log) -- a pre-existing limitation shared by every real (non-synthetic)
built-in dataset already in this repo, not something specific to
``image_folder``. Real dataset + real backbone runs are cluster jobs
(CLAUDE.md), never local.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml
from PIL import Image

from clover.cli import _apply_default_transforms, main
from clover.config import load_yaml, resolve_config
from clover.core.spec import DatasetInfo
from clover.core.stream import build_benchmark
from clover.datasets import get_dataset
from clover.training.trainer import ExperienceDataset


def _make_fixture(root, classes_and_counts):
    for split in ("train", "test"):
        for class_name, count in classes_and_counts.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(count):
                img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
                img.save(class_dir / f"{i}.png")


def _write_config(tmp_path, fixture_root):
    config = {
        "run": {"output_dir": str(tmp_path / "runs")},
        "stream": {
            "dataset": {"type": "image_folder", "root": str(fixture_root), "num_classes": 2},
            "init_cls": 1,
            "increment": 1,
        },
        "method": {"name": "simplecil"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_inspect_resolves_inline_image_folder_config(tmp_path, capsys):
    fixture_root = tmp_path / "fixture"
    _make_fixture(fixture_root, {"class_a": 3, "class_b": 3})
    config_path = _write_config(tmp_path, fixture_root)

    exit_code = main(["inspect", config_path])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "dataset: image_folder (2 classes)" in out
    assert "experiences: 2" in out


def test_preflight_passes_for_inline_image_folder_config(tmp_path, capsys):
    fixture_root = tmp_path / "fixture"
    _make_fixture(fixture_root, {"class_a": 3, "class_b": 3})
    config_path = _write_config(tmp_path, fixture_root)

    exit_code = main(["preflight", config_path])
    assert exit_code == 0
    assert "preflight OK" in capsys.readouterr().out


def test_full_dataloader_pipeline_yields_correctly_shaped_transformed_batches(tmp_path):
    fixture_root = tmp_path / "fixture"
    _make_fixture(fixture_root, {"class_a": 3, "class_b": 3})
    config_path = _write_config(tmp_path, fixture_root)

    raw = load_yaml(config_path)
    resolved = resolve_config(raw)

    dataset_cls = get_dataset(resolved.stream_spec.dataset)
    kwargs = {
        "root": resolved.stream_spec.data_root,
        "num_classes": resolved.stream_spec.dataset_num_classes,
    }
    train_dataset = dataset_cls(train=True, **kwargs)
    test_dataset = dataset_cls(train=False, **kwargs)
    _apply_default_transforms(train_dataset)
    _apply_default_transforms(test_dataset)

    info = DatasetInfo(resolved.stream_spec.dataset, train_dataset.num_classes)
    benchmark = build_benchmark(
        resolved.stream_spec,
        info,
        train_dataset.get_class_to_indices(),
        test_dataset.get_class_to_indices(),
    )
    assert benchmark.nb_experiences == 2

    exp = benchmark.train_stream[0]
    loader = torch.utils.data.DataLoader(
        ExperienceDataset(train_dataset, exp.image_indices), batch_size=2, shuffle=True
    )
    images, targets = next(iter(loader))
    assert images.shape[1:] == (3, 224, 224)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.int64
