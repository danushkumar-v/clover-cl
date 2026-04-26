"""Tests for clover/utils/visualizer.py."""

from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend — must be set before any other matplotlib import


@pytest.fixture
def baseline_dm(patched_cifar100):
    from clover.core.data_manager import OverlapDataManager

    return OverlapDataManager("cifar100", init_cls=10, increment=10)


@pytest.fixture
def overlap_dm(patched_cifar100):
    from clover.core.data_manager import OverlapDataManager
    from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec

    spec = OverlapSpec(
        mode="partial",
        pairs=[OverlapPair(tasks=(0, 5), shared_classes=[0, 1, 2])],
        image_split=ImageSplit(strategy="duplicate"),
        seed=42,
    )
    return OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)


def test_plot_overlap_matrix_creates_file(baseline_dm, tmp_path):
    from clover.utils.visualizer import plot_overlap_matrix

    out = str(tmp_path / "overlap.png")
    plot_overlap_matrix(baseline_dm, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_class_frequency_creates_file(baseline_dm, tmp_path):
    from clover.utils.visualizer import plot_class_frequency

    out = str(tmp_path / "freq.png")
    plot_class_frequency(baseline_dm, out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_plot_overlap_matrix_with_overlap(overlap_dm, tmp_path):
    from clover.utils.visualizer import plot_overlap_matrix

    out = str(tmp_path / "overlap_ov.png")
    plot_overlap_matrix(overlap_dm, out, title="Test overlap")
    assert os.path.exists(out)


def test_plot_overlap_matrix_creates_parent_dir(baseline_dm, tmp_path):
    from clover.utils.visualizer import plot_overlap_matrix

    nested = str(tmp_path / "subdir" / "deep" / "overlap.png")
    plot_overlap_matrix(baseline_dm, nested)
    assert os.path.exists(nested)
