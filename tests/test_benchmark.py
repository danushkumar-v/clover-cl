"""Tests for clover/core/benchmark.py."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from clover import build_benchmark, StreamSpec, RevisitSpec


@pytest.fixture
def tiny_spec(patched_cifar100):
    return StreamSpec(
        dataset="cifar100",
        init_cls=10,
        increment=10,
        revisits=[RevisitSpec(classes=[5, 15], times=1, placement="random", min_gap=2)],
        stream_seed=42,
    )


@pytest.fixture
def tiny_bench(tiny_spec):
    return build_benchmark(tiny_spec)


def test_nb_experiences(tiny_bench):
    assert tiny_bench.nb_experiences == 10


def test_train_test_same_nb_experiences(tiny_bench):
    assert len(tiny_bench.train_stream) == len(tiny_bench.test_stream)


def test_total_classes(tiny_bench):
    assert tiny_bench.total_classes == 100


def test_name_contains_dataset(tiny_bench):
    assert "cifar100" in tiny_bench.name


def test_overlap_matrix_shape(tiny_bench):
    mat = tiny_bench.overlap_matrix()
    T = tiny_bench.nb_experiences
    assert mat.shape == (T, T)


def test_overlap_matrix_matches_engine(tiny_bench):
    mat_bench = tiny_bench.overlap_matrix()
    mat_engine = tiny_bench.engine.get_overlap_matrix()
    np.testing.assert_array_equal(mat_bench, mat_engine)


def test_save_manifest(tiny_bench, tmp_path):
    path = str(tmp_path / "manifest.json")
    tiny_bench.save_manifest(path)
    assert os.path.exists(path)
    with open(path) as fh:
        doc = json.load(fh)
    assert "_header" in doc
    assert "_stream_metrics" in doc
    assert "train" in doc
    assert "nb_experiences" in doc["_stream_metrics"]


def test_from_yaml(patched_cifar100, tmp_path):
    import yaml

    spec_dict = {
        "dataset": "cifar100",
        "init_cls": 10,
        "increment": 10,
        "shuffle_seed": 1993,
        "stream_seed": 1,
        "image_strategy": "disjoint",
        "image_split_ratio": 0.5,
        "data_root": "./data",
        "revisits": [
            {"classes": [0, 1], "times": 1, "placement": "random", "min_gap": 2}
        ],
    }
    cfg = tmp_path / "bench.yaml"
    cfg.write_text(yaml.dump(spec_dict))

    from clover.core.benchmark import Benchmark
    bench = Benchmark.from_yaml(str(cfg))
    assert bench.nb_experiences == 10
