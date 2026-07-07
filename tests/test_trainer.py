"""Trainer end-to-end on the synthetic disjoint stream with SimpleCIL (SPEC §6.2)."""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import torch

from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method
from clover.training import RunConfig, Trainer
from clover.training.trainer import _seed_everything


def _make_benchmark():
    spec = StreamSpec(dataset="synthetic", init_cls=4, increment=4)
    info = DatasetInfo("synthetic", 20)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())
    return benchmark, train_ds, test_ds


def test_trainer_writes_expected_run_artifacts(tmp_path):
    benchmark, train_ds, test_ds = _make_benchmark()
    method = get_method("simplecil")()
    trainer = Trainer(method, benchmark, train_ds, test_ds, RunConfig(run_dir=str(tmp_path), seed=42))

    r_matrix = trainer.run()

    files = os.listdir(tmp_path)
    assert "status.json" in files
    assert "R_matrix.npy" in files
    assert "per_task.csv" in files
    for t in range(benchmark.nb_experiences):
        assert f"ckpt_task{t:02d}.pt" in files

    with open(tmp_path / "status.json") as fh:
        status = json.load(fh)
    assert status["state"] == "done"
    assert status["last_completed_task"] == benchmark.nb_experiences - 1
    assert "heartbeat" in status

    saved = np.load(tmp_path / "R_matrix.npy")
    assert np.array_equal(saved, r_matrix.to_array(), equal_nan=True)


def test_trainer_r_matrix_is_finite_and_above_chance_on_diagonal():
    benchmark, train_ds, test_ds = _make_benchmark()
    method = get_method("simplecil")()

    with tempfile.TemporaryDirectory() as run_dir:
        r_matrix = Trainer(
            method, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)
        ).run()

    array = r_matrix.to_array()
    n = benchmark.nb_experiences
    for t in range(n):
        for te in range(t + 1):
            assert math.isfinite(array[te, t])
        assert array[t, t] > 1 / benchmark.total_classes  # diagonal: right after learning it
    for t in range(n):
        for te in range(t + 1, n):
            assert math.isnan(array[te, t])  # never-evaluated cells stay NaN


def test_seed_everything_sets_cudnn_flags():
    _seed_everything(42, cudnn_benchmark=False)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    _seed_everything(42, cudnn_benchmark=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is True


def test_trainer_run_threads_cudnn_benchmark_from_run_config(tmp_path):
    benchmark, train_ds, test_ds = _make_benchmark()
    method = get_method("simplecil")()
    run_config = RunConfig(run_dir=str(tmp_path), seed=42, cudnn_benchmark=True)

    try:
        Trainer(method, benchmark, train_ds, test_ds, run_config).run()
        assert torch.backends.cudnn.benchmark is True
    finally:
        _seed_everything(42, cudnn_benchmark=False)  # restore the default for later tests
