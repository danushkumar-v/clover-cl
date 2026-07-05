"""Interrupted-run resume test (P3's key DONE criterion, SPEC §11 acceptance).

SimpleCIL has no gradient randomness (closed-form prototypes, no dropout),
so a resumed run must reproduce the uninterrupted run's R-matrix and method
state *exactly*, not just approximately.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method
from clover.training import RunConfig, Trainer


def _make_benchmark():
    spec = StreamSpec(dataset="synthetic", init_cls=4, increment=4)
    info = DatasetInfo("synthetic", 20)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())
    return benchmark, train_ds, test_ds


def test_resumed_run_matches_uninterrupted_run_exactly():
    benchmark, train_ds, test_ds = _make_benchmark()

    with tempfile.TemporaryDirectory() as full_dir:
        full_method = get_method("simplecil")()
        r_full = Trainer(
            full_method, benchmark, train_ds, test_ds, RunConfig(run_dir=full_dir, seed=42)
        ).run()

    with tempfile.TemporaryDirectory() as resumed_dir:
        # First "process": completes the whole run, then we simulate a crash
        # by deleting the checkpoints for the last two experiences.
        interrupted_method = get_method("simplecil")()
        Trainer(
            interrupted_method, benchmark, train_ds, test_ds, RunConfig(run_dir=resumed_dir, seed=42)
        ).run()

        checkpoints = sorted(f for f in os.listdir(resumed_dir) if f.startswith("ckpt_task"))
        for f in checkpoints[-2:]:
            os.remove(os.path.join(resumed_dir, f))

        # Second "process": a fresh Trainer + fresh method instance pointed
        # at the same run_dir must detect the remaining checkpoint and
        # resume from there.
        resumed_method = get_method("simplecil")()
        r_resumed = Trainer(
            resumed_method, benchmark, train_ds, test_ds, RunConfig(run_dir=resumed_dir, seed=42)
        ).run()

    assert np.array_equal(r_full.to_array(), r_resumed.to_array(), equal_nan=True)
    assert torch_state_equal(full_method.state_dict(), resumed_method.state_dict())


def torch_state_equal(a, b) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        sub_a, sub_b = a[key], b[key]
        if sub_a.keys() != sub_b.keys():
            return False
        for k in sub_a:
            if not torch.equal(sub_a[k], sub_b[k]):
                return False
    return True


def test_resume_skips_already_completed_experiences(monkeypatch):
    benchmark, train_ds, test_ds = _make_benchmark()

    with tempfile.TemporaryDirectory() as run_dir:
        method1 = get_method("simplecil")()
        Trainer(method1, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()

        # Remove the last checkpoint only -- resume should retrain exactly
        # one experience (the last one), not redo everything.
        checkpoints = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_task"))
        os.remove(os.path.join(run_dir, checkpoints[-1]))

        calls = []
        method2 = get_method("simplecil")()
        original_train_experience = method2.train_experience

        def _tracking_train_experience(exp, loader, ctx):
            calls.append(exp.task_label)
            return original_train_experience(exp, loader, ctx)

        monkeypatch.setattr(method2, "train_experience", _tracking_train_experience)
        Trainer(method2, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()

        assert calls == [benchmark.nb_experiences - 1]
