"""``per_task.csv`` correctness (SPEC §10-§11): standard + CLOVER metrics
written every experience, resume-safe against stale rows for a redone task.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile

from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method
from clover.training import RunConfig, Trainer

_STANDARD_METRICS = {"A_t", "AIA", "BWT", "Forgetting", "FWT"}
_ALWAYS_ON_CLOVER_METRICS = {"RAG_mean", "Repetition_Gain"}
_FINAL_ONLY_METRICS = {"Anchor_Retention", "Long_Range_Retention"}


def _make_benchmark():
    spec = StreamSpec(dataset="synthetic", init_cls=4, increment=4)
    info = DatasetInfo("synthetic", 20)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())
    return benchmark, train_ds, test_ds


def _read_rows(run_dir):
    with open(os.path.join(run_dir, "per_task.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        return [(int(r["task_idx"]), r["metric"], r["value"]) for r in reader]


def test_per_task_csv_has_standard_and_clover_metrics_every_task():
    benchmark, train_ds, test_ds = _make_benchmark()
    method = get_method("simplecil")()

    with tempfile.TemporaryDirectory() as run_dir:
        Trainer(method, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()
        rows = _read_rows(run_dir)

    final_task = benchmark.nb_experiences - 1
    metrics_by_task: dict[int, set[str]] = {}
    for task_idx, metric, _value in rows:
        metrics_by_task.setdefault(task_idx, set()).add(metric)

    for t in range(final_task + 1):
        present = metrics_by_task[t]
        assert _STANDARD_METRICS <= present
        assert _ALWAYS_ON_CLOVER_METRICS <= present
        if t == final_task:
            assert _FINAL_ONLY_METRICS <= present
        else:
            assert not (_FINAL_ONLY_METRICS & present)


def test_per_task_csv_values_are_finite_or_nan_never_error():
    benchmark, train_ds, test_ds = _make_benchmark()
    method = get_method("simplecil")()

    with tempfile.TemporaryDirectory() as run_dir:
        Trainer(method, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()
        rows = _read_rows(run_dir)

    for _task_idx, _metric, value in rows:
        parsed = float(value)
        assert math.isfinite(parsed) or math.isnan(parsed)


def test_resume_does_not_duplicate_stale_rows_for_the_redone_task():
    benchmark, train_ds, test_ds = _make_benchmark()

    with tempfile.TemporaryDirectory() as run_dir:
        method1 = get_method("simplecil")()
        Trainer(method1, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()

        # Simulate a crash *between* the per_task.csv write and the
        # checkpoint write for the last task: delete only the checkpoint,
        # leaving per_task.csv's stale row for that task in place.
        checkpoints = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_task"))
        os.remove(os.path.join(run_dir, checkpoints[-1]))

        method2 = get_method("simplecil")()
        Trainer(method2, benchmark, train_ds, test_ds, RunConfig(run_dir=run_dir, seed=42)).run()
        rows = _read_rows(run_dir)

    final_task = benchmark.nb_experiences - 1
    final_task_rows = [r for r in rows if r[0] == final_task]
    metric_names = [r[1] for r in final_task_rows]
    assert len(metric_names) == len(set(metric_names)), f"duplicate metric rows for the redone task: {metric_names}"


def test_resume_reproduces_the_same_per_task_csv_as_an_uninterrupted_run():
    benchmark, train_ds, test_ds = _make_benchmark()

    with tempfile.TemporaryDirectory() as full_dir:
        Trainer(
            get_method("simplecil")(), benchmark, train_ds, test_ds, RunConfig(run_dir=full_dir, seed=42)
        ).run()
        full_rows = sorted(_read_rows(full_dir))

    with tempfile.TemporaryDirectory() as resumed_dir:
        Trainer(
            get_method("simplecil")(), benchmark, train_ds, test_ds, RunConfig(run_dir=resumed_dir, seed=42)
        ).run()
        checkpoints = sorted(f for f in os.listdir(resumed_dir) if f.startswith("ckpt_task"))
        for f in checkpoints[-2:]:
            os.remove(os.path.join(resumed_dir, f))
        Trainer(
            get_method("simplecil")(), benchmark, train_ds, test_ds, RunConfig(run_dir=resumed_dir, seed=42)
        ).run()
        resumed_rows = sorted(_read_rows(resumed_dir))

    assert full_rows == resumed_rows
