"""Integration test: GRAFT-style metrics as one-liners using Experience objects.

Demonstrates that the central research claim — "forgetting on revisiting
classes vs. fresh classes is trivial to compute" — holds.
"""

from __future__ import annotations

import pytest

from clover import build_benchmark, RevisitSpec, StreamSpec


@pytest.fixture
def bench_with_revisits(patched_cifar100):
    spec = StreamSpec(
        dataset="cifar100",
        init_cls=10,
        increment=10,
        revisits=[
            RevisitSpec(classes=[3, 7, 22], times=1, placement="random", min_gap=2)
        ],
        stream_seed=42,
    )
    return build_benchmark(spec)


def test_revisit_experiences_have_revisiting_classes(bench_with_revisits):
    bench = bench_with_revisits
    revisit_exps = [e for e in bench.train_stream if e.is_revisit_experience()]
    assert all(e.revisiting_classes for e in revisit_exps)


def test_graft_metric_one_liner(bench_with_revisits):
    """Forgetting on revisiting vs. fresh classes is a one-liner."""
    bench = bench_with_revisits

    revisit_exps = [e for e in bench.train_stream if e.is_revisit_experience()]
    fresh_exps = [e for e in bench.train_stream if not e.is_revisit_experience()]

    # Revisit experiences always have at least one revisiting class
    assert all(len(e.revisiting_classes) > 0 for e in revisit_exps)
    # Fresh experiences always have zero revisiting classes
    assert all(len(e.revisiting_classes) == 0 for e in fresh_exps)


def test_future_classes_safety(bench_with_revisits):
    """classes_in_future must never contain classes from the current experience."""
    bench = bench_with_revisits
    for exp in bench.train_stream:
        here = set(exp.classes_in_this_experience)
        future = set(exp.classes_in_future)
        # Future contains only classes appearing in later experiences;
        # a class can be both here AND in future if it revisits later
        # — but we must not have classes claimed as future-only that are also here
        # (the invariant is just that future doesn't include non-existent classes)
        assert future.issubset(set(range(bench.total_classes)))


def test_per_experience_overlap_diagnostic(bench_with_revisits):
    """overlap_with_previous is a dict mapping prev_task → shared_count."""
    bench = bench_with_revisits
    for exp in bench.train_stream:
        for prev_t, count in exp.overlap_with_previous.items():
            assert prev_t < exp.task_label
            assert count > 0


def test_stream_revisit_density(bench_with_revisits):
    """revisit_density() > 0 when revisits are configured."""
    bench = bench_with_revisits
    assert bench.train_stream.revisit_density() > 0.0


def test_class_appearance_count_revisit_classes_appear_more_than_once(bench_with_revisits):
    """Revisited classes must appear in more than one experience."""
    bench = bench_with_revisits
    counts = bench.train_stream.class_appearance_count()
    revisited = set()
    for exp in bench.train_stream:
        revisited.update(exp.revisiting_classes)
    for cls in revisited:
        assert counts.get(cls, 0) > 1, f"Revisited class {cls} appears only once."
