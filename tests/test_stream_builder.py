"""Tests for clover/core/stream_builder.py."""

from __future__ import annotations

import pytest

from clover import build_benchmark, RevisitSpec, StreamSpec


def _base_spec(**kwargs):
    defaults = dict(
        dataset="cifar100",
        init_cls=10,
        increment=10,
        stream_seed=42,
        revisits=[],
    )
    defaults.update(kwargs)
    return StreamSpec(**defaults)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture
def revisit_spec(patched_cifar100):
    return _base_spec(
        revisits=[RevisitSpec(classes=[3, 7], times=1, placement="random", min_gap=2)]
    )


def test_same_spec_same_seed_byte_identical(revisit_spec):
    b1 = build_benchmark(revisit_spec)
    b2 = build_benchmark(revisit_spec)
    for e1, e2 in zip(b1.train_stream, b2.train_stream):
        assert e1.classes_in_this_experience == e2.classes_in_this_experience
        assert e1.revisiting_classes == e2.revisiting_classes
        assert e1.image_indices == e2.image_indices


def test_different_stream_seed_different_placement(patched_cifar100):
    rv = RevisitSpec(classes=3, times=1, placement="random", min_gap=2)
    b1 = build_benchmark(_base_spec(revisits=[rv], stream_seed=1))
    b2 = build_benchmark(_base_spec(revisits=[rv], stream_seed=2))
    # With random placement and different seeds, placement should differ
    placements_1 = [e.revisiting_classes for e in b1.train_stream]
    placements_2 = [e.revisiting_classes for e in b2.train_stream]
    assert placements_1 != placements_2


def test_same_stream_seed_same_structure(patched_cifar100):
    rv = RevisitSpec(classes=3, times=1, placement="random", min_gap=2)
    b1 = build_benchmark(_base_spec(revisits=[rv], stream_seed=99))
    b2 = build_benchmark(_base_spec(revisits=[rv], stream_seed=99))
    n_revisit_1 = sum(1 for e in b1.train_stream if e.is_revisit_experience())
    n_revisit_2 = sum(1 for e in b2.train_stream if e.is_revisit_experience())
    assert n_revisit_1 == n_revisit_2


# ---------------------------------------------------------------------------
# Placement strategies
# ---------------------------------------------------------------------------

def test_placement_spaced_distributes_evenly(patched_cifar100):
    rv = RevisitSpec(classes=[5], times=1, placement="spaced", min_gap=1)
    bench = build_benchmark(_base_spec(revisits=[rv]))
    revisit_exps = [e for e in bench.train_stream if 5 in e.revisiting_classes]
    assert len(revisit_exps) == 1
    # Spaced should place near the middle of the stream
    t = revisit_exps[0].task_label
    assert 1 <= t <= bench.nb_experiences - 1


def test_placement_end_of_stream(patched_cifar100):
    rv = RevisitSpec(classes=[0], times=1, placement="end_of_stream", min_gap=1)
    bench = build_benchmark(_base_spec(revisits=[rv]))
    revisit_exps = [e for e in bench.train_stream if 0 in e.revisiting_classes]
    assert len(revisit_exps) == 1
    # end_of_stream places in the last experiences
    assert revisit_exps[0].task_label >= bench.nb_experiences // 2


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def test_unsatisfiable_too_many_revisits_raises(patched_cifar100):
    # class 0 is in experience 0; only ~8 experiences available after gap
    rv = RevisitSpec(classes=[0], times=15, placement="random", min_gap=1)
    with pytest.raises(ValueError, match="[Cc]annot place"):
        build_benchmark(_base_spec(revisits=[rv]))


def test_invalid_class_id_raises(patched_cifar100):
    rv = RevisitSpec(classes=[999], times=1, placement="random", min_gap=1)
    with pytest.raises(ValueError, match="out of range"):
        build_benchmark(_base_spec(revisits=[rv]))


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_benchmark(StreamSpec(dataset="unknownXYZ", init_cls=10, increment=10))


# ---------------------------------------------------------------------------
# Experience metadata correctness
# ---------------------------------------------------------------------------

def test_revisiting_classes_in_classes_in_this_experience(patched_cifar100):
    rv = RevisitSpec(classes=[3, 7], times=1, placement="random", min_gap=2)
    bench = build_benchmark(_base_spec(revisits=[rv]))
    for exp in bench.train_stream:
        assert set(exp.revisiting_classes).issubset(set(exp.classes_in_this_experience))


def test_first_appearance_disjoint_from_revisiting(patched_cifar100):
    rv = RevisitSpec(classes=[3, 7], times=1, placement="random", min_gap=2)
    bench = build_benchmark(_base_spec(revisits=[rv]))
    for exp in bench.train_stream:
        assert not (set(exp.revisiting_classes) & set(exp.first_appearance_of))


def test_classes_seen_so_far_grows_monotonically(patched_cifar100):
    bench = build_benchmark(_base_spec())
    prev_seen = set()
    for exp in bench.train_stream:
        current_seen = set(exp.classes_seen_so_far)
        assert prev_seen.issubset(current_seen)
        prev_seen = current_seen


def test_no_overlap_spec_gives_disjoint_stream(patched_cifar100):
    bench = build_benchmark(_base_spec(revisits=[]))
    for exp in bench.train_stream:
        assert exp.revisiting_classes == []


# ---------------------------------------------------------------------------
# StreamSpec YAML round-trip
# ---------------------------------------------------------------------------

def test_stream_spec_yaml_round_trip(patched_cifar100, tmp_path):
    import yaml

    spec_dict = {
        "dataset": "cifar100",
        "init_cls": 10,
        "increment": 10,
        "shuffle_seed": 1993,
        "stream_seed": 5,
        "image_strategy": "disjoint",
        "image_split_ratio": 0.5,
        "data_root": "./data",
        "revisits": [
            {"classes": [1, 2], "times": 1, "placement": "spaced", "min_gap": 2}
        ],
    }
    cfg = tmp_path / "spec.yaml"
    cfg.write_text(yaml.dump(spec_dict))

    spec = StreamSpec.from_yaml(str(cfg))
    bench = build_benchmark(spec)
    assert bench.nb_experiences == 10
