"""Tests for clover/core/overlap_spec.py."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec


# ---------------------------------------------------------------------------
# Construction and validation — valid configs
# ---------------------------------------------------------------------------

def _make_pair(tasks=(0, 1), classes=None):
    return OverlapPair(tasks=tasks, shared_classes=classes or [0, 1, 2])


def test_none_mode_validates():
    spec = OverlapSpec(mode="none", pairs=[], image_split=ImageSplit(), seed=42)
    spec.validate()  # must not raise


def test_exact_replay_validates():
    spec = OverlapSpec(
        mode="exact_replay",
        pairs=[_make_pair()],
        image_split=ImageSplit(strategy="duplicate"),
    )
    spec.validate()


def test_partial_mode_validates():
    spec = OverlapSpec(mode="partial", pairs=[_make_pair()])
    spec.validate()


@pytest.mark.parametrize(
    "mode",
    [
        "hierarchical",
        "distribution_shift",
        "long_range_revisit",
        "cumulative_drift",
        "symmetric_pair",
    ],
)
def test_all_modes_validate(mode):
    spec = OverlapSpec(mode=mode, pairs=[_make_pair()])
    spec.validate()


def test_near_miss_empty_shared_classes_ok():
    spec = OverlapSpec(
        mode="near_miss",
        pairs=[OverlapPair(tasks=(0, 1), shared_classes=[])],
    )
    spec.validate()


# ---------------------------------------------------------------------------
# Validation — invalid configs
# ---------------------------------------------------------------------------

def test_none_mode_with_pairs_raises():
    spec = OverlapSpec(mode="none", pairs=[_make_pair()])
    with pytest.raises(ValueError, match="empty pairs"):
        spec.validate()


def test_negative_task_index_raises():
    spec = OverlapSpec(mode="partial", pairs=[_make_pair(tasks=(-1, 1))])
    with pytest.raises(ValueError, match="negative task index"):
        spec.validate()


def test_empty_shared_classes_raises_for_non_near_miss():
    spec = OverlapSpec(
        mode="partial",
        pairs=[OverlapPair(tasks=(0, 1), shared_classes=[])],
    )
    with pytest.raises(ValueError, match="shared_classes is empty"):
        spec.validate()


def test_invalid_ratio_raises():
    spec = OverlapSpec(
        mode="partial",
        pairs=[_make_pair()],
        image_split=ImageSplit(strategy="disjoint", ratio=1.5),
    )
    with pytest.raises(ValueError, match="ratio"):
        spec.validate()


def test_invalid_overlap_pct_raises():
    spec = OverlapSpec(
        mode="partial",
        pairs=[_make_pair()],
        image_split=ImageSplit(strategy="partial_duplicate", overlap_pct=-0.1),
    )
    with pytest.raises(ValueError, match="overlap_pct"):
        spec.validate()


def test_unknown_mode_raises():
    spec = OverlapSpec(mode="unknown_xyz")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown mode"):
        spec.validate()


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def test_yaml_round_trip():
    spec = OverlapSpec(
        mode="partial",
        pairs=[_make_pair(tasks=(0, 3), classes=[2, 5, 7])],
        image_split=ImageSplit(strategy="partial_duplicate", ratio=0.6, overlap_pct=0.2),
        seed=99,
    )
    raw = {
        "overlap_spec": {
            "mode": spec.mode,
            "seed": spec.seed,
            "image_split": {
                "strategy": spec.image_split.strategy,
                "ratio": spec.image_split.ratio,
                "overlap_pct": spec.image_split.overlap_pct,
            },
            "pairs": [
                {"tasks": list(p.tasks), "shared_classes": p.shared_classes}
                for p in spec.pairs
            ],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(raw, fh)
        tmp_path = fh.name

    try:
        loaded = OverlapSpec.from_yaml(tmp_path)
        assert loaded.mode == spec.mode
        assert loaded.seed == spec.seed
        assert loaded.image_split.strategy == spec.image_split.strategy
        assert loaded.image_split.ratio == pytest.approx(spec.image_split.ratio)
        assert loaded.image_split.overlap_pct == pytest.approx(spec.image_split.overlap_pct)
        assert len(loaded.pairs) == len(spec.pairs)
        assert loaded.pairs[0].tasks == spec.pairs[0].tasks
        assert loaded.pairs[0].shared_classes == spec.pairs[0].shared_classes
    finally:
        os.unlink(tmp_path)


def test_to_dict_round_trip():
    spec = OverlapSpec(
        mode="exact_replay",
        pairs=[_make_pair()],
        image_split=ImageSplit(strategy="duplicate"),
        seed=7,
    )
    d = spec.to_dict()
    assert d["mode"] == "exact_replay"
    assert d["seed"] == 7
    assert d["image_split"]["strategy"] == "duplicate"
    assert len(d["pairs"]) == 1
