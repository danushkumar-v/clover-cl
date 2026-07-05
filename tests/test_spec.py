"""Structural validation failure modes for StreamSpec/RevisitSpec (SPEC §3.2)."""

from __future__ import annotations

import pytest

from clover.core.spec import ImageRelation, RevisitSpec, StreamSpec


def _base_spec_dict(**overrides):
    d = {"dataset": "synthetic", "init_cls": 5, "increment": 5}
    d.update(overrides)
    return d


def test_valid_spec_round_trips_through_dict():
    spec = StreamSpec.from_dict(
        _base_spec_dict(
            revisits=[
                {"classes": "task0", "placement": "end_of_stream", "label": "new", "images": "new"}
            ]
        )
    )
    assert spec.to_dict() == StreamSpec.from_dict(spec.to_dict()).to_dict()


def test_unknown_top_level_key_rejected_with_suggestion():
    with pytest.raises(ValueError, match="unknown stream spec key 'incrament'.*did you mean 'increment'"):
        StreamSpec.from_dict(_base_spec_dict(incrament=5))


def test_unknown_revisit_key_rejected_with_suggestion():
    with pytest.raises(ValueError, match="unknown revisit spec key 'palcement'.*did you mean 'placement'"):
        StreamSpec.from_dict(
            _base_spec_dict(revisits=[{"classes": "task0", "palcement": "end_of_stream"}])
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"init_cls": 0},
        {"increment": 0},
        {"task_size": "shrink"},
    ],
)
def test_bad_stream_field_rejected(overrides):
    with pytest.raises(ValueError):
        StreamSpec.from_dict(_base_spec_dict(**overrides)).validate()


@pytest.mark.parametrize(
    "revisit_overrides",
    [
        {"placement": "sometimes"},
        {"label": "maybe"},
        {"images": "duplicate"},
        {"images": "partial:1.5"},
        {"min_gap": 0},
        {"times": 0},
    ],
)
def test_bad_revisit_field_rejected(revisit_overrides):
    raw = {"classes": "task0", **revisit_overrides}
    with pytest.raises(ValueError):
        RevisitSpec.from_dict(raw)


def test_classes_random_dict_must_be_exactly_random_key():
    with pytest.raises(ValueError, match="classes dict must be exactly"):
        RevisitSpec.from_dict({"classes": {"radnom": 3}})


def test_classes_list_must_not_be_empty():
    with pytest.raises(ValueError, match="classes list must not be empty"):
        RevisitSpec(classes=[]).validate()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("same", ImageRelation("same")),
        ("new", ImageRelation("new")),
        ("partial:0.3", ImageRelation("partial", 0.3)),
        ("partial:1.0", ImageRelation("partial", 1.0)),
        ("partial:0", ImageRelation("partial", 0.0)),
    ],
)
def test_image_relation_parses(raw, expected):
    assert ImageRelation.parse(raw) == expected


@pytest.mark.parametrize("raw", ["disjoint", "partial:", "partial:2", "partial:-0.1", "random"])
def test_image_relation_rejects_bad_strings(raw):
    with pytest.raises(ValueError):
        ImageRelation.parse(raw)
