"""Inline ``dataset: {type: ..., root: ..., num_classes: ...}`` parsing
(SPEC §7's config-only path for ``image_folder``)."""

from __future__ import annotations

import pytest

from clover.config.schema import StreamSection


def test_inline_dataset_mapping_sets_type_root_and_num_classes():
    section = StreamSection.from_dict(
        {
            "dataset": {"type": "image_folder", "root": "/data/my_plants", "num_classes": 7},
            "init_cls": 2,
            "increment": 2,
        }
    )
    assert section.dataset == "image_folder"
    assert section.data_root == "/data/my_plants"
    assert section.dataset_num_classes == 7


def test_inline_dataset_mapping_requires_type():
    with pytest.raises(ValueError, match="missing required key 'type'"):
        StreamSection.from_dict(
            {
                "dataset": {"root": "/data/my_plants", "num_classes": 7},
                "init_cls": 2,
                "increment": 2,
            }
        )


def test_inline_dataset_mapping_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown inline dataset config key 'roott'.*did you mean 'root'"):
        StreamSection.from_dict(
            {
                "dataset": {"type": "image_folder", "roott": "/data/my_plants", "num_classes": 7},
                "init_cls": 2,
                "increment": 2,
            }
        )


def test_string_dataset_leaves_dataset_num_classes_none():
    section = StreamSection.from_dict({"dataset": "synthetic", "init_cls": 2, "increment": 2})
    assert section.dataset == "synthetic"
    assert section.dataset_num_classes is None
    assert section.data_root == "./data"


def test_dataset_num_classes_round_trips_through_to_dict():
    section = StreamSection.from_dict(
        {
            "dataset": {"type": "image_folder", "root": "/data/my_plants", "num_classes": 7},
            "init_cls": 2,
            "increment": 2,
        }
    )
    as_dict = section.to_dict()
    assert as_dict["dataset_num_classes"] == 7
    assert as_dict["data_root"] == "/data/my_plants"
