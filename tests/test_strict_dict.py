"""Shared unknown-key rejection helper (used by spec.py and config schemas)."""

from __future__ import annotations

import pytest

from clover.utils.strict_dict import reject_unknown_keys


def test_no_unknown_keys_is_a_no_op():
    reject_unknown_keys({"a": 1, "b": 2}, frozenset({"a", "b", "c"}), "widget")


def test_unknown_key_raises_with_suggestion():
    with pytest.raises(ValueError, match="unknown widget key 'colour'.*did you mean 'color'"):
        reject_unknown_keys({"color": 1, "colour": 2}, frozenset({"color", "size"}), "widget")


def test_unknown_key_without_close_match_has_no_suggestion():
    with pytest.raises(ValueError, match=r"unknown widget key 'zzz'\. allowed keys:"):
        reject_unknown_keys({"zzz": 1}, frozenset({"color", "size"}), "widget")


def test_multiple_unknown_keys_all_named():
    with pytest.raises(ValueError, match="'colour'") as exc_info:
        reject_unknown_keys(
            {"colour": 1, "sizee": 2}, frozenset({"color", "size"}), "widget"
        )
    assert "'sizee'" in str(exc_info.value)


def test_allowed_keys_listed_in_message():
    with pytest.raises(ValueError, match=r"allowed keys: \['color', 'size'\]"):
        reject_unknown_keys({"zzz": 1}, frozenset({"color", "size"}), "widget")
