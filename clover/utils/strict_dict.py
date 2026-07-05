"""Shared unknown-key rejection with "did you mean" suggestions.

Used by ``StreamSpec``/``RevisitSpec`` (``clover/core/spec.py``) and the
config schemas (``clover/config/schema.py``) — one implementation of a
pattern needed in 3+ places now, per the registries' share-one-
implementation convention (``clover/utils/registry.py``).
"""

from __future__ import annotations

import difflib
from typing import Mapping


def reject_unknown_keys(raw: Mapping[str, object], allowed: frozenset[str], kind: str) -> None:
    """Raise ``ValueError`` naming every key in *raw* not in *allowed*.

    Args:
        raw: The dict being validated (e.g. loaded from YAML).
        allowed: The complete set of permitted keys.
        kind: Describes what's being validated, e.g. ``"stream spec"`` or
            ``"run config"`` — used in the error message.
    """
    unknown = set(raw) - allowed
    if not unknown:
        return
    parts = []
    for key in sorted(unknown):
        suggestion = difflib.get_close_matches(key, allowed, n=1)
        hint = f" did you mean {suggestion[0]!r}?" if suggestion else ""
        parts.append(f"unknown {kind} key {key!r}.{hint}")
    raise ValueError(" ".join(parts) + f" allowed keys: {sorted(allowed)}")
