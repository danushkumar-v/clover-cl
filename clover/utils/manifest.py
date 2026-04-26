"""Manifest serialisation and deserialisation for CLOVER."""

from __future__ import annotations

import json
from typing import Any, Dict


def save_manifest(manifest: Dict[str, Any], path: str) -> None:
    """Write a manifest dict to *path* as pretty-printed JSON.

    Args:
        manifest: Dict as returned by :meth:`OverlapDataManager.get_manifest`.
        path: Destination file path.
    """
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def load_manifest(path: str) -> Dict[str, Any]:
    """Load and return a manifest from a JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        The manifest dict.  Raises :class:`ValueError` if the file is missing
        mandatory top-level keys.
    """
    with open(path, "r") as fh:
        data = json.load(fh)

    _validate_manifest(data)
    return data


def _validate_manifest(data: Dict[str, Any]) -> None:
    """Raise :class:`ValueError` if *data* is missing expected structure."""
    if "_header" not in data and "train" not in data:
        # Older flat format (task_id -> class_id -> indices)
        return
    if "_header" in data:
        header = data["_header"]
        for key in ("clover_version", "dataset", "init_cls", "increment"):
            if key not in header:
                raise ValueError(
                    f"Manifest header is missing required key {key!r}. "
                    "The file may be corrupted or from an incompatible version."
                )
