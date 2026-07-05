"""Reproducibility manifests: feature-cache keys include the manifest hash."""

from __future__ import annotations

import json
from typing import Any


def save_manifest(manifest: dict[str, Any], path: str) -> None:
    """Write a manifest dict to *path* as pretty-printed JSON."""
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def load_manifest(path: str) -> dict[str, Any]:
    """Load and return a manifest from a JSON file.

    Raises:
        ValueError: If the file is missing mandatory header keys.
    """
    with open(path) as fh:
        data = json.load(fh)
    _validate_manifest(data)
    return data


def _validate_manifest(data: dict[str, Any]) -> None:
    if "_header" not in data:
        return
    header = data["_header"]
    for key in ("clover_version", "dataset", "init_cls", "increment"):
        if key not in header:
            raise ValueError(
                f"Manifest header is missing required key {key!r}. "
                "The file may be corrupted or from an incompatible version."
            )
