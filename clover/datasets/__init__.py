"""Dataset plugins: CLDataset ABC + built-ins land in P7/P8 (SPEC §7).

This module only wires up the registry so later phases have a stable place
to register against.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("dataset")

register_dataset = _registry.register
get_dataset = _registry.get
list_datasets = _registry.list

__all__ = ["register_dataset", "get_dataset", "list_datasets"]
