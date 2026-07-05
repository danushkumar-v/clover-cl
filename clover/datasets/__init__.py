"""Dataset plugins: CLDataset ABC (SPEC §7).

``synthetic`` (P2, SPEC §12.2) is the only built-in for now; the real
datasets (CIFAR-100, CUB-200, ImageNet-R/A, OmniBenchmark, VTAB) land in
P7/P8 on top of the same ``CLDataset`` ABC.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("dataset")

register_dataset = _registry.register
get_dataset = _registry.get
list_datasets = _registry.list

__all__ = ["register_dataset", "get_dataset", "list_datasets"]

# Imported for registration side effects only; must come after the
# register_dataset binding above, since the module imports it back.
from clover.datasets import synthetic  # noqa: E402

__all__ += ["synthetic"]
