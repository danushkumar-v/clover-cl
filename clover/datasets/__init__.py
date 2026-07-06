"""Dataset plugins: CLDataset ABC (SPEC §7).

``synthetic`` (P2, SPEC §12.2) plus the 6 real built-ins (P8): CIFAR-100,
CUB-200, ImageNet-R/A, OmniBenchmark, VTAB.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("dataset")

register_dataset = _registry.register
get_dataset = _registry.get
list_datasets = _registry.list

__all__ = ["register_dataset", "get_dataset", "list_datasets"]

# Imported for registration side effects only; must come after the
# register_dataset binding above, since these modules import it back.
from clover.datasets import (  # noqa: E402
    cifar100,
    cub200,
    imagenet_a,
    imagenet_r,
    omnibench,
    synthetic,
    vtab,
)

__all__ += ["synthetic", "cifar100", "cub200", "imagenet_r", "imagenet_a", "omnibench", "vtab"]
