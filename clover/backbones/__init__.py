"""Backbone plugins: config-selectable base models + ViT wrappers (SPEC §6.4).

This module only wires up the registry; wrappers (prompt-pool, prefix,
adapter) are added in P5.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("backbone")

register_backbone = _registry.register
get_backbone = _registry.get
list_backbones = _registry.list

__all__ = ["register_backbone", "get_backbone", "list_backbones"]
