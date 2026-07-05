"""Backbone plugins: config-selectable base models + ViT wrappers (SPEC §6.4).

``tiny_mlp`` (P2, SPEC §12.2) is a fast CPU stub for tests; the real
config-selectable registry (timm/HF names) and prompt-pool/prefix/adapter
wrappers land in P5.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("backbone")

register_backbone = _registry.register
get_backbone = _registry.get
list_backbones = _registry.list

__all__ = ["register_backbone", "get_backbone", "list_backbones"]

# Imported for registration side effects only; must come after the
# register_backbone binding above, since the module imports it back.
from clover.backbones import tiny_mlp  # noqa: E402

__all__ += ["tiny_mlp"]
