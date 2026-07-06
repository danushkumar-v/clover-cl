"""Backbone plugins: config-selectable base models + ViT wrappers (SPEC §6.4).

``tiny_mlp`` (P2) and ``tiny_vit``/``vit_prompt_pool`` (P5) are fast CPU
stand-ins for tests; the real config-selectable base-model resolution
(timm/HF name or a user class) lives in ``clover/backbones/loader.py``.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("backbone")

register_backbone = _registry.register
get_backbone = _registry.get
list_backbones = _registry.list

__all__ = ["register_backbone", "get_backbone", "list_backbones"]

# Imported for registration side effects only; must come after the
# register_backbone binding above, since these modules import it back.
from clover.backbones import (  # noqa: E402
    adapter,
    adapter_ease,
    adapter_mos,
    adapter_tuna,
    coda_prompt,
    dual_prompt,
    prompt_pool,
    tiny_mlp,
    vit,
)

__all__ += [
    "tiny_mlp",
    "vit",
    "prompt_pool",
    "dual_prompt",
    "coda_prompt",
    "adapter",
    "adapter_ease",
    "adapter_mos",
    "adapter_tuna",
]
