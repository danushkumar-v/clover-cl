"""Method plugins: CLMethod implementations (SPEC §6).

``simplecil`` (P3) and the full prompt trio -- ``l2p``, ``dualprompt``,
``coda_prompt`` (P5) -- are registered. The adapter family (APER-Adapter,
EASE, RanPAC, MOS, TUNA, P6) is now complete: all 9 methods registered.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("method")

register_method = _registry.register
get_method = _registry.get
list_methods = _registry.list

__all__ = ["register_method", "get_method", "list_methods"]

# Imported for registration side effects only; must come after the
# register_method binding above, since these modules import it back.
from clover.methods import (  # noqa: E402
    aper_adapter,
    coda_prompt,
    dual_prompt,
    ease,
    l2p,
    mos,
    ranpac,
    simple_cil,
    tuna,
)

__all__ += [
    "simple_cil",
    "l2p",
    "dual_prompt",
    "coda_prompt",
    "aper_adapter",
    "ranpac",
    "ease",
    "mos",
    "tuna",
]
