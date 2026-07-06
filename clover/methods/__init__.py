"""Method plugins: CLMethod implementations (SPEC §6).

``simplecil`` (P3) and the full prompt trio -- ``l2p``, ``dualprompt``,
``coda_prompt`` (P5) -- are registered. The adapter family (APER-Adapter,
EASE, RanPAC, MOS, TUNA) lands in P6.
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
from clover.methods import coda_prompt, dual_prompt, l2p, simple_cil  # noqa: E402

__all__ += ["simple_cil", "l2p", "dual_prompt", "coda_prompt"]
