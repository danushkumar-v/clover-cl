"""Method plugins: CLMethod implementations (SPEC §6).

``simplecil`` (P3, SPEC §6.5) is the first registered method; the prompt/
adapter families land in P5/P6.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("method")

register_method = _registry.register
get_method = _registry.get
list_methods = _registry.list

__all__ = ["register_method", "get_method", "list_methods"]

# Imported for registration side effects only; must come after the
# register_method binding above, since the module imports it back.
from clover.methods import simple_cil  # noqa: E402

__all__ += ["simple_cil"]
