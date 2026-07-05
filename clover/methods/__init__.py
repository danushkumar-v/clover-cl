"""Method plugins: CLMethod implementations (SPEC §6).

This module only wires up the registry. ``base.py`` (CLMethod ABC),
``heads.py``, and ``losses.py`` are filled in starting P2/P3; concrete
methods (SimpleCIL, L2P, ...) register against this registry from P3 on.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("method")

register_method = _registry.register
get_method = _registry.get
list_methods = _registry.list

__all__ = ["register_method", "get_method", "list_methods"]
