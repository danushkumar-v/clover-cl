"""Scenario plugins: named StreamSpec factories (SPEC §8).

This module only wires up the registry; scenario factories are added
starting P1 (the 6 core scenarios) with extras in P8.
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("scenario")

register_scenario = _registry.register
get_scenario = _registry.get
list_scenarios = _registry.list

__all__ = ["register_scenario", "get_scenario", "list_scenarios"]
