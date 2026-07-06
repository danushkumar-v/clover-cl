"""Scenario plugins: named StreamSpec factories (SPEC §8).

The 6 core scenarios (P1) are imported below to trigger their
``@register_scenario`` registration; ``distribution_shift`` is P8's one
extra (SPEC §8 permits the rest -- ``symmetric_pair``/``near_miss``/
``hierarchical`` -- to stay out of scope; see docs/concepts.md).
"""

from __future__ import annotations

from clover.utils.registry import Registry

_registry: Registry = Registry("scenario")

register_scenario = _registry.register
get_scenario = _registry.get
list_scenarios = _registry.list

__all__ = ["register_scenario", "get_scenario", "list_scenarios"]

# Imported for registration side effects only; must come after the
# register_scenario binding above, since each module imports it back.
from clover.scenarios import (  # noqa: E402
    cumulative_drift,
    disjoint_baseline,
    distribution_shift,
    exact_replay,
    long_range_revisit,
    mid_range_revisit,
    partial_overlap,
)

__all__ += [
    "cumulative_drift",
    "disjoint_baseline",
    "distribution_shift",
    "exact_replay",
    "long_range_revisit",
    "mid_range_revisit",
    "partial_overlap",
]
