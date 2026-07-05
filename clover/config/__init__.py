"""Typed config schema, loading, validation, resolution (SPEC §9)."""

from __future__ import annotations

from clover.config.loader import ResolvedConfig, load_yaml, resolve_config
from clover.config.schema import (
    MethodSection,
    OptimizerConfig,
    RunSection,
    StreamSection,
    TrainingSection,
)

__all__ = [
    "ResolvedConfig",
    "load_yaml",
    "resolve_config",
    "RunSection",
    "StreamSection",
    "MethodSection",
    "TrainingSection",
    "OptimizerConfig",
]
