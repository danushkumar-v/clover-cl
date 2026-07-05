"""Core-owned Trainer: loop, AMP, seeding, checkpoints (SPEC §6.2)."""

from __future__ import annotations

from clover.training.trainer import ExperienceDataset, RunConfig, Trainer

__all__ = ["Trainer", "RunConfig", "ExperienceDataset"]
