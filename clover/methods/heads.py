"""Incremental classifier heads (SPEC §6.2): implemented once, shared by all methods."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalHead(nn.Module):
    """A linear classifier head that grows its class dimension over time.

    Args:
        feature_dim: Backbone feature width.
        cosine: If ``True``, L2-normalizes both features and weight rows and
            scales the resulting cosine similarities by a learned scalar
            (SimpleCIL's mode: each row *is* a class prototype direction).
            If ``False``, a plain linear layer.
        num_classes: Initial width (usually 0; grown via ``expand_to``).
        scale: Initial value of the learned cosine scale (ignored if
            ``cosine=False``).
    """

    def __init__(
        self, feature_dim: int, cosine: bool = False, num_classes: int = 0, scale: float = 10.0
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cosine = cosine
        self.weight = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.scale = nn.Parameter(torch.tensor(float(scale))) if cosine else None

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    def expand_to(self, new_num_classes: int) -> None:
        """Grow the head to *new_num_classes*, preserving existing rows.

        No-op if the head is already at least that wide (idempotent --
        class counts never shrink in continual learning).
        """
        if new_num_classes <= self.num_classes:
            return
        old_weight = self.weight.data
        new_weight = torch.zeros(new_num_classes, self.feature_dim, dtype=old_weight.dtype)
        new_weight[: old_weight.shape[0]] = old_weight
        nn.init.normal_(new_weight[old_weight.shape[0] :], std=0.01)
        self.weight = nn.Parameter(new_weight)

    def set_prototype(self, class_id: int, vector: torch.Tensor) -> None:
        """Directly assign class *class_id*'s weight row (closed-form, no grad)."""
        if class_id >= self.num_classes:
            raise ValueError(
                f"class_id {class_id} exceeds current head width {self.num_classes}; "
                "call expand_to first."
            )
        with torch.no_grad():
            self.weight.data[class_id] = vector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.cosine:
            features = F.normalize(features, dim=-1)
            weight = F.normalize(self.weight, dim=-1)
            assert self.scale is not None
            return self.scale * F.linear(features, weight)
        return F.linear(features, self.weight)
