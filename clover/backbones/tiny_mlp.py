"""2-layer backbone stub (SPEC §12.2): fast, CPU, feature extractor for tests.

Hosts the revisit-safety gate (§5.3), the smoke profile, and trainer/resume
tests in later phases. The real config-selectable backbone registry
(timm/HF names, prompt-pool/prefix/adapter wrappers) lands in P5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from clover.backbones import register_backbone


@register_backbone("tiny_mlp")
class TinyMLP(nn.Module):
    """``Linear -> ReLU -> Linear`` feature extractor over flattened input."""

    def __init__(
        self, input_size: int = 8, in_chans: int = 1, hidden_dim: int = 32, feature_dim: int = 16
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * input_size * in_chans, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
