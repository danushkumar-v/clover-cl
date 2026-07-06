"""EASE's mechanism (SPEC §6.5): a growing per-experience list of
bottleneck adapters, all but the most recent one frozen. Confirmed via
read-only research on ``models/ease.py``/``backbone/vit_ease.py``:
``after_task()`` always freezes the just-trained adapter and appends it to
a running list, then allocates a fresh one for the next task -- never
sharing or growing a single adapter set the way MOS/TUNA do. At evaluation
time, the *same* image is passed through every stored adapter and the
resulting ``[CLS]`` features are concatenated, so the feature width grows
by one adapter's worth every experience.

``grow()`` is called once per experience from ``EASE.before_experience``
(not ``after_experience``): the Trainer's resume-replay only re-calls
``before_experience`` for already-completed experiences, so growth (a new
adapter set appended, the previous one frozen) has to live there to
reconstruct the right structure before ``load_state_dict`` runs -- the same
reason ``IncrementalHead.expand_to``/CODA-Prompt's ``pool.start_new_task``
are called from ``before_experience`` rather than ``train_experience``.
"""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from clover.backbones import register_backbone
from clover.backbones.adapter import Adapter
from clover.backbones.loader import resolve_base_model


class EaseAdapterViT(nn.Module):
    """Frozen base ViT + a growing list of per-experience adapter sets.

    Every entry in ``adapter_sets`` is itself an ``nn.ModuleList`` of one
    ``Adapter`` per transformer block (the same per-block-adapter shape
    ``AdapterViT`` uses for a single adapter). Only the *last* entry is
    ever trainable; ``grow()`` freezes it and appends a fresh one.
    """

    def __init__(self, base: nn.Module, bottleneck_dim: int = 8, scale: float = 0.1) -> None:
        super().__init__()
        self.depth = len(base.blocks)  # type: ignore[arg-type]
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()

        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        self._bottleneck_dim = bottleneck_dim
        self._scale = scale
        self.adapter_sets = nn.ModuleList()

    def _new_adapter_set(self) -> nn.ModuleList:
        return nn.ModuleList(
            [Adapter(self.feature_dim, self._bottleneck_dim, self._scale) for _ in range(self.depth)]
        )

    @property
    def num_blocks(self) -> int:
        return len(self.adapter_sets)

    @property
    def cur_adapter(self) -> nn.ModuleList:
        if len(self.adapter_sets) == 0:
            raise RuntimeError("grow() must be called before cur_adapter is accessed.")
        return cast(nn.ModuleList, self.adapter_sets[-1])

    def grow(self) -> None:
        """Freeze the current (most recent) adapter set, if any, then
        allocate a fresh trainable one for the upcoming experience."""
        if len(self.adapter_sets) > 0:
            self.adapter_sets[-1].requires_grad_(False)
            self.adapter_sets[-1].eval()
        self.adapter_sets.append(self._new_adapter_set())

    def _forward_with(self, x: torch.Tensor, adapter_set: nn.ModuleList) -> torch.Tensor:
        adapter: Dict[int, nn.Module] = dict(enumerate(adapter_set))
        return self.base(x, adapter=adapter)

    def forward_current(self, x: torch.Tensor) -> torch.Tensor:
        """Single-adapter pass through ``cur_adapter`` -- used only to
        gradient-train the current experience's adapter."""
        return self._forward_with(x, self.cur_adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenation of every stored adapter set's feature --
        ``[batch, num_blocks, feature_dim]`` -- for the growing-dim head."""
        if self.num_blocks == 0:
            raise RuntimeError("EaseAdapterViT.forward() called before grow().")
        features = [
            self._forward_with(x, cast(nn.ModuleList, adapter_set)) for adapter_set in self.adapter_sets
        ]
        return torch.stack(features, dim=1)


@register_backbone("vit_adapter_ease")
def vit_adapter_ease(
    base_model: str = "tiny_vit", bottleneck_dim: int = 8, scale: float = 0.1, **base_kwargs: Any
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return EaseAdapterViT(base, bottleneck_dim=bottleneck_dim, scale=scale)
