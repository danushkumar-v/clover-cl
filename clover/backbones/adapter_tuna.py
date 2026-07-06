"""TUNA's mechanism (SPEC §6.5): a growing per-experience adapter list
(like EASE -- a fresh adapter per experience, only the most recent one
trainable), combined via **EMR-merge** (Exclusive Mask and Rescale) into a
single consensus adapter used directly for evaluation, instead of EASE's
concatenation or MOS's continuous EMA blend. Confirmed via read-only
research on ``models/tuna.py``/``backbone/vit_tuna.py`` (``TaskVector``/
``emr_merge``): per parameter tensor, elect a sign by majority vote across
every stored task's adapter, keep the max-magnitude value among the task
vectors agreeing with that sign (zeroing the rest), then rescale to
preserve overall magnitude -- a "resolve sign conflicts, keep the
strongest agreeing evidence" consensus merge.

``grow()`` (freeze the previous entry, allocate a fresh trainable one) is
called from ``TUNA.before_experience``, same resume-safety reason as
EASE's/MOS's equivalents. ``recompute_merge()`` is a *value*-only update
(the merged adapter's shape never changes) and lives in
``TUNA.train_experience``, right after that round's training finishes --
matching PILOT's own cadence (merge right after a task's adapter is
appended, so evaluation of that same round already sees its contribution,
unlike deferring to the next round's ``before_experience``).
"""

from __future__ import annotations

from typing import Any, Dict, cast

import torch
import torch.nn as nn

from clover.backbones import register_backbone
from clover.backbones.adapter import Adapter
from clover.backbones.loader import resolve_base_model

_LAYER_NAMES = ("down_proj", "up_proj")
_PARAM_NAMES = ("weight", "bias")


def _zero_adapter_set(adapter_set: nn.ModuleList) -> None:
    with torch.no_grad():
        for module in adapter_set:
            adapter = cast(Adapter, module)
            adapter.down_proj.weight.zero_()
            adapter.down_proj.bias.zero_()
            adapter.up_proj.weight.zero_()
            adapter.up_proj.bias.zero_()


def _emr_merge(stacked: torch.Tensor) -> torch.Tensor:
    """EMR-merge across the leading (task) dimension of ``stacked``
    (``[T, *shape]``): elect a per-element sign via majority vote among
    the ``T`` task tensors (falling back to the mean's sign on an exact
    tie), keep the max-magnitude value among task tensors agreeing with
    that sign (zero for disagreeing ones), then rescale so the merged
    tensor's average magnitude matches the mean task tensor's average
    magnitude -- electing the max at every position would otherwise
    systematically inflate the merged tensor's overall scale relative to
    any single task's.
    """
    signs = torch.sign(stacked)
    elected_sign = torch.sign(signs.sum(dim=0))
    mean_vec = stacked.mean(dim=0)
    elected_sign = torch.where(elected_sign == 0, torch.sign(mean_vec), elected_sign)

    agree = signs == elected_sign.unsqueeze(0)
    magnitudes = stacked.abs()
    masked_magnitudes = torch.where(agree, magnitudes, torch.zeros_like(magnitudes))
    max_magnitude, _ = masked_magnitudes.max(dim=0)
    merged = elected_sign * max_magnitude

    mean_abs = stacked.abs().mean()
    merged_abs = merged.abs().mean()
    if merged_abs > 1e-12:
        merged = merged * (mean_abs / merged_abs)
    return merged


class TunaAdapterViT(nn.Module):
    """Frozen base ViT + a growing list of per-experience adapters (only
    the most recent one trainable, like EASE) + one EMR-merged consensus
    adapter recomputed after every experience's training.
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

        self.adapter_list = nn.ModuleList()
        self.merged_adapter = self._new_adapter_set()
        _zero_adapter_set(self.merged_adapter)
        self.merged_adapter.requires_grad_(False)

    def _new_adapter_set(self) -> nn.ModuleList:
        return nn.ModuleList(
            [Adapter(self.feature_dim, self._bottleneck_dim, self._scale) for _ in range(self.depth)]
        )

    @property
    def cur_adapter(self) -> nn.ModuleList:
        if len(self.adapter_list) == 0:
            raise RuntimeError("grow() must be called before cur_adapter is accessed.")
        return cast(nn.ModuleList, self.adapter_list[-1])

    def grow(self) -> None:
        """Freeze the current (most recent) adapter, if any, then allocate
        a fresh trainable one for the upcoming experience."""
        if len(self.adapter_list) > 0:
            self.adapter_list[-1].requires_grad_(False)
            self.adapter_list[-1].eval()
        self.adapter_list.append(self._new_adapter_set())

    def _forward_with(self, x: torch.Tensor, adapter_set: nn.ModuleList) -> torch.Tensor:
        adapter: Dict[int, nn.Module] = dict(enumerate(adapter_set))
        return self.base(x, adapter=adapter)

    def forward_current(self, x: torch.Tensor) -> torch.Tensor:
        """Single pass through the currently-training adapter -- used to
        gradient-train it."""
        return self._forward_with(x, self.cur_adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single pass through the EMR-merged consensus adapter -- what
        the classifier uses at evaluation time (PILOT's per-sample
        entropy-based adapter routing is simplified out; the merged
        adapter is used directly)."""
        return self._forward_with(x, self.merged_adapter)

    def recompute_merge(self) -> None:
        """Recompute ``merged_adapter``'s values via EMR-merge over every
        stored adapter (including the just-finished current one). A
        value-only update -- safe to call from ``train_experience``
        (unlike structural growth, which must live in
        ``before_experience`` for resume-safety)."""
        with torch.no_grad():
            for block_idx in range(self.depth):
                for layer_name in _LAYER_NAMES:
                    for param_name in _PARAM_NAMES:
                        stacked = torch.stack(
                            [
                                getattr(
                                    getattr(
                                        cast(Adapter, cast(nn.ModuleList, adapter_set)[block_idx]),
                                        layer_name,
                                    ),
                                    param_name,
                                )
                                for adapter_set in self.adapter_list
                            ]
                        )
                        merged_value = _emr_merge(stacked)
                        target_layer = getattr(
                            cast(Adapter, self.merged_adapter[block_idx]), layer_name
                        )
                        getattr(target_layer, param_name).data.copy_(merged_value)


@register_backbone("vit_adapter_tuna")
def vit_adapter_tuna(
    base_model: str = "tiny_vit", bottleneck_dim: int = 8, scale: float = 0.1, **base_kwargs: Any
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return TunaAdapterViT(base, bottleneck_dim=bottleneck_dim, scale=scale)
