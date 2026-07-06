"""MOS's mechanism (SPEC §6.5): a single continuously-trained adapter,
regularized every optimizer step toward the running mean of all earlier
experiences' (frozen) snapshots. Confirmed via read-only research on
``models/mos.py``/``backbone/vit_mos.py``: unlike EASE (a fresh adapter
per experience) or APER-Adapter/RanPAC (one adapter, trained once), MOS
keeps ONE ``cur_adapter`` training continuously across the whole run,
pulled toward its own history via an EMA blend (``momentum``) after every
optimizer step -- this *is* the "Mixture-of-Subspace" name.

``snapshot()`` (freeze a copy of ``cur_adapter``'s current state into
``adapter_list``, fold it into the running sum) is called once per
experience (after the first) from ``MOS.before_experience`` -- the same
resume-safety reason EASE's ``grow()`` lives there: the Trainer's
resume-replay only re-invokes ``before_experience`` for already-completed
experiences, so any structural growth needed before ``load_state_dict``
succeeds must happen there, not in ``train_experience``/
``after_experience``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, cast

import torch
import torch.nn as nn

from clover.backbones import register_backbone
from clover.backbones.adapter import Adapter
from clover.backbones.loader import resolve_base_model


def _zero_adapter_set(adapter_set: nn.ModuleList) -> None:
    with torch.no_grad():
        for module in adapter_set:
            adapter = cast(Adapter, module)
            adapter.down_proj.weight.zero_()
            adapter.down_proj.bias.zero_()
            adapter.up_proj.weight.zero_()
            adapter.up_proj.bias.zero_()


class MOSAdapterViT(nn.Module):
    """Frozen base ViT + one continuously-trained adapter, EMA-regularized
    toward the mean of its own frozen history.

    ``adapter_list`` holds frozen snapshots of ``cur_adapter`` taken at the
    end of each earlier experience (see ``snapshot``); ``_adapter_sum`` is
    a same-shaped running sum of those snapshots' parameters (a plain
    accumulator, not itself trained -- ``requires_grad_(False)``), used to
    compute the EMA pull target in ``merge_step`` without re-summing
    ``adapter_list`` from scratch every step.
    """

    def __init__(
        self, base: nn.Module, bottleneck_dim: int = 8, scale: float = 0.1, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.depth = len(base.blocks)  # type: ignore[arg-type]
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()

        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        self._bottleneck_dim = bottleneck_dim
        self._scale = scale
        self.momentum = momentum

        self.cur_adapter = self._new_adapter_set()
        self.adapter_list = nn.ModuleList()
        self._adapter_sum = self._new_adapter_set()
        _zero_adapter_set(self._adapter_sum)
        self._adapter_sum.requires_grad_(False)

    def _new_adapter_set(self) -> nn.ModuleList:
        return nn.ModuleList(
            [Adapter(self.feature_dim, self._bottleneck_dim, self._scale) for _ in range(self.depth)]
        )

    def _forward_with(self, x: torch.Tensor, adapter_set: nn.ModuleList) -> torch.Tensor:
        adapter: Dict[int, nn.Module] = dict(enumerate(adapter_set))
        return self.base(x, adapter=adapter)

    def forward_current(self, x: torch.Tensor) -> torch.Tensor:
        """Single pass through ``cur_adapter`` -- used to gradient-train it."""
        return self._forward_with(x, self.cur_adapter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Every stored adapter's feature (frozen history + the current,
        live one), stacked as ``[num_adapters, batch, feature_dim]`` -- the
        shared head applies to every slice, and the classifier averages
        the resulting logits (a plain ensemble; PILOT's own entropy-based
        self-refinement search is simplified out here)."""
        adapters = list(self.adapter_list) + [self.cur_adapter]
        features = [self._forward_with(x, a) for a in adapters]  # type: ignore[arg-type]
        return torch.stack(features, dim=0)

    def snapshot(self) -> None:
        """Freeze a deep copy of ``cur_adapter``'s current state into
        ``adapter_list`` and fold it into the running sum."""
        frozen = copy.deepcopy(self.cur_adapter)
        frozen.requires_grad_(False)
        frozen.eval()
        self.adapter_list.append(frozen)
        with torch.no_grad():
            for summed_module, snap_module in zip(self._adapter_sum, frozen):
                summed, snap = cast(Adapter, summed_module), cast(Adapter, snap_module)
                summed.down_proj.weight.data.add_(snap.down_proj.weight)
                summed.down_proj.bias.data.add_(snap.down_proj.bias)
                summed.up_proj.weight.data.add_(snap.up_proj.weight)
                summed.up_proj.bias.data.add_(snap.up_proj.bias)

    def merge_step(self) -> None:
        """EMA-blend ``cur_adapter``'s parameters toward the running mean
        of all earlier snapshots -- call after every optimizer step during
        training. No-op before any snapshot exists (nothing to pull toward
        yet, matching PILOT's own first-experience behavior)."""
        n = len(self.adapter_list)
        if n == 0:
            return
        with torch.no_grad():
            for cur, summed in zip(self.cur_adapter, self._adapter_sum):
                for layer_name in ("down_proj", "up_proj"):
                    cur_layer = getattr(cur, layer_name)
                    sum_layer = getattr(summed, layer_name)
                    for param_name in ("weight", "bias"):
                        cur_param = getattr(cur_layer, param_name)
                        mean_val = getattr(sum_layer, param_name) / n
                        cur_param.data.mul_(1 - self.momentum).add_(mean_val, alpha=self.momentum)


@register_backbone("vit_adapter_mos")
def vit_adapter_mos(
    base_model: str = "tiny_vit",
    bottleneck_dim: int = 8,
    scale: float = 0.1,
    momentum: float = 0.1,
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return MOSAdapterViT(base, bottleneck_dim=bottleneck_dim, scale=scale, momentum=momentum)
