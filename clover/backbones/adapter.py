"""Adapter-family shared building block (SPEC §6.5): an AdaptFormer-style
bottleneck adapter, spliced as a parallel branch around each transformer
block's MLP via ``TinyViT``'s ``adapter`` hook (``clover/backbones/vit.py``).

Confirmed via read-only research on LAMDA-PILOT's ``vit_adapter.py``/
``vit_ease.py``/``vit_mos.py``/``vit_tuna.py``: all five adapter-family
methods (APER-Adapter, EASE, RanPAC, MOS, TUNA) use this exact module
shape (down-proj -> ReLU -> up-proj, scaled); they differ only in how many
copies are kept live across experiences and how those copies are combined.
This module provides the single-adapter-per-block wrapper used directly by
APER-Adapter/RanPAC; EASE/MOS/TUNA each need their own per-task-list/merge
wrapper (queued) built on the same ``Adapter`` class + hook.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model

#: Every adapter-family method's ``ffn_num``/bottleneck width was
#: previously a bare literal (8) sized for TinyViT's 16-dim feature. PILOT's
#: published widths are fixed constants tuned for a 768-dim ViT-B/16
#: checkpoint (see ``bench/configs/methods/*.yaml``): APER-Adapter/RanPAC/
#: EASE use ``ffn_num=64`` (a 1:12 ratio to feature_dim), MOS/TUNA use 16
#: (1:48). ``default_bottleneck_dim`` recovers the published value exactly
#: at ViT-B/16 scale by applying that same ratio to *any* base's
#: ``feature_dim``, floored at 8 -- the literal this project's TinyViT
#: safety gate was already tuned against -- so a synthetic backbone's
#: bottleneck never shrinks below what's known to reliably clear
#: above-chance accuracy in the gate's epoch budget.
APER_EASE_RANPAC_BOTTLENECK_RATIO = 12  # ffn_num=64 at feature_dim=768
MOS_TUNA_BOTTLENECK_RATIO = 48  # ffn_num=16 at feature_dim=768
_MIN_BOTTLENECK_DIM = 8


def default_bottleneck_dim(feature_dim: int, published_ratio: int) -> int:
    """Scale a published adapter bottleneck width to *feature_dim*.

    Args:
        feature_dim: The base model's own feature width (``base.feature_dim``).
        published_ratio: ``feature_dim / bottleneck_dim`` at the PILOT
            reference scale (ViT-B/16, 768-dim) -- see the module docstring
            for which methods use which ratio and why.

    Returns:
        ``round(feature_dim / published_ratio)``, floored at
        :data:`_MIN_BOTTLENECK_DIM` so a tiny synthetic backbone (e.g.
        TinyViT's ``embed_dim=16``) doesn't collapse to a degenerate width.
    """
    return max(_MIN_BOTTLENECK_DIM, round(feature_dim / published_ratio))


class Adapter(nn.Module):
    """Down-proj -> ReLU -> up-proj bottleneck, scaled.

    Zero-initialized up-proj (kaiming-uniform down-proj, per PILOT's own
    init) so a freshly attached adapter starts as a no-op and doesn't
    perturb the frozen base's features until it's actually trained.
    """

    def __init__(self, embed_dim: int, bottleneck_dim: int = 8, scale: float = 0.1) -> None:
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = scale
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.act(self.down_proj(x))) * self.scale


class AdapterViT(nn.Module):
    """Wraps a frozen base ViT with ONE trainable adapter per block.

    APER-Adapter/RanPAC's shape: a single one-shot-tuned adapter that is
    never grown into a per-task list (unlike EASE/MOS/TUNA, which each get
    their own wrapper backbone on top of the same ``Adapter`` primitive).
    ``base`` stays reachable as ``self.base`` so a caller can also take a
    plain (adapter-free) pass through the *same* frozen weights -- used by
    APER-Adapter's dual-branch concatenation.
    """

    def __init__(
        self, base: nn.Module, bottleneck_dim: Optional[int] = None, scale: float = 0.1
    ) -> None:
        super().__init__()
        depth = len(base.blocks)  # type: ignore[arg-type]
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()

        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        if bottleneck_dim is None:
            bottleneck_dim = default_bottleneck_dim(feature_dim, APER_EASE_RANPAC_BOTTLENECK_RATIO)
        self.bottleneck_dim = bottleneck_dim
        self.adapters = nn.ModuleList(
            [Adapter(feature_dim, bottleneck_dim, scale) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter: Dict[int, nn.Module] = dict(enumerate(self.adapters))
        return self.base(x, adapter=adapter)


@register_backbone("vit_adapter")
def vit_adapter(
    base_model: str = "tiny_vit",
    bottleneck_dim: Optional[int] = None,
    scale: float = 0.1,
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return AdapterViT(base, bottleneck_dim=bottleneck_dim, scale=scale)
