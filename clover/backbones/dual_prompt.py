"""DualPrompt's mechanism (SPEC §6.5): a general prompt (always active, at
fixed shallow layers) plus a task-conditioned expert prompt pool (top-k
selected, like L2P's), both injected as **prefix key/value pairs** spliced
into specific attention blocks -- unlike L2P's whole-sequence token
prepend, this needs the base ViT's ``prefix_kv`` hook (``clover/backbones/
vit.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model


class DualPromptViT(nn.Module):
    """Wraps a base ViT with DualPrompt's general + expert prefix prompts.

    Args:
        base: The frozen base ViT (must expose ``.blocks`` with
            ``.attn.num_heads``/``.attn.head_dim``, and accept
            ``forward(x, prefix_kv={layer_idx: (k, v)})`` -- ``TinyViT``'s
            interface).
        g_length: Token length of the general prompt.
        e_length: Token length of each expert prompt.
        pool_size: Number of expert prompt candidates.
        top_k: How many expert candidates to select and average per input.
        g_layers: Block indices the general prompt is spliced into.
        e_layers: Block indices the expert prompt is spliced into (must be
            disjoint from ``g_layers``).
    """

    def __init__(
        self,
        base: nn.Module,
        g_length: int = 2,
        e_length: int = 2,
        pool_size: int = 10,
        top_k: int = 1,
        g_layers: Sequence[int] = (0,),
        e_layers: Sequence[int] = (1,),
    ) -> None:
        super().__init__()
        if set(g_layers) & set(e_layers):
            raise ValueError(
                f"g_layers and e_layers must be disjoint, got g_layers={list(g_layers)} "
                f"e_layers={list(e_layers)}."
            )
        depth = len(base.blocks)  # type: ignore[arg-type]
        max_layer = max(list(g_layers) + list(e_layers))
        if max_layer >= depth:
            raise ValueError(
                f"g_layers/e_layers reference block {max_layer} but the base ViT only has "
                f"{depth} blocks (0..{depth - 1})."
            )

        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()

        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        self.num_heads: int = base.blocks[0].attn.num_heads  # type: ignore[index,union-attr]
        self.head_dim: int = base.blocks[0].attn.head_dim  # type: ignore[index,union-attr]
        self.g_layers = list(g_layers)
        self.e_layers = list(e_layers)
        self.top_k = top_k

        self.g_prompt = nn.Parameter(torch.zeros(len(self.g_layers), 2, g_length, feature_dim))
        nn.init.uniform_(self.g_prompt, -1.0, 1.0)

        self.e_pool = nn.Parameter(
            torch.zeros(pool_size, len(self.e_layers), 2, e_length, feature_dim)
        )
        self.e_key = nn.Parameter(torch.zeros(pool_size, feature_dim))
        nn.init.uniform_(self.e_pool, -1.0, 1.0)
        nn.init.uniform_(self.e_key, -1.0, 1.0)

    def _to_prefix(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, length, embed_dim] -> [B, num_heads, length, head_dim]``."""
        b, length, _ = x.shape
        x = x.reshape(b, length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _select_experts(self, query: torch.Tensor) -> torch.Tensor:
        """Return ``[B, n_e_layers, 2, e_length, embed_dim]``: the top-k
        expert candidates, averaged."""
        query_n = F.normalize(query, dim=-1)
        key_n = F.normalize(self.e_key, dim=-1)
        similarity = query_n @ key_n.t()  # [B, pool_size]
        _, topk_idx = similarity.topk(self.top_k, dim=-1)  # [B, top_k]
        selected = self.e_pool[topk_idx]  # [B, top_k, n_e_layers, 2, e_length, embed_dim]
        return selected.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        query = self.base.query_features(x)  # type: ignore[operator]
        selected_experts = self._select_experts(query)

        prefix_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer_idx in enumerate(self.g_layers):
            k = self.g_prompt[i, 0].unsqueeze(0).expand(b, -1, -1)
            v = self.g_prompt[i, 1].unsqueeze(0).expand(b, -1, -1)
            prefix_kv[layer_idx] = (self._to_prefix(k), self._to_prefix(v))

        for i, layer_idx in enumerate(self.e_layers):
            k = selected_experts[:, i, 0]
            v = selected_experts[:, i, 1]
            prefix_kv[layer_idx] = (self._to_prefix(k), self._to_prefix(v))

        return self.base(x, prefix_kv=prefix_kv)


@register_backbone("vit_dual_prompt")
def vit_dual_prompt(
    base_model: str = "tiny_vit",
    g_length: int = 2,
    e_length: int = 2,
    pool_size: int = 10,
    top_k: int = 1,
    g_layers: Sequence[int] = (0,),
    e_layers: Sequence[int] = (1,),
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return DualPromptViT(
        base,
        g_length=g_length,
        e_length=e_length,
        pool_size=pool_size,
        top_k=top_k,
        g_layers=g_layers,
        e_layers=e_layers,
    )
