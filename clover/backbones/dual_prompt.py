"""DualPrompt's mechanism (SPEC §6.5): a general prompt (always active, at
fixed shallow layers) plus a task-conditioned expert prompt pool (top-k
selected, like L2P's), both injected as **prefix key/value pairs** spliced
into specific attention blocks -- unlike L2P's whole-sequence token
prepend, this needs the base ViT's ``prefix_kv`` hook (``clover/backbones/
vit.py``).

``g_prompt_length``/``e_prompt_length``/``pool_size``/``top_k`` default to
the published values (``bench/configs/methods/dualprompt.yaml``:
``g_prompt_length=5``, ``length=5``, ``size=10``, ``top_k=1``) -- none of
these are depth-dependent. ``g_layers``/``e_layers`` are different: PILOT
places them at fixed block indices (``g_prompt_layer_idx=[0,1]``,
``e_prompt_layer_idx=[2,3,4]``) that only make sense *relative to* a
12-block ViT-B/16. ``default_layer_split`` re-expresses that placement as a
depth ratio (first 2/12 blocks for g, next 3/12 for e) so it recovers the
published indices exactly at depth 12 while still producing a valid,
disjoint split at TinyViT's depth 2 (see its docstring).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model

#: PILOT's published DualPrompt (``dualprompt.yaml``): ``g_prompt_layer_idx:
#: [0, 1]``, ``e_prompt_layer_idx: [2, 3, 4]`` -- 2 g-blocks and 3 e-blocks
#: out of a 12-block ViT-B/16. Expressed as ratios of depth, not literals, so
#: :func:`default_layer_split` can recover them exactly at depth 12 while
#: still producing a sane split at any other depth (notably TinyViT's 2).
_PUBLISHED_DEPTH = 12
_PUBLISHED_G_COUNT = 2
_PUBLISHED_E_COUNT = 3


def default_layer_split(depth: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Derive default ``(g_layers, e_layers)`` from the base's depth.

    Applies PILOT's published 2:3 (g:e) block-count ratio at a 12-block
    ViT-B/16 to *any* depth, each count floored at 1 block so a shallow base
    still gets a valid, disjoint, in-range split instead of one side
    collapsing to zero blocks. At depth 2 (TinyViT) this reduces to
    ``((0,), (1,))`` -- the literal this project's safety gate was already
    tuned against -- and at depth 12 it recovers PILOT's own
    ``([0, 1], [2, 3, 4])`` exactly.

    Args:
        depth: The base model's number of transformer blocks
            (``len(base.blocks)``).

    Returns:
        ``(g_layers, e_layers)``: two disjoint, consecutive ranges of block
        indices, g first, then e.

    Raises:
        ValueError: If *depth* is too shallow to fit at least one block of
            each prompt type (i.e. ``depth < 2``).
    """
    if depth < 2:
        raise ValueError(
            f"DualPrompt needs at least 2 transformer blocks (one for g_layers, one for "
            f"e_layers) to derive a default split; the base model only has {depth}. Pass "
            f"g_layers/e_layers explicitly if this base is intentional."
        )
    n_g = max(1, round(depth * _PUBLISHED_G_COUNT / _PUBLISHED_DEPTH))
    n_g = min(n_g, depth - 1)  # leave room for >=1 e block
    n_e = max(1, round(depth * _PUBLISHED_E_COUNT / _PUBLISHED_DEPTH))
    n_e = min(n_e, depth - n_g)
    return tuple(range(n_g)), tuple(range(n_g, n_g + n_e))


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
        g_layers: Block indices the general prompt is spliced into. ``None``
            (the default) derives it from the base's own depth via
            :func:`default_layer_split` -- pass explicit indices to override.
        e_layers: Block indices the expert prompt is spliced into (must be
            disjoint from ``g_layers``). ``None`` behaves like ``g_layers``.

    Raises:
        ValueError: If *g_layers*/*e_layers* overlap, reference a block
            beyond the base's depth, or (when defaulted) the base is too
            shallow to fit one block of each type.
    """

    def __init__(
        self,
        base: nn.Module,
        g_length: int = 5,
        e_length: int = 5,
        pool_size: int = 10,
        top_k: int = 1,
        g_layers: Optional[Sequence[int]] = None,
        e_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        depth = len(base.blocks)  # type: ignore[arg-type]
        if g_layers is None or e_layers is None:
            default_g, default_e = default_layer_split(depth)
            g_layers = default_g if g_layers is None else g_layers
            e_layers = default_e if e_layers is None else e_layers
        if set(g_layers) & set(e_layers):
            raise ValueError(
                f"g_layers and e_layers must be disjoint, got g_layers={list(g_layers)} "
                f"e_layers={list(e_layers)}."
            )
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
        attn0 = base.blocks[0].attn  # type: ignore[index,union-attr]
        self.num_heads: int = attn0.num_heads
        self.head_dim: int = attn0.head_dim
        # The K/V prefix reshape (`_to_prefix`) splits the prompt's last axis
        # into `num_heads * head_dim` -- timm's `attn_dim`, which can differ
        # from `feature_dim` on exotic configs even though the two coincide
        # for every base tested here (vit_tiny/vit_base, TinyViT). Sizing to
        # `attn_dim` rather than `feature_dim` keeps that reshape valid
        # regardless.
        prompt_embed_dim: int = getattr(attn0, "attn_dim", self.num_heads * self.head_dim)
        self.g_layers = list(g_layers)
        self.e_layers = list(e_layers)
        self.top_k = top_k

        self.g_prompt = nn.Parameter(torch.zeros(len(self.g_layers), 2, g_length, prompt_embed_dim))
        nn.init.uniform_(self.g_prompt, -1.0, 1.0)

        self.e_pool = nn.Parameter(
            torch.zeros(pool_size, len(self.e_layers), 2, e_length, prompt_embed_dim)
        )
        self.e_key = nn.Parameter(torch.zeros(pool_size, feature_dim))
        nn.init.uniform_(self.e_pool, -1.0, 1.0)
        nn.init.uniform_(self.e_key, -1.0, 1.0)
        #: See ``PromptPoolViT.query_fn`` (``clover/backbones/prompt_pool.py``)
        #: for why this indirection exists -- overridden by
        #: ``clover/methods/prompt_common.py`` at method build time.
        self.query_fn: Callable[[torch.Tensor], torch.Tensor] = self.base.query_features  # type: ignore[assignment]

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
        query = self.query_fn(x)
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
    g_length: int = 5,
    e_length: int = 5,
    pool_size: int = 10,
    top_k: int = 1,
    g_layers: Optional[Sequence[int]] = None,
    e_layers: Optional[Sequence[int]] = None,
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
