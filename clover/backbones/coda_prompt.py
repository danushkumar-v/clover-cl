"""CODA-Prompt's mechanism (SPEC §6.5): a **soft**, attention-weighted
combination of every prompt-pool component (no hard top-k, unlike L2P/
DualPrompt), injected as prefix key/value pairs via the same hook
DualPrompt uses. Each task unlocks a new slice of the pool, Gram-Schmidt
-orthogonalized against everything already unlocked rather than
reinitialized -- reducing interference between tasks' components.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model


class CodaPromptPool(nn.Module):
    """Pool of prompt components combined via soft, per-slot attention
    weights over however much of the pool is "unlocked" so far.

    Args:
        pool_size: Total number of prompt components.
        n_layers: How many attention blocks each component spans.
        length: Token length of each component's K/V.
        embed_dim: Base ViT's feature width.
        nb_experiences: Total experiences in the stream -- determines how
            many pool slots unlock per task (``pool_size // nb_experiences``).
    """

    def __init__(
        self, pool_size: int, n_layers: int, length: int, embed_dim: int, nb_experiences: int
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.slots_per_task = max(1, pool_size // max(1, nb_experiences))

        self.prompt = nn.Parameter(torch.zeros(pool_size, n_layers, 2, length, embed_dim))
        self.key = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.attn_vec = nn.Parameter(torch.zeros(pool_size, embed_dim))
        nn.init.uniform_(self.prompt, -1.0, 1.0)
        nn.init.uniform_(self.key, -1.0, 1.0)
        nn.init.uniform_(self.attn_vec, -1.0, 1.0)

        # A buffer (not a plain attribute) so it round-trips through
        # state_dict()/load_state_dict() automatically -- a resumed run
        # must know how much of the pool was already unlocked pre-crash.
        self.register_buffer("unlocked", torch.tensor(0, dtype=torch.long))

    def start_new_task(self, task_index: int) -> None:
        """Unlock this task's pool slice, Gram-Schmidt-orthogonalizing its
        prompt components against every already-unlocked component.
        No-op if this task's slice is already unlocked (e.g. re-entered via
        Trainer's resume replay -- ``load_state_dict`` overwrites the result
        anyway, same as incremental heads' ``expand_to`` during replay).
        """
        current = int(self.unlocked.item())  # type: ignore[operator]
        new_unlocked = min(self.pool_size, (task_index + 1) * self.slots_per_task)
        if new_unlocked <= current:
            return
        with torch.no_grad():
            shape = self.prompt.data[0].shape
            for i in range(current, new_unlocked):
                vec = self.prompt.data[i].flatten().clone()
                for j in range(i):
                    basis = self.prompt.data[j].flatten()
                    denom = basis @ basis
                    if denom > 1e-12:
                        vec = vec - ((vec @ basis) / denom) * basis
                self.prompt.data[i] = vec.reshape(shape)
        self.unlocked.fill_(new_unlocked)  # type: ignore[operator]

    def combine(self, query: torch.Tensor) -> torch.Tensor:
        """query: ``[B, embed_dim]`` -> ``[B, n_layers, 2, length, embed_dim]``."""
        n = int(self.unlocked.item()) or self.pool_size  # type: ignore[operator]
        keys = self.key[:n]
        attn_vecs = self.attn_vec[:n]
        prompts = self.prompt[:n]

        weighted_query = query.unsqueeze(1) * attn_vecs.unsqueeze(0)  # [B, n, embed_dim]
        similarity = (F.normalize(weighted_query, dim=-1) * F.normalize(keys, dim=-1).unsqueeze(0)).sum(
            dim=-1
        )  # [B, n]
        weights = F.softmax(similarity, dim=-1)
        return torch.einsum("bn,nlkte->blkte", weights, prompts)


class CodaPromptViT(nn.Module):
    """Wraps a base ViT with CODA-Prompt's soft-combined prefix prompt."""

    def __init__(
        self,
        base: nn.Module,
        pool_size: int = 10,
        length: int = 2,
        layers: Sequence[int] = (0, 1),
        nb_experiences: int = 1,
    ) -> None:
        super().__init__()
        depth = len(base.blocks)  # type: ignore[arg-type]
        if max(layers) >= depth:
            raise ValueError(
                f"layers references block {max(layers)} but the base ViT only has "
                f"{depth} blocks (0..{depth - 1})."
            )

        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()

        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        self.num_heads: int = base.blocks[0].attn.num_heads  # type: ignore[index,union-attr]
        self.head_dim: int = base.blocks[0].attn.head_dim  # type: ignore[index,union-attr]
        self.layers = list(layers)
        self.pool = CodaPromptPool(pool_size, len(self.layers), length, feature_dim, nb_experiences)

    def _to_prefix(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        x = x.reshape(b, length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.base.query_features(x)  # type: ignore[operator]
        combined = self.pool.combine(query)  # [B, n_layers, 2, length, embed_dim]

        prefix_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer_idx in enumerate(self.layers):
            k = combined[:, i, 0]
            v = combined[:, i, 1]
            prefix_kv[layer_idx] = (self._to_prefix(k), self._to_prefix(v))

        return self.base(x, prefix_kv=prefix_kv)


@register_backbone("vit_coda_prompt")
def vit_coda_prompt(
    base_model: str = "tiny_vit",
    pool_size: int = 10,
    length: int = 2,
    layers: Sequence[int] = (0, 1),
    nb_experiences: int = 1,
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return CodaPromptViT(
        base, pool_size=pool_size, length=length, layers=layers, nb_experiences=nb_experiences
    )
