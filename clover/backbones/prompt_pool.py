"""L2P's mechanism (SPEC §6.5): a learnable prompt pool selected by cosine
similarity against a frozen backbone's query features, prepended as extra
input tokens (single insertion point -- no attention-internals surgery,
unlike DualPrompt/CODA-Prompt).
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model


class PromptPool(nn.Module):
    """``[pool_size, prompt_length, embed_dim]`` prompts + ``[pool_size,
    embed_dim]`` keys, selected top-k by cosine similarity to a query."""

    def __init__(self, pool_size: int, prompt_length: int, embed_dim: int) -> None:
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt = nn.Parameter(torch.zeros(pool_size, prompt_length, embed_dim))
        self.key = nn.Parameter(torch.zeros(pool_size, embed_dim))
        nn.init.uniform_(self.prompt, -1.0, 1.0)
        nn.init.uniform_(self.key, -1.0, 1.0)

    def select(self, query: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(selected_prompts [B, top_k*prompt_length, embed_dim], similarity [B, top_k])``."""
        query_n = F.normalize(query, dim=-1)
        key_n = F.normalize(self.key, dim=-1)
        similarity = query_n @ key_n.t()  # [B, pool_size]
        topk_similarity, topk_idx = similarity.topk(top_k, dim=-1)  # [B, top_k]
        selected = self.prompt[topk_idx]  # [B, top_k, prompt_length, embed_dim]
        batch = query.shape[0]
        selected = selected.reshape(batch, top_k * self.prompt_length, -1)
        return selected, topk_similarity


class PromptPoolViT(nn.Module):
    """Wraps a base ViT with L2P's prompt pool. Base is frozen; the pool
    (prompt + key) is the only thing this wrapper trains, along with
    whatever head sits on top."""

    def __init__(self, base: nn.Module, pool_size: int = 10, prompt_length: int = 5, top_k: int = 4) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        # base is duck-typed (any backbone exposing feature_dim/query_features/
        # forward(x, prompt_tokens=...), not just TinyViT specifically).
        feature_dim: int = base.feature_dim  # type: ignore[assignment]
        self.feature_dim = feature_dim
        self.pool = PromptPool(pool_size, prompt_length, self.feature_dim)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.base.query_features(x)  # type: ignore[operator]
        selected_prompts, _ = self.pool.select(query, self.top_k)
        return self.base(x, prompt_tokens=selected_prompts)


@register_backbone("vit_prompt_pool")
def vit_prompt_pool(
    base_model: str = "tiny_vit",
    pool_size: int = 10,
    prompt_length: int = 5,
    top_k: int = 4,
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return PromptPoolViT(base, pool_size=pool_size, prompt_length=prompt_length, top_k=top_k)
