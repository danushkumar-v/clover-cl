"""CODA-Prompt's mechanism (SPEC §6.5): a **soft**, attention-weighted
combination of every prompt-pool component (no hard top-k, unlike L2P/
DualPrompt), injected as prefix key/value pairs via the same hook
DualPrompt uses. Each task unlocks a new slice of the pool, Gram-Schmidt
-orthogonalized against everything already unlocked rather than
reinitialized -- reducing interference between tasks' components.

``pool_size``/``length`` default to PILOT's published ``prompt_param``
(``bench/configs/methods/coda_prompt.yaml``: ``[100, 8.0, 0.0]`` ->
``e_pool_size=100``, ``e_p_length=8``; the third value, ``ortho_mu``, is an
orthogonality *loss* penalty that PILOT's own config leaves at 0 -- i.e.
disabled -- so there is nothing to carry over structurally). ``layers``
defaults from the base's depth via :func:`default_layers`: PILOT hardcodes
``e_layers=[0,1,2,3,4]`` in ``CodaPrompt._init_smart`` regardless of
``prompt_param`` -- 5 of a 12-block ViT-B/16, a ratio, not a literal, once a
different-depth base is involved.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones import register_backbone
from clover.backbones.loader import resolve_base_model

#: PILOT hardcodes ``e_layers = [0, 1, 2, 3, 4]`` (``backbone/prompt.py``'s
#: ``CodaPrompt._init_smart``) -- 5 of a 12-block ViT-B/16. Expressed as a
#: ratio so :func:`default_layers` recovers it exactly at depth 12.
_PUBLISHED_DEPTH = 12
_PUBLISHED_LAYER_COUNT = 5
#: The previous TinyViT-tuned literal default (``layers=(0, 1)``), kept as a
#: floor -- same shape of rule as the adapter family's
#: ``default_bottleneck_dim`` (``clover/backbones/adapter.py``): a bare
#: ratio-based floor of 1 block measurably regressed the revisit-safety
#: gate's above-chance accuracy assertion for CODA-Prompt at TinyViT's
#: depth 2 (caught by ``tests/test_method_registry_safety_gate.py``), so the
#: floor has to be the literal this project already validated, not the
#: mathematical minimum of "at least one block."
_MIN_LAYER_COUNT = 2


def default_layers(depth: int) -> Tuple[int, ...]:
    """Derive default prefix-KV block indices from the base's depth.

    Applies PILOT's published ratio (5 of 12 blocks) to *any* depth, floored
    at :data:`_MIN_LAYER_COUNT` and capped at *depth* itself. At depth 12
    this recovers PILOT's own ``(0, 1, 2, 3, 4)`` exactly; at TinyViT's
    depth 2 the floor keeps it at ``(0, 1)`` -- numerically unchanged from
    the literal default that existed before this ratio did, which is what
    the safety gate's above-chance accuracy assertion was tuned against.

    Args:
        depth: The base model's number of transformer blocks.

    Returns:
        A tuple of consecutive block indices starting at 0.
    """
    n = max(_MIN_LAYER_COUNT, round(depth * _PUBLISHED_LAYER_COUNT / _PUBLISHED_DEPTH))
    n = min(n, depth)
    return tuple(range(n))


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
    """Wraps a base ViT with CODA-Prompt's soft-combined prefix prompt.

    Args:
        base: The frozen base ViT.
        pool_size: Total number of prompt components.
        length: Token length of each component's K/V.
        layers: Block indices the prompt is spliced into. ``None`` (the
            default) derives it from the base's own depth via
            :func:`default_layers`.
        nb_experiences: Total experiences in the stream.

    Raises:
        ValueError: If *layers* references a block beyond the base's depth.
    """

    def __init__(
        self,
        base: nn.Module,
        pool_size: int = 100,
        length: int = 8,
        layers: Optional[Sequence[int]] = None,
        nb_experiences: int = 1,
    ) -> None:
        super().__init__()
        depth = len(base.blocks)  # type: ignore[arg-type]
        if layers is None:
            layers = default_layers(depth)
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
        attn0 = base.blocks[0].attn  # type: ignore[index,union-attr]
        self.num_heads: int = attn0.num_heads
        self.head_dim: int = attn0.head_dim
        # Unlike DualPromptViT, `CodaPromptPool`'s prompt tensor and its
        # key/attn_vec share one `embed_dim` (the key/attn_vec must stay
        # `feature_dim`-wide to compare against the query), so there's no
        # single dimension to substitute `attn_dim` for without splitting
        # that pool's API -- not done here since `attn_dim == feature_dim`
        # for every base actually in use (see DualPromptViT's `_to_prefix`
        # comment for the general caveat).
        self.layers = list(layers)
        self.pool = CodaPromptPool(pool_size, len(self.layers), length, feature_dim, nb_experiences)
        #: See ``PromptPoolViT.query_fn`` (``clover/backbones/prompt_pool.py``).
        self.query_fn: Callable[[torch.Tensor], torch.Tensor] = self.base.query_features  # type: ignore[assignment]

    def _to_prefix(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        x = x.reshape(b, length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_fn(x)
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
    pool_size: int = 100,
    length: int = 8,
    layers: Optional[Sequence[int]] = None,
    nb_experiences: int = 1,
    **base_kwargs: Any,
) -> nn.Module:
    base = resolve_base_model(base_model, **base_kwargs)
    return CodaPromptViT(
        base, pool_size=pool_size, length=length, layers=layers, nb_experiences=nb_experiences
    )
