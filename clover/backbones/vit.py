"""Clean-room minimal ViT (SPEC §6.4), sized for the synthetic dataset.

Each block's attention accepts an optional ``prefix_kv`` -- prepended to
K/V before the softmax. L2P (P5) doesn't use it; DualPrompt and CODA-Prompt
inject their prefix prompts through exactly this hook.

Each block also accepts an optional ``adapter`` module (P6, SPEC §6.5's
adapter family: APER-Adapter/EASE/RanPAC/MOS/TUNA) -- a parallel branch
reading the pre-MLP residual stream, added alongside the MLP's own output.
Distinct hook from ``prefix_kv``: adapters never touch attention K/V, only
the residual stream around the MLP (confirmed via read-only research on
LAMDA-PILOT's ``vit_adapter.py``/``vit_ease.py``/``vit_mos.py``/
``vit_tuna.py`` -- all splice identically here, never wrapping ``fc1``/
``fc2`` directly).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from clover.backbones import register_backbone


class Attention(nn.Module):
    """Multi-head self-attention with an optional key/value prefix."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, N, head_dim]

        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv  # each [B, heads, prefix_len, head_dim]
            k = torch.cat([prefix_k, k], dim=2)
            v = torch.cat([prefix_v, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        adapter: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), prefix_kv=prefix_kv)
        mlp_out = self.mlp(self.norm2(x))
        if adapter is not None:
            # Parallel branch off the pre-MLP residual stream `x`, added
            # alongside the MLP's own output -- never wrapping fc1/fc2.
            mlp_out = mlp_out + adapter(x)
        x = x + mlp_out
        return x


@register_backbone("tiny_vit")
class TinyViT(nn.Module):
    """A from-scratch minimal ViT: patch embed, cls token, learned position
    embedding, a small stack of transformer blocks. Sized by default for
    the 8x8 synthetic dataset (SPEC §12.2), not for real images."""

    def __init__(
        self,
        input_size: int = 8,
        in_chans: int = 1,
        patch_size: int = 4,
        embed_dim: int = 16,
        depth: int = 2,
        num_heads: int = 2,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by patch_size ({patch_size}).")
        self.feature_dim = embed_dim
        num_patches = (input_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """images -> patch tokens, no cls/position/prompt (for L2P-style queries)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add a channel dim for grayscale synthetic images
        tokens = self.patch_embed(x)
        return tokens.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

    def query_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen, mean-pooled patch features -- the query a prompt-pool
        selection mechanism compares against its keys."""
        with torch.no_grad():
            return self.patch_tokens(x).mean(dim=1)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        adapter: Optional[Dict[int, nn.Module]] = None,
    ) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            block_prefix = None if prefix_kv is None else prefix_kv.get(i)
            block_adapter = None if adapter is None else adapter.get(i)
            tokens = block(tokens, prefix_kv=block_prefix, adapter=block_adapter)
        return self.norm(tokens)

    def forward(
        self,
        x: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor] = None,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        adapter: Optional[Dict[int, nn.Module]] = None,
    ) -> torch.Tensor:
        """images -> pooled (cls-token) feature.

        Args:
            prompt_tokens: extra tokens prepended before the cls+patch
                sequence (L2P-style whole-sequence injection).
            prefix_kv: ``{block_index: (prefix_k, prefix_v)}`` spliced into
                specific blocks' attention K/V (DualPrompt/CODA-style;
                unused by L2P).
            adapter: ``{block_index: Adapter}`` parallel bottleneck branches
                around each block's MLP (APER-Adapter/EASE/RanPAC/MOS/TUNA;
                unused by the prompt family).
        """
        b = x.shape[0]
        patches = self.patch_tokens(x)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, patches], dim=1) + self.pos_embed

        if prompt_tokens is not None:
            seq = torch.cat([prompt_tokens, seq], dim=1)
            cls_index = prompt_tokens.shape[1]
        else:
            cls_index = 0

        seq = self.forward_tokens(seq, prefix_kv=prefix_kv, adapter=adapter)
        return seq[:, cls_index]
