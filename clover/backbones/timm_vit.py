"""Real (pretrained) timm ViT exposing CLOVER's backbone hook surface.

``TinyViT`` (``clover/backbones/vit.py``) is a synthetic-sized stand-in that
implements the hooks every wrapper mechanism relies on:
``patch_tokens`` / ``query_features`` (L2P-style prompt queries),
``forward_tokens(tokens, prefix_kv=, adapter=)`` and
``forward(x, prompt_tokens=, prefix_kv=, adapter=)``.

A stock ``timm`` ``VisionTransformer`` has none of them: its ``Block.forward``
takes only ``x``, and its ``Attention.forward`` has no key/value prefix.  So
until now only methods with a plain ``forward(x) -> features`` contract
(SimpleCIL) could use a real pretrained checkpoint, and the eight
prompt/adapter methods were stuck on an untrained stand-in.

This module closes that gap by re-expressing timm's block/attention forward
*through timm's own submodules* (``norm1``/``attn.{qkv,q_norm,k_norm,norm,
proj,proj_drop}``/``ls1``/``drop_path1``/``norm2``/``mlp``/``ls2``/
``drop_path2``), inserting the two hook points at exactly the positions
``TinyViT`` uses:

* ``prefix_kv`` -- concatenated onto K/V *after* ``q_norm``/``k_norm``, with
  queries untouched, so the output sequence length is unchanged and only
  what each query may attend to widens (DualPrompt / CODA-Prompt).
* ``adapter``   -- a parallel branch reading the *post-attention residual
  stream* ``x`` (never ``norm2(x)``), added alongside the MLP output
  (APER-Adapter / EASE / RanPAC / MOS / TUNA).

Semantic fidelity, not "it runs", is the bar: with no hooks passed, this
must reproduce timm's own ``forward`` bit-for-bit (pinned by
``tests/test_timm_vit_hooks.py``), and with hooks passed it must splice at
the same two points ``TinyViT`` does -- so a method is structurally
identical on both backbones and only scale differs.

No timm weights are modified and no timm code is copied: the arithmetic is
the standard ViT formulation, driven entirely by timm's own parameter
modules, so pretrained checkpoints load and behave normally.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _attend(
    attn: Any,
    x: torch.Tensor,
    prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """timm ``Attention`` forward, re-expressed with an optional K/V prefix.

    Args:
        attn: A timm ``Attention`` module (supplies every weight used here).
        x: Pre-normalised tokens ``[B, N, C]``.
        prefix_kv: ``(prefix_k, prefix_v)``, each ``[B, heads, prefix_len,
            head_dim]``, prepended to K/V. Queries are left alone, so the
            output is still ``[B, N, C]``.

    Returns:
        Attention output ``[B, N, C]``.
    """
    b, n, _ = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # each [B, heads, N, head_dim]
    # Order matters: timm normalises q/k *before* attention, so a prefix
    # concatenated afterwards is exactly the "prefix tuning" formulation
    # DualPrompt/CODA-Prompt assume -- prefix K/V bypass the q/k norms the
    # same way TinyViT's (norm-free) attention leaves them untouched.
    q, k = attn.q_norm(q), attn.k_norm(k)

    if prefix_kv is not None:
        prefix_k, prefix_v = prefix_kv
        k = torch.cat([prefix_k, k], dim=2)
        v = torch.cat([prefix_v, v], dim=2)

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=attn.attn_drop.p if attn.training else 0.0,
        scale=attn.scale,
    )

    # `attn_dim` (heads * head_dim) rather than x's own C: timm allows a
    # head dimension that doesn't tile the embedding dim exactly.
    attn_dim = getattr(attn, "attn_dim", attn.num_heads * attn.head_dim)
    out = out.transpose(1, 2).reshape(b, n, attn_dim)
    out = attn.norm(out)  # `scale_norm` variants; nn.Identity on a plain ViT
    out = attn.proj(out)
    return attn.proj_drop(out)


def _block_forward(
    block: Any,
    x: torch.Tensor,
    prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
    adapter: Optional[nn.Module],
) -> torch.Tensor:
    """timm ``Block`` forward with CLOVER's two hooks.

    Hook positions are identical to ``vit.TransformerBlock.forward`` -- in
    particular the adapter reads ``x`` (the post-attention residual stream),
    never ``norm2(x)``, so a method behaves the same on a real backbone as
    it does on ``TinyViT`` in the safety gate.

    Args:
        block: A timm ``Block`` module.
        x: Input tokens ``[B, N, C]``.
        prefix_kv: Optional K/V prefix for this block's attention.
        adapter: Optional parallel bottleneck branch for this block's MLP.

    Returns:
        Output tokens ``[B, N, C]``.
    """
    x = x + block.drop_path1(block.ls1(_attend(block.attn, block.norm1(x), prefix_kv)))

    mlp_out = block.mlp(block.norm2(x))
    if adapter is not None:
        mlp_out = mlp_out + adapter(x)
    return x + block.drop_path2(block.ls2(mlp_out))


class TimmViTHooks(nn.Module):
    """Wrap a timm ``VisionTransformer`` in CLOVER's hook surface.

    Exposes the same API as :class:`~clover.backbones.vit.TinyViT`, so every
    wrapper mechanism (``vit_prompt_pool``, ``vit_dual_prompt``,
    ``vit_coda_prompt``, ``vit_adapter*``) accepts it as a ``base_model``
    unchanged.

    Pooling is the class token, matching ``TinyViT`` and timm's own default
    ``global_pool="token"``; ``base.fc_norm`` is still applied so that a
    hook-free forward equals ``base(x)`` exactly.

    Args:
        base: A constructed timm ``VisionTransformer`` (``num_classes=0``).

    Raises:
        TypeError: If *base* is not shaped like a timm ``VisionTransformer``.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        # Every attribute this wrapper's forward actually touches, checked
        # up front so a wrong `base_model` fails at build time with a name,
        # not mid-training with an AttributeError.
        for attr in ("patch_embed", "_pos_embed", "patch_drop", "norm_pre", "blocks", "norm", "fc_norm"):
            if not hasattr(base, attr):
                raise TypeError(
                    f"{type(base).__name__} is not a timm VisionTransformer-shaped model "
                    f"(missing {attr!r}); CLOVER's prompt/adapter wrappers need that structure. "
                    f"Use a ViT-family timm name (e.g. vit_base_patch16_224)."
                )
        # Annotated `Any`, not `nn.Module`: every timm-specific submodule this
        # wrapper drives (`patch_drop`, `attn.q_norm`, `ls1`, ...) is reached
        # through `nn.Module.__getattr__`, which is typed `Tensor | Module`.
        # Assignment still registers `base` as a submodule at runtime, so its
        # parameters/state_dict are unaffected.
        timm_base: Any = base
        self.base: Any = timm_base
        self.feature_dim: int = int(timm_base.num_features)
        self.depth: int = len(timm_base.blocks)
        #: How many non-patch tokens ``_pos_embed`` prepends (cls, plus any
        #: register tokens). The class token is always first, so the pooled
        #: feature is index 0 regardless -- kept because prompt mechanisms
        #: that need to skip the prefix read it.
        self.num_prefix_tokens: int = int(getattr(base, "num_prefix_tokens", 1))

    @property
    def blocks(self) -> nn.ModuleList:
        """timm's own block list, surfaced at the top level because the
        wrapper mechanisms read it directly -- for depth (one adapter per
        block) and attention geometry (``blocks[0].attn.num_heads`` /
        ``.head_dim`` for prefix-KV shapes). A property, not a second
        registration: the blocks stay owned by ``self.base``, so their
        parameters are counted exactly once."""
        return self.base.blocks

    # -- query path (L2P-style prompt selection) --------------------------

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Images -> patch tokens; no cls token, position embedding, or prompts.

        Args:
            x: Images ``[B, C, H, W]`` (or ``[B, H, W]`` for single-channel
               data, which ``TinyViT`` also accepts -- the synthetic dataset
               stores plain ``(H, W)`` arrays).

        Returns:
            Patch tokens ``[B, num_patches, embed_dim]``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.base.patch_embed(x)

    def query_features(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen, mean-pooled patch features -- the query a prompt pool
        compares against its keys.

        Identical in definition to ``TinyViT.query_features``, which is the
        point: the mechanism must not change with the backbone. Note it is
        a *weaker* query than L2P/DualPrompt's published one (a full frozen
        forward's class token) -- on a random TinyViT that distinction is
        immaterial, but on a pretrained ViT this reads only the patch-embed
        convolution, leaving 12 blocks of pretrained signal out of prompt
        selection. Changing it is a per-method fidelity decision (P11-B1),
        not a backbone-layer one.

        Args:
            x: Images.

        Returns:
            Query features ``[B, embed_dim]``, detached from the graph.
        """
        with torch.no_grad():
            return self.patch_tokens(x).mean(dim=1)

    # -- token path -------------------------------------------------------

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        adapter: Optional[Dict[int, nn.Module]] = None,
    ) -> torch.Tensor:
        """Run the block stack over an already-embedded token sequence.

        Args:
            tokens: Token sequence ``[B, N, embed_dim]``.
            prefix_kv: ``{block_index: (prefix_k, prefix_v)}``.
            adapter: ``{block_index: Adapter}``.

        Returns:
            Normalised tokens ``[B, N, embed_dim]``.
        """
        for i, block in enumerate(self.base.blocks):
            tokens = _block_forward(
                block,
                tokens,
                None if prefix_kv is None else prefix_kv.get(i),
                None if adapter is None else adapter.get(i),
            )
        return self.base.norm(tokens)

    def forward(
        self,
        x: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor] = None,
        prefix_kv: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        adapter: Optional[Dict[int, nn.Module]] = None,
    ) -> torch.Tensor:
        """Images -> pooled (class-token) feature, with optional hooks.

        Args:
            x: Images.
            prompt_tokens: Extra tokens ``[B, P, embed_dim]`` prepended
                before the cls+patch sequence (L2P-style whole-sequence
                injection). The pooled feature follows the class token, so
                it shifts to index ``P``.
            prefix_kv: ``{block_index: (prefix_k, prefix_v)}`` spliced into
                specific blocks' attention K/V (DualPrompt/CODA-style).
            adapter: ``{block_index: Adapter}`` parallel bottleneck branches
                around each block's MLP (adapter family).

        Returns:
            Pooled features ``[B, feature_dim]``.
        """
        # `base._pos_embed` is timm's own routine for prepending the class
        # token and adding position embeddings (it also handles variants
        # such as `no_embed_class` and register tokens), so a pretrained
        # checkpoint's geometry is preserved exactly rather than re-derived.
        seq = self.base._pos_embed(self.patch_tokens(x))
        seq = self.base.patch_drop(seq)
        seq = self.base.norm_pre(seq)

        cls_index = 0
        if prompt_tokens is not None:
            seq = torch.cat([prompt_tokens, seq], dim=1)
            cls_index = prompt_tokens.shape[1]

        seq = self.forward_tokens(seq, prefix_kv=prefix_kv, adapter=adapter)
        return self.base.fc_norm(seq[:, cls_index])


def wrap_timm_vit(base: nn.Module) -> nn.Module:
    """Give a timm ViT CLOVER's hook surface, leaving anything else alone.

    Args:
        base: Any resolved base model.

    Returns:
        *base* wrapped in :class:`TimmViTHooks` when it is shaped like a
        timm ``VisionTransformer``; otherwise *base* unchanged -- models
        that already expose the hooks (``TinyViT``, a wrapper mechanism, a
        user class) or that have no ViT internals to splice into
        (``TinyMLP``, a CNN) pass straight through and stay usable by
        plain-``forward`` methods such as SimpleCIL.
    """
    if hasattr(base, "forward_tokens"):
        return base
    if all(hasattr(base, attr) for attr in ("blocks", "patch_embed", "_pos_embed", "norm")):
        return TimmViTHooks(base)
    return base
