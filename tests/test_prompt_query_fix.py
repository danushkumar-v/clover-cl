"""The ``query_features`` fidelity fix (P11-B1, Task 2).

``TimmViTHooks.query_features``/``TinyViT.query_features`` mean-pool raw
patch-embedding output -- deliberately backbone-agnostic (the *backbone*
layer must not change behaviour based on which base is used, see
``clover/backbones/timm_vit.py``'s docstring). But published L2P/DualPrompt
(``bench/configs/methods/{l2p,dualprompt}.yaml``: ``get_original_backbone:
true``, ``embedding_key: "cls"``) compute the prompt-selection query from a
**frozen full forward pass of the pretrained backbone's class token** -- on
a pretrained ViT that's the difference between "prompt selection sees 12
blocks of pretrained signal" and "prompt selection sees only the patch-embed
convolution."

``clover/methods/prompt_common.py:published_query`` closes that gap at the
*method* layer (not ``timm_vit.py``, which CLOVER's adapter-family owner
would need to touch): ``PromptMethodBase.build()`` overrides each mechanism
wrapper's ``query_fn`` with a no-grad, hook-free forward through the frozen
base.

The test that matters here is the one proving this actually changes
behaviour: perturbing a *late* transformer block's weights must change the
query now (it flows through every block), which was false before this fix
(the old query only ever touched the patch-embed conv, before any block
runs).
"""

from __future__ import annotations

import torch

from clover.backbones.loader import resolve_base_model
from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.prompt_common import published_query

_REAL_BASE = "vit_tiny_patch16_224"


def _perturb(weight: torch.Tensor) -> None:
    """Structured, per-entry noise -- not a uniform ``add_(constant)``.

    A constant added to *every* entry of a linear layer's weight shifts
    every output feature by the *same* scalar for a given token (``sum_i
    (W[j,i]+c) h_i = original_j + c * sum_i(h_i)``, identical across ``j``).
    A per-token-uniform shift is exactly what a subsequent LayerNorm
    (``fc_norm``/``norm``, present at the end of every base tested here)
    cancels out by construction -- so a uniform perturbation is not a valid
    probe for "does the query depend on this block" and produced a false
    negative during development. Independent per-entry noise varies
    unevenly across output features and survives normalisation.
    """
    with torch.no_grad():
        weight.add_(torch.randn_like(weight))


def test_published_query_is_a_pooled_class_token_not_a_patch_embed_mean():
    """``published_query(base)(x)`` must equal ``base(x)`` (a full,
    hook-free forward's pooled class-token feature) -- not
    ``base.query_features(x)`` (patch-embed-only)."""
    base = resolve_base_model(_REAL_BASE, pretrained=False)
    base.eval()
    x = torch.randn(2, 3, 224, 224)

    query_fn = published_query(base)
    with torch.no_grad():
        expected = base(x)
        weak_query = base.query_features(x)

    assert torch.allclose(query_fn(x), expected)
    assert not torch.allclose(query_fn(x), weak_query)


def test_published_query_depends_on_a_late_blocks_weights_on_a_real_timm_base():
    """The property the old (patch-embed-only) query lacked: perturbing
    block 11 (the last of 12) must change the query. Pinned directly against
    ``published_query``, independent of any particular method."""
    base = resolve_base_model(_REAL_BASE, pretrained=False)
    base.eval()
    query_fn = published_query(base)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        before = query_fn(x)
        last_block = base.blocks[len(base.blocks) - 1]
        _perturb(last_block.mlp.fc2.weight)
        after = query_fn(x)

    assert not torch.allclose(before, after)


def test_published_query_depends_on_a_late_blocks_weights_on_tinyvit():
    """Same property, but on the synthetic backbone the safety gate runs
    on -- the fix must work on both bases, not just a real timm one."""
    from clover.backbones.vit import TinyViT

    base = TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=3, num_heads=2)
    base.eval()
    query_fn = published_query(base)
    x = torch.randn(2, 8, 8)

    with torch.no_grad():
        before = query_fn(x)
        _perturb(base.blocks[-1].mlp[-1].weight)
        after = query_fn(x)

    assert not torch.allclose(before, after)


def test_published_query_does_not_build_a_training_graph():
    base = resolve_base_model(_REAL_BASE, pretrained=False)
    base.eval()
    query_fn = published_query(base)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    out = query_fn(x)
    assert not out.requires_grad
    assert out.grad_fn is None


def test_published_query_does_not_change_base_train_eval_mode():
    base = resolve_base_model(_REAL_BASE, pretrained=False)
    base.eval()
    query_fn = published_query(base)
    query_fn(torch.randn(1, 3, 224, 224))
    assert not base.training  # still eval() afterwards, untouched


# --- wired through the method layer (build() overrides query_fn) --------


def test_l2p_backbone_query_fn_is_overridden_by_build_on_a_real_timm_base():
    """Before this fix, ``method.backbone.query_fn`` would still be the
    wrapper's own ``base.query_features`` (patch-embed-only) even after
    ``build()``. Perturbing a late block must now change what L2P's prompt
    pool selects against."""
    method = get_method("l2p")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=224, channels=3)
    method.build(info, {"base_model": _REAL_BASE, "pretrained": False})

    x = torch.randn(2, 3, 224, 224)
    backbone = method.backbone
    with torch.no_grad():
        before = backbone.query_fn(x)
        last_block = backbone.base.blocks[len(backbone.base.blocks) - 1]
        _perturb(last_block.mlp.fc2.weight)
        after = backbone.query_fn(x)

    assert not torch.allclose(before, after)


def test_dualprompt_backbone_query_fn_is_overridden_by_build_on_a_real_timm_base():
    method = get_method("dualprompt")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=224, channels=3)
    method.build(info, {"base_model": _REAL_BASE, "pretrained": False})

    x = torch.randn(2, 3, 224, 224)
    backbone = method.backbone
    with torch.no_grad():
        before = backbone.query_fn(x)
        last_block = backbone.base.blocks[len(backbone.base.blocks) - 1]
        _perturb(last_block.mlp.fc2.weight)
        after = backbone.query_fn(x)

    assert not torch.allclose(before, after)


def test_coda_prompt_backbone_query_fn_is_overridden_by_build_on_a_real_timm_base():
    method = get_method("coda_prompt")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=224, channels=3)
    method.build(info, {"base_model": _REAL_BASE, "pretrained": False})

    x = torch.randn(2, 3, 224, 224)
    backbone = method.backbone
    with torch.no_grad():
        before = backbone.query_fn(x)
        last_block = backbone.base.blocks[len(backbone.base.blocks) - 1]
        _perturb(last_block.mlp.fc2.weight)
        after = backbone.query_fn(x)

    assert not torch.allclose(before, after)


def test_backbone_key_misconfiguration_leaves_query_fn_untouched():
    """The degenerate ``backbone:`` case (mechanism silently dropped, see
    docs/real_backbones.md): ``build()`` must not crash just because the
    resulting object has neither ``.base`` nor ``.query_fn``."""
    method = get_method("l2p")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=224, channels=3)
    method.build(info, {"backbone": _REAL_BASE, "pretrained": False})  # wrong key, on purpose
    assert not hasattr(method.backbone, "pool")


def test_query_fix_survives_before_experience_device_move(make_experience):
    """The query closure captures the base module by reference, so moving
    the backbone (``before_experience``'s ``.to(ctx.device)``) must not make
    it stale -- this is the device-correctness property Task 4 asks to be
    checked for prompt selection specifically."""
    method = get_method("l2p")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8, channels=1)
    method.build(info, {"pool_size": 4, "top_k": 2})
    ctx = TrainContext(device=torch.device("cpu"))
    exp = make_experience(0, [0, 1], head_size=2)

    method.before_experience(exp, ctx)  # moves backbone (and its base) to ctx.device
    with torch.no_grad():
        out = method.backbone.query_fn(torch.randn(2, 8, 8))
    assert torch.isfinite(out).all()
