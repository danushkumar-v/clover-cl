"""A real timm ViT must satisfy the same hook contract as ``TinyViT``, so
every prompt/adapter method can use a pretrained backbone (not just the
plain-forward ones like SimpleCIL).

Two properties get pinned individually, because "it runs" is not the
acceptance bar for this wrapper -- a method must be *structurally* the same
on TinyViT and on a real ViT, only scaled:

1. the adapter branch reads the post-attention residual stream ``x``, never
   ``norm2(x)`` (``vit.TransformerBlock.forward``);
2. ``prefix_kv`` is concatenated onto K/V *after* ``q_norm``/``k_norm``,
   with queries untouched.

``pretrained=False`` throughout -- these tests build the architecture only,
never download weights, so they stay offline and CPU-only.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from clover.backbones import get_backbone
from clover.backbones.adapter import Adapter
from clover.backbones.loader import resolve_base_model
from clover.backbones.timm_vit import TimmViTHooks, _attend, _block_forward
from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext

_TIMM_VIT = "vit_tiny_patch16_224"  # smallest real ViT; same structure as ViT-B/16
_WRAPPER_METHODS = ["l2p", "dualprompt", "coda_prompt", "aper_adapter", "ease", "mos", "tuna", "ranpac"]


@pytest.fixture(scope="module")
def base():
    model = resolve_base_model(_TIMM_VIT, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def qk_norm_block():
    """One timm block whose ``q_norm``/``k_norm`` are *not* identities, so a
    test can tell "normalise then concat the prefix" apart from "concat then
    normalise" -- on a plain ViT both norms are ``nn.Identity`` and the two
    orderings are indistinguishable."""
    import timm

    model = timm.create_model(_TIMM_VIT, pretrained=False, num_classes=0, qk_norm=True)
    model.eval()
    assert not isinstance(model.blocks[0].attn.q_norm, nn.Identity)
    return model.blocks[0]


# --- surface parity with TinyViT ---------------------------------------


def test_resolve_wraps_a_timm_vit_in_the_hook_surface(base):
    assert isinstance(base, TimmViTHooks)
    from clover.backbones.vit import TinyViT

    for hook in ("feature_dim", "depth", "blocks", "patch_tokens", "query_features"):
        assert hasattr(base, hook), hook
    for method_name in ("forward_tokens", "forward"):
        tiny = inspect.signature(getattr(TinyViT, method_name))
        real = inspect.signature(getattr(TimmViTHooks, method_name))
        assert list(tiny.parameters) == list(real.parameters), method_name


def test_blocks_property_exposes_depth_and_attention_geometry(base):
    # Wrapper mechanisms size themselves off exactly these three reads.
    assert len(base.blocks) == base.depth == 12
    assert base.blocks[0].attn.num_heads * base.blocks[0].attn.head_dim == base.feature_dim
    # A property, not a second registration: params must be counted once.
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in base.base.parameters())


def test_plain_forward_matches_timm_feature_dim(base):
    out = base(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, base.feature_dim)
    assert torch.isfinite(out).all()


def test_hook_free_forward_reproduces_timms_own_forward_exactly():
    """The fidelity floor: with no hooks passed, this wrapper must be a
    re-expression of timm's forward, not an approximation of it -- otherwise
    a pretrained checkpoint's features silently change meaning."""
    import timm

    raw = timm.create_model(_TIMM_VIT, pretrained=False, num_classes=0).eval()
    wrapped = TimmViTHooks(raw)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(wrapped(x), raw(x), atol=1e-5)


# --- hook semantics: the two properties that must not drift -------------


def test_adapter_reads_the_post_attention_residual_stream_not_norm2(base):
    """``vit.TransformerBlock.forward`` feeds the adapter ``x`` -- the
    residual stream *after* the attention block -- and adds its output
    alongside the MLP's. Feeding it ``norm2(x)`` instead would still run and
    still train, but it would no longer be the same mechanism the safety
    gate validates on TinyViT."""
    block = base.blocks[0]
    tokens = torch.randn(2, 7, base.feature_dim)

    seen = {}

    class _Spy(nn.Module):
        def forward(self, x):
            seen["input"] = x.detach().clone()
            return torch.zeros_like(x)

    with torch.no_grad():
        _block_forward(block, tokens, None, _Spy())
        post_attention = tokens + block.drop_path1(block.ls1(_attend(block.attn, block.norm1(tokens), None)))

    assert torch.allclose(seen["input"], post_attention, atol=1e-6)
    assert not torch.allclose(seen["input"], block.norm2(post_attention), atol=1e-4)


def test_prefix_kv_is_concatenated_after_the_q_and_k_norms(qk_norm_block):
    """Prefix tuning prepends *raw* key/value vectors: they are not passed
    through ``q_norm``/``k_norm``, and queries are not touched at all. On a
    plain ViT both norms are identities, so this uses a ``qk_norm=True``
    model where the two orderings actually differ."""
    attn = qk_norm_block.attn
    b, n, c = 2, 6, attn.qkv.in_features
    x = torch.randn(b, n, c)
    prefix_k = torch.randn(b, attn.num_heads, 3, attn.head_dim)
    prefix_v = torch.randn(b, attn.num_heads, 3, attn.head_dim)

    def _reference(norm_after_concat: bool) -> torch.Tensor:
        qkv = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if norm_after_concat:
            q = attn.q_norm(q)
            k = attn.k_norm(torch.cat([prefix_k, k], dim=2))
        else:
            q, k = attn.q_norm(q), attn.k_norm(k)
            k = torch.cat([prefix_k, k], dim=2)
        v = torch.cat([prefix_v, v], dim=2)
        weights = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, n, attn.num_heads * attn.head_dim)
        return attn.proj_drop(attn.proj(attn.norm(out)))

    with torch.no_grad():
        actual = _attend(attn, x, (prefix_k, prefix_v))
        assert torch.allclose(actual, _reference(norm_after_concat=False), atol=1e-5)
        assert not torch.allclose(actual, _reference(norm_after_concat=True), atol=1e-4)


def test_prefix_kv_leaves_the_output_sequence_length_unchanged(base):
    """Queries are untouched, so a prefix only widens what each query may
    attend to -- the token count out must equal the token count in, or every
    downstream ``seq[:, cls_index]`` read silently shifts."""
    tokens = torch.randn(2, 9, base.feature_dim)
    heads, head_dim = base.blocks[0].attn.num_heads, base.blocks[0].attn.head_dim
    prefix = (torch.randn(2, heads, 5, head_dim), torch.randn(2, heads, 5, head_dim))
    with torch.no_grad():
        out = base.forward_tokens(tokens, prefix_kv={0: prefix, 3: prefix})
    assert out.shape == tokens.shape


def test_prefix_kv_hook_changes_the_output_without_changing_its_shape(base):
    x = torch.randn(2, 3, 224, 224)
    heads = base.blocks[0].attn.num_heads
    head_dim = base.blocks[0].attn.head_dim
    prefix = (torch.randn(2, heads, 5, head_dim), torch.randn(2, heads, 5, head_dim))
    with torch.no_grad():
        plain = base(x)
        prefixed = base(x, prefix_kv={0: prefix})
    assert prefixed.shape == plain.shape
    assert not torch.allclose(prefixed, plain)


def test_adapter_hook_is_a_parallel_branch_that_is_inert_when_zeroed(base):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        plain = base(x)
    # Adapter up-projection is zero-initialised, so an untrained adapter must
    # be a no-op -- the property every adapter method's first step relies on.
    adapter = Adapter(base.feature_dim, 8, 1.0)
    with torch.no_grad():
        with_adapter = base(x, adapter={0: adapter})
    assert torch.allclose(with_adapter, plain, atol=1e-5)

    with torch.no_grad():
        adapter.up_proj.weight.normal_(std=0.5)
        perturbed = base(x, adapter={0: adapter})
    assert not torch.allclose(perturbed, plain)


def test_prompt_tokens_are_prepended_and_the_cls_feature_still_returned(base):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = base(x, prompt_tokens=torch.randn(2, 4, base.feature_dim))
    assert out.shape == (2, base.feature_dim)
    assert torch.isfinite(out).all()


def test_patch_tokens_carry_no_cls_token_or_position_embedding(base):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = base.patch_tokens(x)
        query = base.query_features(x)
    assert tokens.shape == (2, base.base.patch_embed.num_patches, base.feature_dim)
    assert query.shape == (2, base.feature_dim)
    assert not query.requires_grad


def test_a_non_vit_model_passes_through_unwrapped():
    """``wrap_timm_vit`` must not claim models it cannot splice into --
    SimpleCIL-style plain-forward methods keep working with anything."""
    model = resolve_base_model("tiny_mlp", input_size=8, in_chans=1)
    assert not isinstance(model, TimmViTHooks)


# --- the actual gap this closes: real methods on a real base ------------


@pytest.mark.parametrize("method_name", _WRAPPER_METHODS)
def test_wrapper_methods_build_and_forward_on_a_real_timm_base(method_name, make_experience):
    """Before this module, only plain-forward methods (SimpleCIL) could take
    a real pretrained backbone.

    Note the config surface -- ``base_model`` swaps the base *underneath*
    the method's mechanism wrapper (prompt pool / prefix / adapter), which
    stays in place. Setting ``backbone`` instead would replace the whole
    mechanism with a bare ViT and silently disable the method.

    EASE and MOS allocate their adapters in ``before_experience``
    (``grow()``/``snapshot()``), so a forward pass is only defined once the
    method has been driven through its real lifecycle -- as the Trainer
    does, and as this test does.
    """
    method = get_method(method_name)()
    info = StreamInfo(
        dataset="imagenet_r", nb_experiences=3, total_classes=30, input_size=224, channels=3
    )
    method.build(info, {"base_model": _TIMM_VIT, "pretrained": False})
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
        epochs=1,
    )
    method.before_experience(make_experience(0, [0, 1], head_size=2), ctx)

    # The classifier, not the raw backbone: feature shape varies by family
    # (MOS stacks over adapters, EASE over blocks) but `images -> logits`
    # is the one contract every method owes the shared evaluator.
    with torch.no_grad():
        logits = method.classifier()(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_simplecil_takes_a_real_timm_vit_directly_as_backbone():
    """SimpleCIL has no mechanism wrapper, so its base IS its backbone."""
    method = get_method("simplecil")()
    info = StreamInfo(
        dataset="imagenet_r", nb_experiences=3, total_classes=30, input_size=224, channels=3
    )
    method.build(info, {"backbone": _TIMM_VIT, "pretrained": False})
    with torch.no_grad():
        feats = method.backbone(torch.randn(2, 3, 224, 224))
    assert feats.shape == (2, method.backbone.feature_dim)


def test_backbone_key_on_a_wrapper_method_silently_drops_the_mechanism():
    """Documented in docs/real_backbones.md and pinned here so the footgun
    stays visible: ``backbone:`` replaces the whole mechanism wrapper."""
    method = get_method("l2p")()
    info = StreamInfo(
        dataset="imagenet_r", nb_experiences=3, total_classes=30, input_size=224, channels=3
    )
    method.build(info, {"backbone": _TIMM_VIT, "pretrained": False})
    assert isinstance(method.backbone, TimmViTHooks)
    assert not hasattr(method.backbone, "pool")  # the prompt pool is gone


def test_wrapper_backbone_factories_are_discoverable_by_their_base_model_arg():
    """The safety gate picks ``backbone:`` vs ``base_model:`` per method from
    this signature check rather than a hardcoded list, so a new method is
    covered automatically."""
    assert "base_model" in inspect.signature(get_backbone("vit_prompt_pool")).parameters
    assert "base_model" not in inspect.signature(get_backbone("tiny_mlp")).parameters
