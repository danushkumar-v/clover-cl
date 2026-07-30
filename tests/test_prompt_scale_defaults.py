"""Scale-derived prompt-family defaults (P11-B1, Task 1): published PILOT
hyperparameter *values* (``bench/configs/methods/{l2p,dualprompt,
coda_prompt}.yaml``, read-only reference, never imported/copied --
CLAUDE.md) recovered exactly at ViT-B/16 scale (``depth=12``), with
``TinyViT``-scale behaviour (``depth=2``) either numerically unchanged
(DualPrompt's layer split) or still valid/above-chance (CODA-Prompt's
layers, narrowed by the new ratio-based rule -- see
``clover/backbones/coda_prompt.py:default_layers``'s docstring) so the
existing 6-scenario safety gate stays green.

L2P's pool size/prompt length/top-k are *not* depth-dependent (L2P injects
prompt tokens once, at the input, not per-block) -- the same defaults are
correct at both scales, so there is no split-by-depth rule to pin for it,
just the published values themselves.
"""

from __future__ import annotations

import pytest

from clover.backbones import get_backbone
from clover.backbones.coda_prompt import CodaPromptViT, default_layers
from clover.backbones.dual_prompt import DualPromptViT, default_layer_split
from clover.backbones.loader import resolve_base_model
from clover.backbones.prompt_pool import PromptPoolViT
from clover.backbones.vit import TinyViT

# vit_base_patch16_224 architecture-only construction takes a few seconds;
# built once per module and reused (read-only structural checks) across
# every ViT-B/16-scale test below rather than once per test.
_VIT_B16_NAME = "vit_base_patch16_224"


@pytest.fixture(scope="module")
def real_vit_b16():
    return resolve_base_model(_VIT_B16_NAME, pretrained=False)


def _tiny_base(depth: int = 2) -> TinyViT:
    return TinyViT(input_size=8, patch_size=4, embed_dim=16, depth=depth, num_heads=2)


# --- L2P: pool_size / prompt_length / top_k -----------------------------


def test_l2p_pool_defaults_match_published_values():
    # bench/configs/methods/l2p.yaml: size=10, length=5, top_k=5.
    wrapped = PromptPoolViT(_tiny_base())
    assert wrapped.pool.prompt.shape[0] == 10  # pool_size
    assert wrapped.pool.prompt_length == 5
    assert wrapped.top_k == 5


def test_l2p_factory_default_matches_published_values():
    wrapped = get_backbone("vit_prompt_pool")(base_model="tiny_vit", input_size=8)
    assert wrapped.pool.prompt.shape[0] == 10
    assert wrapped.pool.prompt_length == 5
    assert wrapped.top_k == 5


def test_l2p_defaults_are_scale_independent(real_vit_b16):
    """Unlike DualPrompt/CODA-Prompt, none of L2P's defaults are a function
    of depth -- the same values are correct on a real ViT-B/16 base."""
    wrapped = PromptPoolViT(real_vit_b16)
    assert wrapped.pool.prompt.shape[0] == 10
    assert wrapped.pool.prompt_length == 5
    assert wrapped.top_k == 5


# --- DualPrompt: default_layer_split -- the rule itself ------------------


def test_default_layer_split_matches_published_indices_at_vitb16_scale():
    # dualprompt.yaml: g_prompt_layer_idx=[0, 1], e_prompt_layer_idx=[2, 3, 4].
    g_layers, e_layers = default_layer_split(12)
    assert g_layers == (0, 1)
    assert e_layers == (2, 3, 4)


def test_default_layer_split_unchanged_at_tinyvit_scale():
    # The literal this project's safety gate was already tuned against.
    g_layers, e_layers = default_layer_split(2)
    assert g_layers == (0,)
    assert e_layers == (1,)


def test_default_layer_split_rejects_a_base_too_shallow_for_both_prompts():
    with pytest.raises(ValueError, match="at least 2 transformer blocks"):
        default_layer_split(1)


@pytest.mark.parametrize("depth", [2, 3, 6, 12, 24])
def test_default_layer_split_is_always_disjoint_and_in_range(depth):
    g_layers, e_layers = default_layer_split(depth)
    assert not set(g_layers) & set(e_layers)
    assert max(g_layers + e_layers) < depth


def test_dual_prompt_g_prompt_length_and_pool_defaults_match_published_values():
    # dualprompt.yaml: g_prompt_length=5, length=5, size=10, top_k=1.
    wrapped = DualPromptViT(_tiny_base())
    assert wrapped.g_prompt.shape[2] == 5  # g_length
    assert wrapped.e_pool.shape[3] == 5  # e_length
    assert wrapped.e_pool.shape[0] == 10  # pool_size
    assert wrapped.top_k == 1


# --- end-to-end: the backbone wrapper wires the default through ---------


def test_dual_prompt_backbone_default_layers_at_both_scales(real_vit_b16):
    tiny = DualPromptViT(_tiny_base())
    assert tiny.g_layers == [0]
    assert tiny.e_layers == [1]

    real = DualPromptViT(real_vit_b16)
    assert real.g_layers == [0, 1]
    assert real.e_layers == [2, 3, 4]


def test_dual_prompt_factory_default_is_scale_derived():
    tiny = get_backbone("vit_dual_prompt")(base_model="tiny_vit", input_size=8, depth=2)
    assert tiny.g_layers == [0]
    assert tiny.e_layers == [1]


# --- CODA-Prompt: default_layers -- the rule itself ----------------------


def test_default_layers_matches_published_indices_at_vitb16_scale():
    # PILOT hardcodes e_layers=[0,1,2,3,4] regardless of prompt_param.
    assert default_layers(12) == (0, 1, 2, 3, 4)


def test_default_layers_at_tinyvit_scale():
    # Floored at the previous TinyViT-tuned literal (2 layers) -- a bare
    # ratio-based floor of 1 measurably regressed the revisit-safety gate's
    # above-chance accuracy assertion for CODA-Prompt at depth 2 (see
    # `default_layers`'s docstring / `_MIN_LAYER_COUNT`).
    assert default_layers(2) == (0, 1)


@pytest.mark.parametrize("depth", [1, 2, 3, 6, 12, 24])
def test_default_layers_is_always_in_range_and_nonempty(depth):
    layers = default_layers(depth)
    assert layers
    assert max(layers) < depth


def test_coda_prompt_pool_size_and_length_defaults_match_published_prompt_param():
    # coda_prompt.yaml: prompt_param=[100, 8.0, 0.0] -> pool_size=100, length=8.
    wrapped = CodaPromptViT(_tiny_base())
    assert wrapped.pool.pool_size == 100
    assert wrapped.pool.prompt.shape[3] == 8  # length


def test_coda_prompt_backbone_default_layers_at_both_scales(real_vit_b16):
    tiny = CodaPromptViT(_tiny_base())
    assert tiny.layers == [0, 1]  # unchanged from the pre-P11-B1 literal

    real = CodaPromptViT(real_vit_b16)
    assert real.layers == [0, 1, 2, 3, 4]


def test_coda_prompt_factory_default_is_scale_derived():
    tiny = get_backbone("vit_coda_prompt")(base_model="tiny_vit", input_size=8, depth=2)
    assert tiny.layers == [0, 1]


# --- CODA-Prompt: Gram-Schmidt still behaves at a non-tiny pool_size -----


def test_gram_schmidt_orthogonalizes_correctly_at_published_pool_size():
    """Task 1 explicitly asks to verify this: Gram-Schmidt orthogonalisation
    was only ever exercised at a handful of slots (pool_size<=6) before this
    phase; the published default is 100."""
    wrapped = CodaPromptViT(_tiny_base(), nb_experiences=5)  # pool_size=100 default
    assert wrapped.pool.slots_per_task == 20  # 100 // 5

    for task in range(5):
        wrapped.pool.start_new_task(task)
    assert wrapped.pool.unlocked.item() == 100

    # Every pair of unlocked prompt vectors must be (near-)orthogonal.
    flat = wrapped.pool.prompt.data.flatten(start_dim=1)  # [100, n_layers*2*length*embed_dim]
    gram = flat @ flat.t()
    off_diagonal = gram - gram.diag().diag()
    assert off_diagonal.abs().max().item() < 1e-3
