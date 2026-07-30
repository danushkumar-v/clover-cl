# Real pretrained backbones: what changed, and why

Status: **in progress (P11).** The "before" columns are final and were
measured; the "after" columns are filled in as each method family lands.

## The problem this document records

CLOVER v2 was built and tested against `TinyViT` — a clean-room minimal
Vision Transformer (`depth=2`, `embed_dim=16`, 4 patches, randomly
initialised) sized for the 8×8 synthetic dataset that the CI safety gate
runs on. That was a deliberate choice: it makes the whole test suite run on
CPU in seconds, and it let every method be validated for *structural*
correctness — finite losses, revisit-safe label handling, resume fidelity —
without a GPU or a dataset download.

It also meant that until now, **only methods whose backbone contract is a
plain `forward(x) -> features` could use a real pretrained checkpoint.**
That is exactly one of the nine: SimpleCIL. The other eight are
prompt-based (L2P, DualPrompt, CODA-Prompt) or adapter-based
(APER-Adapter, RanPAC, EASE, MOS, TUNA), and they all work by splicing
*into* a transformer's internals — prepending prompt tokens, injecting
prefix key/value pairs into specific attention blocks, or hanging parallel
bottleneck branches off each block's MLP. A stock `timm`
`VisionTransformer` offers no way in: its `Block.forward` accepts only `x`,
and its `Attention.forward` has no key/value prefix.

The measured consequence, from two completed cluster batches on CIFAR-100:

| batch | backbone | methods | AIA |
|---|---|---|---|
| `matrix_simplecil_vitb16` | ViT-B/16, ImageNet-pretrained | SimpleCIL only | **0.82** |
| `matrix_full` | TinyViT, random init | all 9 | **0.08 – 0.11** |

SimpleCIL is the same code in both, so the 10× gap is the backbone alone.
Chance accuracy on 100 classes is 0.01, and a stream-averaged AIA around
0.08–0.11 is what an untrained feature extractor produces. `matrix_full`
therefore validates that the *framework* runs end to end at scale — 189/189
runs completed on Snellius, resumable, with finite metrics throughout — but
its accuracies are not a method comparison and must never be presented as
one.

## Three layers of change (not one)

Wiring a real backbone in is not a single patch. Three independent things
were sized for TinyViT:

**1. The hook surface.** *(landed, P11-A.)* `TinyViT` exposes
`patch_tokens`, `query_features`, `forward_tokens(tokens, prefix_kv=,
adapter=)` and `forward(x, prompt_tokens=, prefix_kv=, adapter=)`. timm
exposes none of them. Fix: `clover/backbones/timm_vit.py:TimmViTHooks`,
which re-expresses timm's block and attention forward *through timm's own
submodules* — so pretrained weights load and behave normally — with the two
hook points inserted at exactly the positions `TinyViT` uses.
`resolve_base_model` applies it automatically to any ViT-shaped timm model;
anything else passes through untouched.

Semantic fidelity is the bar, and three properties are pinned by test:
a hook-free forward reproduces timm's own `forward` exactly; the adapter
branch reads the post-attention residual stream (not `norm2(x)`); and
`prefix_kv` is concatenated onto K/V after the q/k norms with queries
untouched, so the output sequence length never changes. A method is
structurally identical on both backbones; only scale differs.

Four consequences worth knowing:

- The wrapper owns the timm model as a submodule, so backbone `state_dict`
  keys gain a `base.` prefix — a checkpoint written before this change
  cannot be resumed by a run after it.
- `TimmViTHooks` pools the class token (matching `TinyViT`) rather than
  honouring a base's `global_pool="avg"`.
- `query_features` mean-pools raw patch-embedding output, exactly as
  `TinyViT` does, and this is still true of the *backbone* layer today —
  correctly so, since the mechanism must not change with the base. **Fixed
  at the method layer (P11-B1):** L2P/DualPrompt/CODA-Prompt's published
  query is a full frozen forward's class token (`l2p.yaml`/
  `dualprompt.yaml`: `get_original_backbone: true`, `embedding_key: "cls"`),
  and prompt selection now actually gets it —
  `clover/methods/prompt_common.py:published_query` builds a no-grad,
  hook-free `base(x)` call (the pooled class-token feature every base
  already returns from a plain forward), and `PromptMethodBase.build()`
  overrides each wrapper's `query_fn` with it. Costs one extra full forward
  through the base per batch (the mechanism's own forward, with the
  selected prompts spliced in, still runs separately — the query must be
  known before it can select what to splice in, so the two forwards can't
  merge into one). Pinned by `tests/test_prompt_query_fix.py`: perturbing a
  late block's weights now changes the query on both `TinyViT` and a real
  timm base, which was false before this fix (the old query never touched
  any block).
- A `base_model: vit_*_224` config cannot be run under `--profile smoke`.
  Smoke substitutes the 8×8 synthetic dataset but keeps the method block,
  and a 224-resolution timm model rejects an 8×8 input. `resolve_base_model`
  deliberately drops CLOVER's `input_size` for timm models (a timm model's
  resolution comes from its own pretrained config), and overriding
  `img_size` alone wouldn't help — patch 16 doesn't tile an 8×8 image.
  A smoke-clean example config therefore has to leave `base_model` at its
  `tiny_vit` default.

**2. Scale-dependent structural defaults.** Several methods carry constants
that are only meaningful at TinyViT's dimensions. DualPrompt's default
general/expert prompt layers are `(0,)` and `(1,)` — which exist because
TinyViT has exactly two blocks. RanPAC's random-projection width was
reduced from the published 10000 to 256 to suit a 16-dimensional feature.
Adapter bottlenecks are 8 wide against a 16-dimensional stream. Run these
unchanged on ViT-B/16 and nothing crashes — the methods simply stop being
the published methods. Fix: derive every such constant from the base
model's own `depth`/`feature_dim`, defaulting to the published value at
ViT-B/16 scale.

**3. Training hyperparameters.** *(landed, P11-C.)* The safety gate uses
`epochs=40`, `lr=3e-2` — values found empirically to make a *randomly
initialised* TinyViT converge on synthetic 8×8 images, and documented as
such. They are actively harmful for a pretrained ViT.

This turned out to be the layer with the sharpest teeth, because it fails
*silently*. The nine published configs disagree far more than expected:

| | spread across the 9 |
|---|---|
| optimizer | SGD (6 methods) vs Adam (3) |
| learning rate | 0.001 → 0.03 (**30×**) |
| batch size | 16 → 256 (**16×**) |
| weight decay | 0.0 → 0.05 |
| schedule | cosine (5) / constant (2) / none |

`configs/matrix_full.yaml` applied one shared `{adam, lr 1.875e-3,
epochs 5, batch 16}` block to all nine — so MOS, published as SGD at 0.03,
would have trained on the wrong optimizer at roughly 1/16 its intended
learning rate. **That invalidates a 9-method comparison on its own,
independently of the backbone problem above**, and unlike chance-level
accuracy it would not announce itself: the numbers would look plausible.

The config layer could not express the fix. Two gaps, both now closed:

- **Missing knobs.** `TrainingSection` gained `scheduler`
  (`constant`/`cosine`) and `min_lr`; `OptimizerConfig` gained
  `weight_decay`. `Trainer` builds a `CosineAnnealingLR` with
  `T_max=epochs`, annealing *within* an experience — each experience
  restarts its own optimizer over its own trainable subset, which matches
  PILOT, where the scheduler is likewise rebuilt per task.
- **No per-method training block.** `MatrixSection.method_overrides`
  reaches only the *method* section, so every cell of a matrix shared one
  `training` block. New `training_overrides` layers per method, merging
  `optimizer` key-wise so an override can set just `lr` without restating
  the optimizer's name.

One structural note worth recording: **the Trainer does not own the epoch
loop.** Every gradient-trained method runs its own
`for _epoch in range(ctx.epochs)` inside `train_experience`, so a per-epoch
schedule can only be stepped by the method. Hence
`TrainContext.make_scheduler(optimizer)`, called at each of the five epoch
loops (`prompt_common`, `adapter_common`, `ease`, `mos`, `tuna` —
SimpleCIL needs none, being closed-form). A scheduler that were built but
never stepped would be a silent no-op, so a test pins that stepping it
actually anneals the learning rate.

The published values ship in `configs/matrix_all9_vitb16_cifar224.yaml` and
`configs/matrix_all9_vitb16_inr.yaml`, with provenance per method in
`docs/methods.md`.

## Per-method change log

Filled in as each family lands. "Before" = state at commit `7ffe941`.

| method | family | before (TinyViT-scale) | after (ViT-B/16) | why |
|---|---|---|---|---|
| SimpleCIL | none (plain forward) | already worked with a real timm ViT | unchanged | its base *is* its backbone; no splice needed |
| L2P | prompt pool | `pool_size=10, prompt_length=5, top_k=4`; query = patch-embed-mean (no class-token forward) | `pool_size=10, prompt_length=5, top_k=5` (`clover/backbones/prompt_pool.py`); query = full frozen-backbone forward's class token (`clover/methods/prompt_common.py:published_query`) | `l2p.yaml`: `size=10, length=5, top_k=5`, `get_original_backbone=true`, `embedding_key="cls"` — none of L2P's sizes are depth-dependent (prompt tokens are prepended once, at input, not per-block), so `top_k` was the only literal to fix; the query gap was the real defect (see "three layers of change" above) |
| DualPrompt | prefix K/V | `g_layers=(0,)`, `e_layers=(1,)` — fits depth 2 only; `g_length=e_length=2`; query = patch-embed-mean | `g_layers`/`e_layers` derived from depth via `default_layer_split(depth)` (`clover/backbones/dual_prompt.py`) — recovers PILOT's `[0,1]`/`[2,3,4]` exactly at depth 12, unchanged `(0,)`/`(1,)` at depth 2; `g_length=e_length=5`; query fixed the same way as L2P | layer indices must derive from depth (as before); `dualprompt.yaml`: `g_prompt_length=5`, `length=5`, `g_prompt_layer_idx=[0,1]`, `e_prompt_layer_idx=[2,3,4]`, `get_original_backbone=true` |
| CODA-Prompt | prefix K/V, soft | `pool_size=10, length=2, layers=(0,1)` — fixed literal, not depth-derived; Gram-Schmidt only ever exercised at <=6 slots; query = patch-embed-mean | `pool_size=100, length=8` (`coda_prompt.yaml`'s `prompt_param=[100, 8.0, 0.0]`); `layers` derived from depth via `default_layers(depth)` (`clover/backbones/coda_prompt.py`) — recovers PILOT's hardcoded `[0,1,2,3,4]` exactly at depth 12, floored at the previous literal (`(0,1)`, numerically unchanged) at depth 2; query fixed the same way as L2P/DualPrompt (CODA-Prompt's soft combination is *also* query-keyed, via `CodaPromptPool.combine`'s per-slot attention weights, so it benefits from the same fix even without a hard top-k) | PILOT hardcodes `e_layers=[0,1,2,3,4]` in `CodaPrompt._init_smart` regardless of `prompt_param`, so it's a ratio (5 of 12 blocks) once a different-depth base is involved, not a literal; a bare ratio-based floor of 1 block measurably regressed the revisit-safety gate's above-chance accuracy assertion at TinyViT's depth 2 (`exact_replay`/`long_range_revisit`/`partial_overlap`), caught by re-running the full gate before considering this done — floored at 2 (the prior literal) instead, same shape of fix as the adapter family's `default_bottleneck_dim`; Gram-Schmidt re-verified orthogonal at the published `pool_size=100` (`tests/test_prompt_scale_defaults.py`) |
| APER-Adapter | adapter | bottleneck 8 (literal) | bottleneck 64 (`default_bottleneck_dim(768, 12)`) | PILOT `ffn_num=64` at 768-dim (`aper_adapter.yaml`); derived from `feature_dim`, floored at 8 so TinyViT is unchanged |
| RanPAC | adapter + RP head | `M=256` (literal, published: 10000) | `M=10000` (`_default_projection_dim(768)`) | `ranpac.yaml`'s `M=10000` restored exactly at 768-dim; scaled proportionally elsewhere, floored at 256 so TinyViT is unchanged. Ridge solve at `M=10000` measured ~50-80s/experience on a 4-thread CPU (see parameter budget below) |
| EASE | growing adapters | one set per experience, 2 blocks, bottleneck 8 (literal) | one set per experience, 12 blocks, bottleneck 64 | `ease.yaml`'s `ffn_num=64`, same derivation as APER-Adapter (shared ratio: both use PILOT's 64-at-768 width); head is `[classes, num_experiences, 768]`, growing every experience — see parameter budget |
| MOS | EMA-merged adapter | bottleneck 8 (literal) | bottleneck 16 (`default_bottleneck_dim(768, 48)`) | `mos.yaml`'s `ffn_num=16` — narrower than APER/EASE/RanPAC's 64, its own published ratio (1:48) |
| TUNA | EMR-merged adapters | bottleneck 8 (literal) | bottleneck 16 | PILOT hardcodes 16 in `init_adapters()` (its own `r: 16` config key is dead code upstream — `tuna.yaml`); same ratio as MOS |

## Configuration surface

Two different keys, and the distinction matters:

```yaml
# SimpleCIL has no mechanism wrapper -- its base IS its backbone:
method: {name: simplecil, backbone: vit_base_patch16_224, pretrained: true}

# The other eight keep their mechanism and swap the base *underneath* it:
method: {name: l2p, base_model: vit_base_patch16_224, pretrained: true}
```

Setting `backbone:` on a wrapper method replaces the whole mechanism with a
bare ViT — the prompt pool or adapter stack silently disappears, and the
method degenerates to a linear probe. The shipped configs use the correct
key per method; if you write your own, check this first. The safety gate
derives the right key per method from the backbone factory's signature
(`base_model` present ⇒ it's a mechanism wrapper), so a new method is
classified automatically rather than added to a list.

### Validating a config for a dataset you don't have

Resolving a config instantiates the dataset purely to read `num_classes`,
so a config naming an unstaged dataset couldn't be checked at all on a
machine without that data — i.e. on the GPU-less box where configs are
actually written. Declare the count and `clover preflight` / `clover
inspect` work offline:

```yaml
stream: {dataset: imagenet_r, dataset_num_classes: 200, init_cls: 20, increment: 20}
```

The declaration stands in for absent data, never overriding it: `clover
run` reads the count from the dataset it actually loads and rejects a
declaration that contradicts it.

## Parameter budget at ViT-B/16 scale

Measured (P11-B2), not estimated: constructed each backbone/head at
ViT-B/16 scale (`depth=12`, `feature_dim=768`) with the scale-derived
defaults above, drove the same growth calls the real methods make
(`grow()`/`snapshot()`/`add_block()`) for a 10-experience stream
(`init_cls=10, increment=10` ⇒ 100 classes after 10 experiences, matching
the bench YAMLs' convention), and summed `numel()` over every stored
parameter. One `Adapter` set = one `Adapter` (down-proj + up-proj, each
with bias) per transformer block × `depth=12`; at `bottleneck_dim=B`, one
set costs `12 × (2 × 768 × B + B + 768)` parameters.

| method | per-experience trainable | total stored after 10 experiences | head size | note |
|---|---|---|---|---|
| APER-Adapter | one adapter set, exp 0 only (~1.19M params, bottleneck 64) | **fixed**: 1,189,632 (adapter, frozen after exp 0) | `[100, 1536]` = 153,600 (dual-branch concat doubles width) | adapter never grows; only the head's prototype rows are rewritten per experience |
| RanPAC | one adapter set, exp 0 only (shares APER-Adapter's mechanism) | **fixed**: 1,189,632 (adapter) + `G` `[10000,10000]`=100,000,000 + `Q` `[10000,100]`=1,000,000 + `w_rand` `[768,10000]`=7,680,000 + `weight` `[100,10000]`=1,000,000 | see above (`w_rand`/`G`/`Q`/`weight` are the "head") | `G` alone is 400MB (fp32) at the published `M=10000` — the dominant cost in this whole family; **ridge-solve tractability finding below** |
| EASE | one new adapter set/experience (~1.19M params, bottleneck 64) | **grows every experience**: 1,189,632 × 10 = 11,896,320 (adapters) + head `[100,10,768]`=768,000 = **12,664,320 total** | `[100, 10, 768]` = 768,000, width grows by 768 every experience | flagged: adapter storage grows *linearly and unboundedly* with stream length — a 20-experience CIFAR-224 stream would store ~24M adapter params, a 50-experience one ~60M, larger than the ViT-B/16 backbone itself (~86M) well before that |
| MOS | one continuously-trained adapter (bottleneck 16, ~304K params) | **fixed** (not growing): cur_adapter + 9 frozen snapshots + 1 running-sum accumulator = 11 same-shaped sets × 304,320 = 3,347,520 | `[100, 768]` = 76,800 (fixed-width, unlike EASE) | snapshots accumulate one per experience but each is the *same* small width (16), so growth is linear in experience count but at 16x smaller per-unit cost than EASE's 64-wide sets — no head growth at all |
| TUNA | one new adapter/experience (bottleneck 16, ~304K params) | **grows every experience**: 304,320 × 10 (per-task) + 304,320 × 1 (merged) = 3,347,520 | `[100, 768]` = 76,800 (fixed-width, plain `IncrementalHead`) | same linear-growth shape as EASE but at MOS's narrower bottleneck (16 vs 64), so ~3.5x less storage per experience than EASE for the same stream length |

**Headline finding — EASE's growth is the one to flag prominently.** Every
other method in this family is either fixed-cost (APER-Adapter/RanPAC: one
adapter, trained once) or grows at a narrow 16-wide bottleneck (MOS/TUNA).
EASE grows a *full 64-wide, 12-block* adapter set every single experience,
with no merging or pruning — storage is `O(n_experiences × depth ×
bottleneck × feature_dim)`, unbounded in stream length by design (the
method's own mechanism: it needs every past adapter set still callable at
inference to reconstruct old classes' features). At 10 experiences this is
already ~11.9M parameters (larger than RanPAC's or TUNA's *entire* stored
state); a real CIFAR-224/ImageNet-R run with more experiences will keep
growing linearly with no ceiling. This is reported here rather than capped
silently, per this phase's instructions — whether it needs an eviction or
merging strategy is a design question for whoever schedules the real
cluster matrices, not something to quietly shrink in this pass.

**RanPAC ridge-solve tractability at the published `M=10000`.** `G` is
`[10000, 10000]`; `torch.linalg.solve(G + ridge·I, Q)` was timed directly
on this CPU (no GPU available locally):

- A single `[10000,10000]` solve: **~6.5-8.3s** (3 trials, 4 BLAS threads).
- `RanPAC._update_head`'s real per-experience cost is *7* solves (a 6-value
  ridge grid search against an 80/20 held-out split, plus 1 final solve
  after `accumulate`), plus the `phi.t() @ phi`/`phi.t() @ y` matmuls that
  build `trial_g`/`trial_q`: **~53s measured end-to-end for one experience**
  (500 synthetic samples, 100 classes).
- Projected over a 10-experience stream: **~530s (~9 minutes)** spent
  purely in RanPAC's closed-form head update, on top of backbone forward
  passes — before any dataset-specific cost.

This machine has no GPU, so all of the above was measured on CPU
(4 threads) -- `G`/`Q`/`weight`/`w_rand` are registered buffers/parameters
of `RandomProjectionRidgeHead`, so they *do* move to CUDA along with the
rest of the head (`RanPAC.before_experience`'s `self.head =
self.head.to(ctx.device)`), and `torch.linalg.solve` would run on the GPU
(cuSOLVER) on a real cluster job -- almost certainly much faster than this
CPU number, but not measurable here. Reported here rather than silently
shrinking `M`, per this phase's instructions; P11-C/cluster scheduling
should get a real GPU timing before assuming this is free, since a
9-method × 7-scenario × multi-dataset matrix that includes RanPAC will
otherwise be planned against an unverified assumption.
