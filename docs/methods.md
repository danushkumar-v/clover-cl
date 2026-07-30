# CLOVER v2 methods

Clean-room reimplementations (SPEC §6.5) of 9 continual-learning methods.
No code is copied from LAMDA-PILOT; each method below cites the mechanism it
reimplements, notes deliberate scope simplifications, and records
hyperparameter provenance. CIFAR-100-vs-published-accuracy sanity runs are
deferred until P7/P8 lands a real CIFAR-100 dataset and the user has
Snellius cluster access (see `PLAN.md`'s Log) — everything else is verified
locally: the registry-driven revisit-safety gate (all 6 core scenarios) and
`clover smoke`.

| Method | Status | Backbone mechanism | Head |
|---|---|---|---|
| SimpleCIL | done (P3) | frozen backbone, no adaptation | cosine prototype |
| L2P | done (P5) | prompt-pool, token-prepend | gradient-trained linear |
| DualPrompt | done (P5) | general + expert prefix-KV | gradient-trained linear |
| CODA-Prompt | done (P5) | soft-combined prefix-KV pool | gradient-trained linear |
| APER-Adapter | done (P6) | one-shot adapter, dual-branch concat | cosine prototype |
| RanPAC | done (P6) | one-shot adapter, frozen after | random-projection ridge regression |
| EASE | done (P6) | growing per-task adapter list | growing-dim cosine + reweight |
| MOS | done (P6) | EMA-merged, one continuously-trained adapter | cosine prototype + CA |
| TUNA | done (P6) | EMR-merged per-task adapter | angular-margin-trained cosine + CA |

## Adapter family (P6)

All adapter-family methods insert an AdaptFormer-style bottleneck adapter
(down-proj → ReLU → up-proj, scaled, zero-init up-proj) as a parallel
branch off the pre-MLP residual stream in every transformer block
(`clover/backbones/adapter.py:Adapter`, spliced via `TinyViT`'s `adapter`
hook in `clover/backbones/vit.py`) — never wrapping `qkv`/`fc1`/`fc2`
directly. This matches every adapter-family method in LAMDA-PILOT
(confirmed via read-only research on `vit_adapter.py`/`vit_ease.py`/
`vit_mos.py`/`vit_tuna.py`); the methods differ in how many adapter copies
are live at once and how they're combined.

### APER-Adapter

Adapter is gradient-trained only during the first experience
(`exp.task_label == 0`), then frozen forever. The classifier is a cosine
prototype head (`IncrementalHead(cosine=True)`) over the **concatenation**
of a plain pass through the frozen base ViT and an adapter-tuned pass
through the *same* base weights (`clover/methods/aper_adapter.py`) —
mirrors PILOT's `MultiBranchCosineIncrementalNet` dual-branch design
without needing a second model instance, since `AdapterViT.base` is frozen
before any adapter training happens.

Hyperparameters (bench provenance: `bench/configs/methods/aper_adapter.yaml`
≈ PILOT `exps/aper_aperpter.json`); `bottleneck_dim` is derived from the
base's own `feature_dim` (P11-B2), recovering the published value exactly
at ViT-B/16 scale while leaving TinyViT unchanged:

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `bottleneck_dim` (PILOT: `ffn_num`) | 64 (at `feature_dim=768`) | `default_bottleneck_dim(feature_dim, 12)` — 64 at ViT-B/16 (`feature_dim=768`), 8 at TinyViT (`feature_dim=16`) | P11-B2: was a bare literal `8` (scaled for `TinyViT`'s `embed_dim=16`, wrong-by-construction on a real ViT); now `round(feature_dim/12)` floored at 8, `clover/backbones/adapter.py:default_bottleneck_dim` |
| `scale` (PILOT: `ffn_adapter_scalar`) | 0.1 | 0.1 | unchanged |
| adapter-tuning epochs | 20 | 40 (safety-gate default) | tuned for the tiny random backbone, see PLAN.md P5 log; training-hyperparameter work (not width/capacity) is P11-C's scope |
| adapter-tuning lr | 0.01 | 3e-2 (safety-gate default) | ditto |

### RanPAC

Same one-shot adapter tuning as APER-Adapter (shared
`clover/methods/adapter_common.py:AdapterMethodBase`), but the head is a
frozen random-projection ridge-regression solve
(`clover/methods/heads.py:RandomProjectionRidgeHead`): `relu(features @
W_rand)` expands frozen features into a wider random nonlinear space; `G`
(Gram matrix) and `Q` (feature/label correlation) sufficient statistics
accumulate across experiences; `weight = solve(G + ridge*I, Q)` is
recomputed in closed form every experience. The ridge value is chosen per
experience by a small grid search against an 80/20 held-out split of that
experience's own data (mirrors PILOT's `optimise_ridge_parameter`).

Hyperparameters (bench provenance: `bench/configs/methods/ranpac.yaml`):

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `projection_dim` (PILOT: `M`) | 10000 (at `feature_dim=768`) | `_default_projection_dim(feature_dim)` — 10000 at ViT-B/16 (`feature_dim=768`), 256 at TinyViT (`feature_dim=16`) | P11-B2: was a bare literal `256` (explicitly reduced to suit `TinyViT`'s 16-dim feature); now `round(feature_dim * 10000/768)` floored at 256, `clover/methods/ranpac.py:_default_projection_dim`. **Ridge-solve cost at `M=10000`**: `torch.linalg.solve` on the resulting `[10000,10000]` `G` matrix measured ~6.5-8.3s/solve on a 4-thread CPU, ~53s for one experience's full ridge-grid-search-plus-solve, ~530s (~9 min) projected over a 10-experience stream -- see `docs/real_backbones.md`'s parameter-budget section. Tractable, not free; not reduced |
| ridge grid | 17 values, 1e-8..1e8 | 6 values, 1e-3..100 | scaled range for the smaller feature space; unchanged by P11-B2 (a training-search-range choice, not a structural width) |
| `bottleneck_dim`/adapter-tuning epochs/lr | same as APER-Adapter | same as APER-Adapter | shared backbone mechanism -- see APER-Adapter's row above for the P11-B2 derivation |
| `use_simplecil` | present, dead config in PILOT (never read by `models/ranpac.py`) | not exposed | not reimplemented -- vestigial upstream |

### EASE

`clover/backbones/adapter_ease.py:EaseAdapterViT` keeps a growing
`nn.ModuleList` of per-experience adapter sets -- only the most recent one
is ever trainable; `grow()` (called from `EASE.before_experience`, not
`after_experience` -- see "Judgment calls" below) freezes it and allocates
a fresh one. At evaluation time, the same image passes through *every*
stored adapter set and the resulting `[CLS]` features concatenate into a
growing-width tensor (`clover/methods/ease.py`).

The head (`clover/methods/heads.py:EaseHead`) stores `weight` as
`[num_classes, num_blocks, block_dim]`, tracking each class's "home block"
(the block it was first introduced in). At inference, cosine similarity is
computed per block and summed, with non-home blocks down-weighted by
`alpha` -- PILOT's `EaseCosineLinear.forward_reweight`. New classes get
real prototype rows in *every* block (old adapters are still available and
this experience's images are still on hand, so there's no need to
approximate); already-existing classes' row in the newly added block is
filled by a cosine-similarity-weighted combination of this round's new
classes' own rows in that block -- a structurally faithful simplification
of PILOT's `solve_similarity`/`solve_sim_reset` (their images are gone, so
a real value can't be recomputed, matching why PILOT itself needs an
approximation there too).

Per-experience training uses a throwaway proxy head
(`IncrementalHead(cosine=False)`, sized to the full global head so it
works directly with `new_class_ce`) over just the current (single)
adapter's features -- P2's loss policy already makes EASE's real
`aux_targets`/`ignore_index=-1` patch unnecessary, and matches its
*effect* precisely: only new-class samples contribute to the loss,
revisited-class samples in the same batch are skipped.

Hyperparameters (bench provenance: `bench/configs/methods/ease.yaml`):

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `bottleneck_dim` (PILOT: `ffn_num`) | 64 (at `feature_dim=768`) | same derivation as APER-Adapter/RanPAC -- 64 at ViT-B/16, 8 at TinyViT | P11-B2: was a bare literal `8`; now `default_bottleneck_dim(feature_dim, 12)`. **Stored state grows every experience** (one full 12-block, 64-wide adapter set per experience, no merging) -- ~11.9M params after 10 experiences at ViT-B/16 scale, unbounded in stream length by design; see `docs/real_backbones.md`'s parameter-budget section (flagged prominently there, not capped) |
| `alpha` | 0.1 | 0.1 | unchanged |
| `beta` (init-PTM block weight) | 0 | not exposed | `use_init_ptm=false` upstream default -- not reimplemented |
| `use_diagonal` | false | not exposed | vestigial ablation flag upstream |
| adapter-tuning epochs/lr | init_epochs=20, init_lr=0.025 | 40 / 3e-2 (safety-gate default) | tuned for the tiny random backbone; training-hyperparameter work is P11-C's scope |

### MOS

Unlike EASE (a fresh adapter per experience) or APER-Adapter/RanPAC (one
adapter, trained once), MOS (`clover/backbones/adapter_mos.py:
MOSAdapterViT`) keeps a **single adapter training continuously across the
whole run** -- this is the "Mixture-of-Subspace" name. After every
optimizer step, `merge_step()` EMA-blends the current adapter's parameters
toward the running mean of all earlier experiences' frozen snapshots
(`momentum`), regularizing it against drifting too far from its own
history. `snapshot()` (freeze a copy into `adapter_list`, fold it into the
running sum) is called from `MOS.before_experience` for the same
resume-safety reason as EASE's `grow()` (see "Judgment calls").

The head is a **fixed-width** `IncrementalHead(cosine=True)` (unlike
EASE's growing dim) -- shared across every adapter era. New classes get
prototype rows set directly (like SimpleCIL/APER-Adapter); the main
training loop also gradient-trains the head jointly with the adapter via
`new_class_ce`, but those values only matter for the *new* classes'
convergence, since `set_prototype` overwrites their rows afterward anyway
(mirrors PILOT's own `replace_fc` following its gradient-trained
`_init_train` pass).

At evaluation time, the classifier ensembles every stored adapter's
prediction (`clover/methods/mos.py:_Classifier`): each adapter's features
pass through the *same* shared head, and the resulting logits are
averaged. **Simplified out**: PILOT's entropy-based test-time
self-refinement search (iteratively re-predicting with whichever adapter
gave the lowest-entropy output) -- a plain average captures "combine
multiple adapters' predictions" without the iterative search, and this was
agreed as a scope simplification before implementing (the EMA merge
itself, MOS's actual headline mechanism, is implemented in full).

**Classifier alignment (CA)**: new shared
`clover/methods/classifier_alignment.py` (`update_class_stats`,
`gaussian_resample_finetune`) -- stores each class's `(mean, covariance)`
(damped by `1e-4 * I`, matching PILOT's own damping) from real features
once, computed via the *current* adapter; after every experience past the
first, the head is fine-tuned via cross-entropy on synthetic features
resampled from `N(mean, cov)` for every class seen so far (old and new),
via an explicit `torch.Generator` (Cholesky + `randn`, not
`torch.distributions.MultivariateNormal.sample()`, which doesn't accept
one) seeded from `exp.task_label` -- not the global RNG, since this runs
inside `train_experience`, which resume-replay skips for already-completed
experiences. Reused as-is by TUNA.

Hyperparameters (bench provenance: `bench/configs/methods/mos.yaml`):

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `bottleneck_dim` (PILOT: `ffn_num`) | 16 (at `feature_dim=768`; not 64 like APER-Adapter/RanPAC/EASE) | `default_bottleneck_dim(feature_dim, 48)` — 16 at ViT-B/16, 8 at TinyViT | P11-B2: was a bare literal `8`; MOS's own published ratio (1:48) is narrower than the other three's (1:12), so it gets its own constant (`MOS_TUNA_BOTTLENECK_RATIO`) rather than reusing APER-Adapter's |
| `momentum` (PILOT: `adapter_momentum`) | 0.1 | 0.1 | unchanged |
| `crct_epochs` | 30 | 10 (constructor default) | reduced for the tiny synthetic setup; not yet config-driven (see below) |
| `ca_lr` | 0.005 | 5e-3 | unchanged |
| `reg` (orthogonality regularizer weight, always-applied in PILOT -- unlike TUNA's optional `use_orth`) | 0.1 | not exposed | not reimplemented -- a secondary regularizer layered on the main loss, not MOS's headline mechanism (the EMA merge) |
| two-param-group LR split (adapter vs. head at 10x lower LR) | present | not reimplemented | single shared LR for adapter+head; a scope simplification, not yet revisited |
| `ensemble`/entropy self-refinement | true / present | plain logit average | see "Simplified out" above |

### TUNA

Like EASE, `clover/backbones/adapter_tuna.py:TunaAdapterViT` grows a fresh
adapter per experience (only the most recent one trainable); unlike EASE's
concatenation or MOS's continuous EMA, TUNA combines every stored
experience's *frozen* adapter into a single consensus adapter via
**EMR-merge** (Exclusive Mask and Rescale, confirmed via read-only research
on `models/tuna.py`/`backbone/vit_tuna.py`'s `TaskVector`/`emr_merge`):
per parameter tensor, elect a sign by majority vote across every stored
task's adapter, keep the max-magnitude value among the task tensors
agreeing with that sign (zeroing the rest), then rescale so the merged
tensor's average magnitude matches the mean task tensor's -- otherwise
electing the max at every position would systematically inflate the
merged tensor's scale. `recompute_merge()` runs at the end of
`train_experience` (a value-only update, unlike the structural adapter
-list growth in `before_experience`) so evaluation of the *same* round
already reflects its own just-finished training, matching PILOT's own
cadence (merge runs right after a task's adapter is appended, not deferred
to the next task).

The classifier uses the merged adapter **directly** for evaluation
(`clover/methods/tuna.py:_Classifier`) -- PILOT's per-sample entropy-based
adapter routing (predict with every stored adapter, pick the
lowest-entropy one, ensemble with the merged "general" prediction) is
simplified out, agreed before implementing: this is more faithful to
"TUNA" than skipping the merge itself would be, and simpler than
reimplementing per-sample routing.

The head is a plain global-width `IncrementalHead(cosine=True)` (not
PILOT's `TunaLinear` -- a list of separate per-task `nn.Linear` objects,
concatenated), trained via a new CosFace-style angular-margin loss
(`clover/methods/losses.py:angular_margin_ce`, restricted to new-class
samples/columns via the same `masked_logits` primitive `new_class_ce`
uses). PILOT's per-task-heads design is an implementation detail for
isolating gradient to the newest task's own columns during training --
`angular_margin_ce`'s masking already gives that for free, so no separate
per-task `nn.Linear` list is needed. PILOT's own default margin (`m=0.0`)
makes `angular_margin_ce` numerically identical to `new_class_ce`; TUNA
still gets its own loss function (not a bare reuse) so a nonzero margin
is genuinely supported. No closed-form prototype-overwrite step exists for
TUNA (unlike MOS/APER-Adapter) -- the head is purely gradient-trained via
the angular loss, matching what the research found for PILOT's own TUNA.
Reuses `clover/methods/classifier_alignment.py` (built for MOS) as-is;
per-class stats are computed via the *merged* adapter's features (not the
still-training current one), matching what evaluation will actually see.

Hyperparameters (bench provenance: `bench/configs/methods/tuna.yaml` ≈
PILOT `exps/tuna_cifar.json`):

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `bottleneck_dim` | 16 (at `feature_dim=768`; PILOT hardcodes this in `init_adapters()`, ignoring its own `r` config key entirely) | `default_bottleneck_dim(feature_dim, 48)` — 16 at ViT-B/16, 8 at TinyViT | P11-B2: was a bare literal `8`; PILOT's `r: 16` hyperparameter is dead config upstream, not reimplemented as a separate knob -- same ratio (1:48) and same helper as MOS. **Stored state grows every experience** (one adapter per task plus the merged consensus adapter, no pruning) -- ~3.3M params after 10 experiences at ViT-B/16 scale (see `docs/real_backbones.md`'s parameter-budget section) |
| `margin` (PILOT: `m`) | 0.0 | 0.0 | unchanged -- PILOT's own default disables the angular margin |
| `scale` (PILOT: cosface `s`) | 20.0 | not a separate knob | folded into whatever scale `IncrementalHead(cosine=True)`'s own learned scale produces -- argmax/accuracy is scale-invariant regardless (see `angular_margin_ce`'s docstring) |
| `use_orth` | false | not exposed | disabled by PILOT's own default; not reimplemented |
| `decay` | false | not exposed | disabled by PILOT's own default (and would misbehave under CLOVER's uneven task sizes if ever turned on, per the original research); not reimplemented |
| `crct_epochs`/`ca_lr` | 30 / 0.005 | 10 (constructor default) / 5e-3 | same reduction as MOS, for the tiny synthetic setup |

## Judgment calls / scope simplifications (P6)

- **"First experience" generalizes PILOT's "`cur_task == 0`"** (APER-Adapter/
  RanPAC): both methods gate adapter training on `exp.task_label == 0`
  rather than a PILOT-style task counter, since CLOVER experiences are the
  direct equivalent.
- **No dropout in `Adapter`**: PILOT's adapter includes a dropout layer:
  omitted here because `clover/training/trainer.py` constructs
  `PerClassEvaluator` (which calls `.eval()` on the shared classifier
  module graph) once before the training loop even starts, so dropout would
  be permanently disabled for the rest of the run regardless -- adding it
  would be dead code across the whole framework, not just these methods.
- **RanPAC's train/val ridge-selection split uses a locally-seeded
  generator** (`torch.Generator().manual_seed(exp.task_label)`), not the
  run's actual seed -- `TrainContext` doesn't thread the run seed through to
  methods. This only affects which ridge candidate is chosen, not the
  correctness of the closed-form solve.
- **EASE's `grow()` (freeze-then-allocate) is called from
  `before_experience`, not `after_experience`**: the Trainer's resume-replay
  only re-invokes `before_experience` for already-completed experiences, so
  any structural growth that must exist before `load_state_dict` runs has
  to live there -- the same reason `IncrementalHead.expand_to`/CODA-Prompt's
  `pool.start_new_task` are called from `before_experience` too. Putting
  `grow()` in `after_experience` instead would silently drop the final
  experience's adapter from `adapter_sets` on a fresh (non-resumed) run,
  since no `before_experience(N)` call ever follows the last experience
  `N-1`.
- **EASE's proxy head is full-global-width** (like `AdapterMethodBase`'s
  `_train_head`), not locally indexed to just this round's new classes
  (PILOT's `proxy_fc` is locally indexed) -- avoids a second
  targets-remapping mechanism alongside `new_class_ce`'s existing
  set-membership masks; functionally equivalent since only the new-class
  columns/rows ever receive gradient either way.
- **`torch.distributions.MultivariateNormal.sample()` draws from the
  global RNG and doesn't accept an explicit generator** -- caught while
  designing MOS's classifier alignment, before it became a resume-safety
  bug (same class of issue P3's DataLoader-RNG fix addressed: CA runs
  inside `train_experience`, which resume-replay skips for completed
  experiences). `classifier_alignment.py` samples by hand instead
  (Cholesky decomposition + `torch.randn(..., generator=...)`), seeded
  from `exp.task_label`.
- **TUNA's merge-recomputation timing differs from EASE's/MOS's
  structural-growth timing on purpose**: `TunaAdapterViT.grow()` (append a
  fresh trainable adapter, freeze the previous one) is structural and
  lives in `before_experience`, same as EASE/MOS; but
  `recompute_merge()` (recompute `merged_adapter`'s *values* via EMR-merge)
  lives in `train_experience`, right after that round's training finishes
  -- it's a value-only update (the merged adapter's shape never changes),
  so it doesn't have the same resume-replay constraint structural growth
  does, and doing it here (rather than deferring to the next round's
  `before_experience`) matches PILOT's own cadence: evaluating round N
  already reflects round N's own contribution to the merge, not a
  one-round-stale version of it.
- **TUNA's per-class CA stats are computed via the *merged* adapter's
  features, not `forward_current`'s** (the still-training current
  adapter) -- unlike MOS (whose evaluation ensembles every adapter
  including the current one, so `forward_current` is a reasonable proxy
  for "the current era"), TUNA's evaluation uses *only* the merged
  adapter, so aligning the head against any other feature space would
  train it for a distribution it's never actually evaluated against.
- **TUNA's head reuses `IncrementalHead(cosine=True)` directly rather than
  reimplementing PILOT's `TunaLinear`** (a list of separate per-task
  `nn.Linear` objects, concatenated, never frozen) -- confirmed while
  designing this that `TunaLinear`'s per-task-heads design is an
  implementation detail for isolating gradient to the newest task's own
  output columns during training, a property `angular_margin_ce`'s
  `masked_logits`-based restriction already provides without needing
  separate per-task objects. Also confirmed PILOT's own default margin
  (`m=0.0`) makes the angular loss numerically identical to plain
  `new_class_ce` -- `angular_margin_ce` still exists as its own function
  (not a bare alias) so a nonzero margin is genuinely supported, not just
  documented as unsupported.
