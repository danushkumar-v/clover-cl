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
| EASE | queued (P6) | growing per-task adapter list | growing-dim cosine + reweight |
| MOS | queued (P6) | EMA-merged per-task adapter | cosine prototype + CA |
| TUNA | queued (P6) | EMR-merged per-task adapter | per-task linear (concat) + CA |

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
≈ PILOT `exps/aper_aperpter.json`), scaled down for `TinyViT`:

| Key | PILOT value | CLOVER default | Note |
|---|---|---|---|
| `bottleneck_dim` (PILOT: `ffn_num`) | 64 | 8 | scaled to `TinyViT`'s `embed_dim=16` |
| `scale` (PILOT: `ffn_adapter_scalar`) | 0.1 | 0.1 | unchanged |
| adapter-tuning epochs | 20 | 40 (safety-gate default) | tuned for the tiny random backbone, see PLAN.md P5 log |
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
| `projection_dim` (PILOT: `M`) | 10000 | 256 | scaled down; PILOT's is sized for a 768-dim pretrained ViT |
| ridge grid | 17 values, 1e-8..1e8 | 6 values, 1e-3..100 | scaled range for the smaller feature space |
| `bottleneck_dim`/adapter-tuning epochs/lr | same as APER-Adapter | same as APER-Adapter | shared backbone mechanism |
| `use_simplecil` | present, dead config in PILOT (never read by `models/ranpac.py`) | not exposed | not reimplemented -- vestigial upstream |

## Judgment calls / scope simplifications (P6, APER-Adapter + RanPAC)

- **"First experience" generalizes PILOT's "`cur_task == 0`"**: both methods
  gate adapter training on `exp.task_label == 0` rather than a PILOT-style
  task counter, since CLOVER experiences are the direct equivalent.
- **No dropout in `Adapter`**: PILOT's adapter includes a dropout layer:
  omitted here because `clover/training/trainer.py` constructs
  `PerClassEvaluator` (which calls `.eval()` on the shared classifier
  module graph) once before the training loop even starts, so dropout would
  be permanently disabled for the rest of the run regardless -- adding it
  would be dead code across the whole framework, not just these two
  methods.
- **RanPAC's train/val ridge-selection split uses a locally-seeded
  generator** (`torch.Generator().manual_seed(exp.task_label)`), not the
  run's actual seed -- `TrainContext` doesn't thread the run seed through to
  methods. This only affects which ridge candidate is chosen, not the
  correctness of the closed-form solve.
