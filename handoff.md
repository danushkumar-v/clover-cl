# Handoff — clover-cl v2 (2026-07-06: P6 complete, all 9 methods done)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **This session's
  full P6 work (all 5 remaining methods) is committed locally in 4 commits
  but not yet pushed** — push only if the user asks.
- `main` is untouched at `a01eaf7` (v1, preserved as-is until the P9 release
  per SPEC.md — do not merge or push to `main` before then).
- `PLAN.md`: **P0-P6 all done and ticked. P7 (Metrics + reporting + matrix)
  is the next unchecked phase.**
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.
- Current test suite: **337 tests**, full `pytest tests/ -v` run takes
  **~5-8 minutes** (varies) — the registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`) is the expensive part (54
  cases: 9 methods × 6 scenarios, each a real short training run). Budget
  for this when working — don't assume the suite is fast.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`, implement the *first unchecked* phase only (P7 next).
2. Read that phase's `SPEC.md` section(s) first.
3. **Work in plan mode first** — research, then present a plan via
   `ExitPlanMode` before writing code. Every phase so far has had at least
   one real scope/design judgment call worth surfacing to the user before
   implementing.
4. Implement, then verify the phase's DONE criteria actually pass locally
   (`ruff check clover tests`, `mypy clover/core clover/config`, `pytest
   tests/ -v`).
5. Tick the box in `PLAN.md`, log any deviations under `PLAN.md`'s `## Log`
   section (dated), commit on `v2`. **Never** add an AI-attribution
   watermark to commit messages (no "Co-Authored-By: Claude", no
   "Generated with Claude Code" line) — this was an explicit, standing
   instruction from the user.
6. Don't push unless the user explicitly asks (they did, once, at the end
   of the P5 session — that's why `v2` is on GitHub through P5).
7. Large phases can span multiple sessions — deliver in sub-passes (P5 did
   L2P-then-DualPrompt/CODA; P6 did each of the 5 methods separately),
   logging progress in `PLAN.md`'s Log each time, ticking the phase's box
   only once *all* of its DONE criteria are met.

## What's done (P0-P6) — one line each

- **P0 — Scaffold**: package layout, decorator registries
  (methods/datasets/scenarios/backbones), pyproject, CI skeleton.
- **P1 — Stream model**: `StreamSpec`/`RevisitSpec`, `planner.resolve` →
  `StreamPlan`, `assignment.assign_images`, `Experience`/`Stream`/
  `Benchmark`, the 6 core scenario factories.
- **P2 — Label space + loss policies**: `LabelSpaceView` (set-membership
  masks), `methods/losses.py` (`new_class_ce`/`seen_class_ce`/
  `masked_logits`/`angular_margin_ce`), synthetic dataset + `tiny_mlp`
  backbone, the revisit-safety gate (proves the v1 masked-loss bug is
  structurally impossible here).
- **P3 — CLMethod + Trainer + SimpleCIL**: `CLMethod` ABC, `TrainContext`,
  `IncrementalHead`, `Trainer` (seeding/checkpoint-resume/status.json),
  `PerClassEvaluator`/`RMatrix`, SimpleCIL (frozen backbone + cosine
  prototypes, no gradient step).
- **P4 — Config + CLI**: typed config schema (`clover/config/`), `clover
  run/smoke/inspect/preflight`, `scripts/slurm_run.sh`.
- **P5 — Backbones + prompt trio**: backbone registry (`resolve_base_model`:
  registry-first, timm fallback, user-class source), `TinyViT` (clean-room,
  with a `prefix_kv` attention hook), L2P (token-prepend prompt pool),
  DualPrompt (prefix-KV general+expert prompts), CODA-Prompt (prefix-KV,
  soft attention-weighted pool + Gram-Schmidt orthogonalization). All three
  share `methods/prompt_common.py:PromptMethodBase`.
- **P6 — All 5 remaining methods, adapter family**: new `adapter` hook
  (parallel branch off the pre-MLP residual stream, `clover/backbones/
  vit.py`/`adapter.py:Adapter`+`AdapterViT`), distinct from P5's `prefix_kv`.
  - **APER-Adapter** + **RanPAC** share `clover/methods/adapter_common.py:
    AdapterMethodBase` ("train adapter once at `exp.task_label == 0`,
    freeze forever, closed-form head every experience after"). APER-Adapter:
    dual-branch cosine-prototype concat (one frozen base ViT read twice —
    plain + adapter-tuned — no second checkpoint). RanPAC:
    `RandomProjectionRidgeHead` (`relu(x @ W_rand)` + accumulated `G`/`Q` +
    grid-searched ridge, re-solved every experience).
  - **EASE** (`adapter_ease.py:EaseAdapterViT` + `heads.py:EaseHead`):
    growing per-experience adapter list (only the most recent trainable),
    growing-dim cosine head with per-class "home block" + `alpha`-scaled
    cross-block reweighting.
  - **MOS** (`adapter_mos.py:MOSAdapterViT`): ONE continuously-trained
    adapter, EMA-blended every optimizer step toward the mean of its own
    frozen history. New shared `classifier_alignment.py` (Gaussian-resample
    CA), reused by TUNA.
  - **TUNA** (`adapter_tuna.py:TunaAdapterViT`): growing per-experience
    adapter list (like EASE) combined via **EMR-merge** (sign-consensus,
    max-magnitude, rescaled) into one consensus adapter used directly for
    evaluation. Reuses `IncrementalHead(cosine=True)` + new
    `angular_margin_ce` loss (not a separate per-task-heads class — PILOT's
    `TunaLinear` design was an implementation detail the framework's
    existing masking already subsumes).
  - All 5 pass the registry-driven safety gate + `clover smoke` at the
    *existing* tuned hyperparameters (`epochs=40, optimizer_lr=3e-2`), no
    retuning needed anywhere in this phase. `docs/methods.md` now covers
    all 9 methods with hyperparameter-provenance tables.

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (every phase from P3
  onward, including all 5 P6 methods) is deferred every time it comes up —
  this machine has no GPU/Snellius access, and CIFAR-100 doesn't exist as a
  dataset yet (P7/P8). The user will need to run these on the actual
  cluster once real datasets land.
- **Per-method typed config schemas** (SPEC's "each method declares a typed
  schema") — still deferred; no method-level config knob has needed
  validation badly enough yet to justify the machinery.
- **CODA-Prompt's gradient masking**, **L2P's auxiliary key-pulling loss**:
  omitted, logged as scope simplifications (P5).
- **No dropout in `Adapter`** (P6, all 5 methods): `Trainer.run()`
  constructs `PerClassEvaluator` (calls `.eval()` on the shared classifier
  graph) once before the experience loop starts, so dropout would be
  permanently disabled framework-wide regardless of method — dead code, not
  a meaningful omission.
- **MOS's entropy-based test-time self-refinement** → replaced with a plain
  logit average over every stored adapter. **TUNA's per-sample entropy
  -based adapter routing** → replaced with using the EMR-merged adapter
  directly. Both agreed as scope simplifications before implementing; both
  methods' actual headline mechanisms (EMA merge; EMR merge) are
  implemented in full.
- **EASE's `beta`/`use_init_ptm`/`use_diagonal`/`moni_adam`**, **MOS's
  always-on orthogonality regularizer + two-param-group optimizer**,
  **TUNA's `use_orth`/`decay`/dead `r` hyperparameter** — none reimplemented
  (upstream defaults them off, or they're secondary regularizers/dead
  config not central to each method's headline mechanism). See
  `docs/methods.md`'s per-method hyperparameter tables for the full list.

## Judgment calls worth knowing about (for P7 and beyond)

- **`PromptMethodBase` (P5) doesn't fit the adapter family (P6) at all** —
  adapters need their own hook (parallel branch off the pre-MLP residual,
  not `prefix_kv`) and every adapter method's lifecycle differs enough from
  every other's (APER-Adapter/RanPAC train once then freeze; EASE grows a
  list; MOS keeps one continuously-trained adapter; TUNA grows a list but
  merges instead of concatenating) that only `AdapterMethodBase` (shared by
  exactly 2 of the 5) and `classifier_alignment.py` (shared by MOS/TUNA)
  turned out reusable — the rest are standalone `CLMethod`s. Lesson for
  future phases: don't assume a shared base class before building at least
  two methods that actually need the same lifecycle.
- **Any per-experience structural growth that must be reflected before
  `load_state_dict` runs belongs in `before_experience`, never
  `train_experience`/`after_experience`** — the Trainer's resume-replay
  only re-invokes `before_experience` for already-completed experiences.
  Bit EASE (`EaseAdapterViT.grow()`), MOS (`MOSAdapterViT.snapshot()`), and
  TUNA (`TunaAdapterViT.grow()`) all the same way; each was caught and
  fixed by reasoning through the resume-replay flow *before* writing the
  buggy version, not by a failing test. **Value-only recomputation** (a
  fixed-shape accumulator's contents changing, not a structure growing) is
  fine anywhere, including `train_experience` — TUNA's `recompute_merge()`
  and every method's prototype/CA updates rely on this distinction. If a
  future phase adds per-experience state, ask "does this change *shape* or
  just *values*?" before deciding where the update call belongs.
- **`torch.distributions.MultivariateNormal.sample()` draws from the global
  RNG and doesn't accept an explicit generator** — caught while designing
  MOS's classifier alignment, before it became a resume-safety bug (same
  class of issue as P3's DataLoader-RNG fix: CA runs inside
  `train_experience`, which resume-replay skips for completed experiences).
  `classifier_alignment.py` samples by hand instead (Cholesky decomposition
  + `torch.randn(..., generator=...)`), seeded from `exp.task_label`. Same
  pattern used by RanPAC's ridge-selection split. Keep this in mind for any
  future Gaussian/multivariate sampling in library code.
- **A method's `backbone.parameters()` is what the safety gate's finiteness
  check actually walks** (`for p in classifier().parameters(): assert
  isfinite`) — keep any closed-form-solved head weight as an `nn.Parameter`
  (RanPAC's ridge weight, e.g.), never a plain buffer, or the check
  silently skips it.
- **mypy's `clover/core clover/config` gate transitively pulls in whatever
  those modules import** — adding a new method/backbone means its type
  errors *will* surface here even though it's not directly named. Common
  P6 fix pattern: `nn.ModuleList` iteration yields plain `nn.Module` per
  mypy's stubs, not the concrete subclass — use `typing.cast` at the access
  site (see `adapter_ease.py`/`adapter_mos.py`/`adapter_tuna.py` for the
  exact pattern) rather than blanket `# type: ignore`.
- **The registry-driven safety gate's `optimizer_lr=3e-2`, `epochs=40`**
  (tuned originally for CODA-Prompt in P5) turned out to need **zero
  retuning** for any of the 9 registered methods — if a future method needs
  different values, re-verify the *whole* gate (all methods × all 6
  scenarios), not just the new one.
- **When debugging a low-accuracy result, root-cause it before tuning
  blindly**: check per-experience *training* accuracy and per-class
  confusion patterns before concluding undertrained vs. a real bug vs. an
  inherent property of the setup (e.g. echo-vs-source confusion for
  feature-based methods on `exact_replay`-style scenarios is expected, not
  a bug).
- **Never let per-experience randomness draw from the global torch RNG in
  `Trainer`** — `Trainer._build_loader` seeds each experience's `DataLoader`
  from `(run.seed, task_label)` specifically to avoid desyncing resumed
  runs; keep doing that for any new per-experience randomness (P3's
  original finding, reconfirmed relevant for MOS/RanPAC's own local
  generators in P6).

## Next up: P7 — Metrics + reporting + matrix (SPEC §10)

Per-class evaluator finalized, standard (A_t/AIA/BWT/FWT/Forgetting) +
CLOVER (RAG, Repetition Gain, Anchor/Long-Range Retention, echo-aware)
metrics ported with fixture tests; `clover report`; `run-matrix`
orchestrator (resume/retry/stale/GPU-pool).

DONE: hand-computed metric fixtures green; two-method synthetic demo report
renders in CI; matrix resume test green.

Suggested first step: read SPEC §10 closely, and read `clover/evaluation/`
(existing `PerClassEvaluator`/`RMatrix` from P3) to see what's already
there vs. what this phase adds. The bench's `analysis/clover_viz.py` and
the CLOVER-specific metrics (RAG, Repetition Gain, Anchor/Long-Range
Retention) are referenced in `AUDIT.md`/`SPEC.md` — worth a read-only look
at `clover-pilot-bench/analysis/` for the metric definitions actually used
in the published headline result (no code copied, same research-then-plan
pattern as P5/P6).

## Quick commands

```
cd D:\Dev\Research\clover-cl
.venv/Scripts/python.exe -m pytest tests/ -v                     # full suite, ~5-8 min
.venv/Scripts/python.exe -m ruff check clover tests
.venv/Scripts/python.exe -m mypy clover/core clover/config
.venv/Scripts/python.exe -m clover.cli smoke                     # all 9 registered methods
git checkout v2 && git pull                                      # resume from here
```

Delete this file once P7 is underway and this content is stale, or update
it in place at the end of each future session — whichever the next session
prefers.
