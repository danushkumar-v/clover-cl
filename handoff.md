# Handoff — clover-cl v2 (2026-07-06, updated same day: P6 in progress)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **This session's
  P6 work (APER-Adapter + RanPAC) is committed locally but not yet
  pushed** — push only if the user asks.
- `main` is untouched at `a01eaf7` (v1, preserved as-is until the P9 release
  per SPEC.md — do not merge or push to `main` before then).
- `PLAN.md`: **P0-P5 done and ticked. P6 is in progress (not ticked) —
  APER-Adapter and RanPAC are done; EASE, MOS, TUNA are queued.**
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`, implement the *first unchecked* phase only.
2. Read that phase's `SPEC.md` section(s) first.
3. **Work in plan mode first** — research, then present a plan via
   `ExitPlanMode` before writing code. Every phase so far has had at least
   one real scope/design judgment call worth surfacing to the user before
   implementing (see "Judgment calls" below for the pattern).
4. Implement, then verify the phase's DONE criteria actually pass locally
   (`ruff check clover tests`, `mypy clover/core clover/config`, `pytest
   tests/ -v`).
5. Tick the box in `PLAN.md`, log any deviations under `PLAN.md`'s `## Log`
   section (dated), commit on `v2`. **Never** add an AI-attribution
   watermark to commit messages (no "Co-Authored-By: Claude", no
   "Generated with Claude Code" line) — this was an explicit, standing
   instruction from the user.
6. Don't push unless the user explicitly asks (they did, once, at the end
   of the P5 session — that's why `v2` is on GitHub now).

## What's done (P0-P5) — one line each

- **P0 — Scaffold**: package layout, decorator registries
  (methods/datasets/scenarios/backbones), pyproject, CI skeleton.
- **P1 — Stream model**: `StreamSpec`/`RevisitSpec`, `planner.resolve` →
  `StreamPlan`, `assignment.assign_images`, `Experience`/`Stream`/
  `Benchmark`, the 6 core scenario factories.
- **P2 — Label space + loss policies**: `LabelSpaceView` (set-membership
  masks), `methods/losses.py` (`new_class_ce`/`seen_class_ce`/
  `masked_logits`), synthetic dataset + `tiny_mlp` backbone, the
  revisit-safety gate (proves the v1 masked-loss bug is structurally
  impossible here).
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
- **P6 (in progress) — APER-Adapter + RanPAC done**: new adapter hook
  (`adapter: Optional[Dict[int, nn.Module]]`, parallel branch off the
  pre-MLP residual stream) threaded through `TransformerBlock`/`TinyViT`
  alongside `prefix_kv`; `clover/backbones/adapter.py` (`Adapter` +
  `AdapterViT`/`vit_adapter`); `clover/methods/adapter_common.py:
  AdapterMethodBase` (shared "train adapter once at `exp.task_label == 0`,
  freeze forever, closed-form head every experience after" lifecycle);
  APER-Adapter (dual-branch cosine-prototype concat, reusing one frozen
  base ViT for both the plain and adapter-tuned pass — no second checkpoint
  needed); RanPAC (`RandomProjectionRidgeHead` in `methods/heads.py`:
  `relu(x @ W_rand)` + accumulated `G`/`Q` sufficient statistics + grid
  -searched ridge, re-solved every experience). EASE, MOS, TUNA are next —
  see PLAN.md's P6 Log entry and "Next up" below for the full per-method
  design already agreed with the user.

Current test suite: **273 tests**, full `pytest tests/ -v` run takes
**~2.5-3 minutes** — the registry-driven safety gate
(`tests/test_method_registry_safety_gate.py`) is the expensive part (36
cases: 6 methods × 6 scenarios, each a real short training run). Budget for
this when working — don't assume the suite is fast. Both new P6 methods
passed the gate at the *existing* tuned hyperparameters (`epochs=40,
optimizer_lr=3e-2`) with no retuning needed.

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (P3, P4's actual sbatch
  submission, P5) is deferred every time it comes up — this machine has no
  GPU/Snellius access, and CIFAR-100 doesn't exist as a dataset yet (P7/P8).
  The user will need to run these on the actual cluster once real datasets
  land.
- **Per-method typed config schemas** (SPEC's "each method declares a typed
  schema") — deferred since no method has real hyperparameters worth
  validating yet. Revisit once P6's adapter methods add real config knobs.
- **CODA-Prompt's gradient masking** (freezing earlier tasks' pool
  components during later training) is explicitly omitted — logged as a
  scope simplification, not a bug.
- **L2P's auxiliary key-pulling loss** is omitted for the same reason.
- **No dropout in the new `Adapter` module** (P6): PILOT's adapter has one;
  omitted here because `Trainer.run()` constructs `PerClassEvaluator`
  (which calls `.eval()` on the shared classifier module graph) once,
  before the experience loop starts, so dropout would be permanently
  disabled framework-wide regardless of method — dead code, not a
  meaningful omission.
- **MOS's iterative entropy-based test-time adapter self-refinement** and
  **TUNA's per-sample entropy-based adapter routing** are planned scope
  simplifications for when those two land (not yet implemented) — both are
  secondary inference-time tricks layered on top of each method's actual
  headline mechanism (MOS's continuous EMA merge; TUNA's EMR merge), which
  *will* be implemented in full. Agreed with the user in this session's
  plan-mode approval before any EASE/MOS/TUNA code was written.

## Judgment calls worth knowing about (so P6 doesn't re-litigate them)

- **CLDataset/backbone/method pulled forward from later phases when SPEC
  frames them as reusable built-ins, not throwaway fixtures** — e.g. the
  `synthetic` dataset (P7/P8's phase) and `tiny_mlp`/`tiny_vit` backbones
  (P5) were built early in P2/P5 because SPEC explicitly calls them
  reusable across the smoke profile and safety gate. When P6 needs
  something similar, look at how P2/P5 justified it before assuming you
  need to wait for the "official" phase.
- **Registries only hold complete backbones or wrapper mechanisms, never
  bare base models** — base-model resolution (timm name / user class) is
  a separate concern (`clover/backbones/loader.py:resolve_base_model`).
- **`clover/methods/__init__.py` and `clover/backbones/__init__.py` both
  import their concrete modules at the *end* of the file**, after the
  `register_x`/`get_x` bindings — new methods/backbones need the same
  pattern (see any existing module for the exact shape) or you'll hit a
  circular-import error.
- **`PromptMethodBase` (`clover/methods/prompt_common.py`) turned out NOT to
  fit the adapter family** — confirmed while building P6: adapters need
  their own hook (parallel branch off the pre-MLP residual, not `prefix_kv`
  key/value splice) and a different lifecycle (APER-Adapter/RanPAC train
  only once then rely on a closed-form head; EASE/MOS/TUNA keep training a
  growing per-task adapter list every experience — neither matches
  `PromptMethodBase`'s "trainable prompt + gradient-trained head every
  experience" shape). New base: `clover/methods/adapter_common.py:
  AdapterMethodBase`, used only by APER-Adapter/RanPAC; EASE/MOS/TUNA will
  each need their own method module (not this shared base) since their
  per-task-list/merge bookkeeping differs enough from each other too.
- **mypy's `clover/core clover/config` gate transitively pulls in whatever
  those modules import** — `clover/config` imports from `clover/training`,
  which imports `clover/methods`, which imports every registered method
  and backbone. Adding a new method/backbone means its type errors *will*
  surface in this gate even though it's not directly named — expect this,
  fix real gaps with precise `# type: ignore[<code>]` only where the
  attribute is genuinely dynamic (duck-typed backbones), never blanket-
  ignore.
- **Never let per-experience randomness draw from the global torch RNG in
  `Trainer`** — a resumed run's `before_experience` replay (no data
  touched) advances the global RNG differently than an uninterrupted run's
  actual training (which does touch data) at the same point, breaking
  bit-identical resume. `Trainer._build_loader` seeds each experience's
  `DataLoader` from `(run.seed, task_label)` specifically to avoid this —
  keep doing that for any new per-experience randomness.
- **When debugging a low-accuracy result, root-cause it before tuning
  blindly**: check per-experience *training* accuracy (not just held-out
  test) and per-class confusion patterns before concluding something is
  undertrained vs. a real bug vs. an inherent property of the setup (e.g.
  echo classes sharing a source's exact visual pattern make some
  echo-vs-source confusion genuinely unavoidable for feature-based
  methods — this is expected, not a bug, and shows up for SimpleCIL, L2P,
  and CODA-Prompt alike on `exact_replay`-style scenarios).
- **The registry-driven safety gate uses `optimizer_lr=3e-2`, `epochs=40`**
  for gradient-trained methods (tuned specifically so CODA-Prompt reliably
  clears chance on echo-style scenarios; verified this doesn't regress any
  other method/scenario combination). **Confirmed still fine for
  APER-Adapter/RanPAC** — both passed all 6 scenarios on the first attempt
  with zero retuning. If EASE/MOS/TUNA need different tuning, re-verify the
  *whole* gate (all methods × all 6 scenarios), not just the new method —
  hyperparameter changes here are global.
- **A method's backbone.parameters() is what the safety gate's finiteness
  check actually walks** (`for p in classifier().parameters(): assert
  isfinite`) — RanPAC's closed-form ridge weight is kept as an
  `nn.Parameter` (never backprop'd, assigned via `solve()`) specifically so
  this check still exercises it; a plain buffer would silently skip the
  check. Keep any future closed-form-solved head weight as a Parameter for
  the same reason.

## Next up: P6 continued — EASE, MOS, TUNA (SPEC §6.5)

APER-Adapter and RanPAC are done (this session). Full per-method design was
agreed with the user via plan-mode approval before any of this session's
code was written — the design still holds for the next 3:

- **EASE** (`clover/backbones/adapter_ease.py` + `clover/methods/ease.py`):
  growing per-experience frozen adapter list (`add_adapter` in
  `after_experience`); new `EaseHead` (`clover/methods/heads.py`) whose
  feature-dim grows every experience (concat of every adapter's `[CLS]`
  features) — new-block class rows via prototype means, cross-block rows
  via cosine-similarity interpolation from other blocks. Per-experience
  training uses a small "proxy" head over just the current adapter +
  `new_class_ce` (P2's loss policy already makes EASE's real
  `aux_targets`/`ignore_index=-1` patch unnecessary).
- **MOS** (`clover/backbones/adapter_mos.py` + `clover/methods/mos.py`):
  per-experience adapter gradient-trained with `new_class_ce`, continuously
  EMA-blended toward the running mean of previous experiences' adapters
  during training (`adapter_momentum`) — implement this merge in full, it's
  MOS's headline mechanism. Fixed-width `IncrementalHead(cosine=True)`
  (unlike EASE, no growing dim). New shared
  `clover/methods/classifier_alignment.py` (`gaussian_resample_finetune`):
  per-class mean/covariance, synthetic-feature CE fine-tuning for
  `crct_epochs` after each experience past the first — reused by TUNA.
  **Simplify out**: the iterative entropy-based test-time adapter
  self-refinement search (secondary inference trick, not the merge itself)
  — replace with a plain average over all adapters' predictions.
- **TUNA** (`clover/backbones/adapter_tuna.py` + `clover/methods/tuna.py`):
  per-experience adapter trained with a small CosFace/angular-margin loss
  (new `angular_margin_ce` helper in `clover/methods/losses.py`, still
  built on set-membership masks), then **EMR-merge** (sign-consensus,
  max-magnitude-among-agreeing-signs, rescaled) across all stored per-task
  adapters — implement in full, it's TUNA's defining mechanism and a
  compact closed-form op, not an iterative search. Per-task `nn.Linear`
  heads (no bias, cosine input) concatenated, unfrozen. Reuses
  `classifier_alignment.py`. **Simplify out**: per-sample entropy-based
  adapter routing at test time — use the EMR-merged adapter directly.

DONE (whole phase, all 5 methods): each passes the safety gate + smoke + a
disjoint CIFAR-100 sanity run vs. published range (CIFAR-100 leg deferred,
same pattern as P3/P4/P5 — implement and verify everything locally
checkable, log the cluster-dependent piece as deferred); comparison table
in `docs/methods.md` covers all 9 methods (currently 6 of 9 rows filled in).

Read-only research on the exact PILOT mechanism (files, line numbers,
hyperparameters, known bugs) for all 5 adapter-family methods was already
done this session — check `PLAN.md`'s 2026-07-06 P6 log entry and this
file's `docs/methods.md` before re-deriving it from scratch.

## Quick commands

```
cd D:\Dev\Research\clover-cl
.venv/Scripts/python.exe -m pytest tests/ -v                     # full suite, ~2.5-3 min
.venv/Scripts/python.exe -m ruff check clover tests
.venv/Scripts/python.exe -m mypy clover/core clover/config
.venv/Scripts/python.exe -m clover.cli smoke                     # all registered methods
git checkout v2 && git pull                                      # resume from here
```

Delete this file once P6 is underway and its content is stale, or update
it in place at the end of each future session — whichever the next
session prefers.
