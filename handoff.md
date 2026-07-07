# Handoff — clover-cl v2 (2026-07-07: P9 docs/configs done, P10 next)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **P6, P7, P8, and
  this P9 session are committed locally but not yet pushed** — push only
  if the user asks.
- `main` is untouched (v1, preserved as-is until release per SPEC.md — do
  not merge or push to `main` before then, and don't create a `v1` branch
  either; the user explicitly deferred all branch/tag mechanics this
  session — see below). **v1 is a legitimate, sanctioned read-only
  reference from P8 onward** — read via `git show main:<path>`, never
  checked out.
- `PLAN.md`: **P0-P8 ticked. P9 (Release candidate) is NOT ticked** — its
  docs/configs/small-fixes work is done, but its literal DONE criterion
  ("full matrix reproduced on Snellius") is genuinely unmet: it needs both
  **P10** (a new phase, added this session — see below) and the user's own
  cluster access. **P10 is the next phase to implement.**
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.
- Current test suite: **451 tests**. Full `pytest tests/ -v` run takes
  **~9 minutes** — the registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`, 54 cases: 9 methods × 6
  scenarios) and `tests/test_cli_run_matrix.py` (real subprocess spawns)
  are the expensive parts. Budget for this when working — don't assume
  the suite is fast.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`, implement the *first unchecked* phase only (P10 next —
   P9 stays unchecked until P10 lands and the user runs the actual cluster
   matrix; don't tick P9 in the meantime).
2. Read that phase's `SPEC.md` section(s) first (P10 isn't in SPEC.md — it
   was added this session; read `PLAN.md`'s P10 entry and its 2026-07-07
   log entry instead).
3. **Work in plan mode first** — research, then present a plan via
   `ExitPlanMode` before writing code. If a phase turns out to have a real
   scope-affecting discovery (like P10 itself, found while scoping P9),
   surface it to the user explicitly — don't silently absorb it into the
   current phase's scope or silently route around it.
4. Implement, then verify the phase's DONE criteria actually pass locally
   (`ruff check clover tests`, `mypy clover/core clover/config`, `pytest
   tests/ -v`, `clover smoke`).
5. Tick the box in `PLAN.md`, log deviations under `PLAN.md`'s `## Log`
   section (dated), commit on `v2`. **Never** add an AI-attribution
   watermark to commit messages (no "Co-Authored-By: Claude", no
   "Generated with Claude Code" line) — explicit standing instruction.
6. Don't push unless the user explicitly asks (they did, once, at the end
   of the P5 session — that's why `v2` is on GitHub through P5). Same for
   any branch/tag/merge operation touching `main` — confirm exact mechanics
   with the user before doing anything there; don't infer it from SPEC
   wording alone (this session's exact lesson).
7. Large phases span multiple sessions — deliver in sub-passes, logging
   progress in `PLAN.md`'s Log each time, ticking the phase's box only
   once *all* of its DONE criteria are met.

## What's done (P0-P8, plus P9's non-cluster work) — one line each

- **P0 — Scaffold**: package layout, decorator registries, pyproject, CI.
- **P1 — Stream model**: `StreamSpec`/planner/`Experience`/`Stream`/
  `Benchmark`, the 6 core scenario factories.
- **P2 — Label space + loss policies**: `LabelSpaceView`, `methods/
  losses.py`, synthetic dataset + `tiny_mlp`, the revisit-safety gate.
- **P3 — CLMethod + Trainer + SimpleCIL**: `CLMethod` ABC, `Trainer`
  (seeding/checkpoint-resume/status.json), `PerClassEvaluator`/`RMatrix`.
- **P4 — Config + CLI**: typed config schema, `clover run/smoke/inspect/
  preflight`, `scripts/slurm_run.sh`.
- **P5 — Backbones + prompt trio**: backbone registry, `TinyViT`
  (`prefix_kv` hook), L2P/DualPrompt/CODA-Prompt (`PromptMethodBase`).
- **P6 — Remaining 5 methods, adapter family**: `AdapterMethodBase`,
  EASE/MOS/TUNA's growing/EMA/EMR-merged adapters + `classifier_alignment
  .py`. All 9 methods pass the safety gate + `clover smoke`; `docs/
  methods.md` covers all of them.
- **P7 — Metrics + reporting + matrix**: `PerClassHistory`, standard +
  CLOVER-specific metrics, `clover report`, `clover run-matrix`
  (resume/retry/stale, sequential dispatch).
- **P8 — Datasets + extras + docs**: CIFAR-100 + 5 `ImageFolder`-backed
  built-ins (`ImageFolderCLDataset` shared base) + `image_folder`
  config-only dataset; `distribution_shift` scenario;
  `symmetric_pair`/`near_miss`/`hierarchical` documented as out-of-scope;
  `docs/CONCEPTS.md`/`docs/extending.md` written for v2.
- **P9 (this session, docs/configs/small-fixes only — checkbox NOT
  ticked)**:
  - **Two real, repo-wide gaps fixed** (both "this codebase never actually
    prepared to run on a GPU" symptoms, found together): `clover/cli.py
    :_build_run_config` hardcoded `device="cpu"` unconditionally — no
    config or flag could ever select a GPU, even on a real Snellius GPU
    node, silently invalidating the entire cluster-workflow premise. Now
    auto-detects `cuda` when available (smoke profile stays CPU-only
    always, per SPEC R7). SPEC §11's run-artifact layout (`manifest.json`,
    `log.txt`) was never actually written — `Benchmark.save_manifest`
    existed but nothing called it; `TrainContext.logger` was plumbed but
    never logged through. Both fixed; also added `cudnn.deterministic`
    on-by-default / `cudnn_benchmark` opt-in-and-recorded (SPEC §11),
    neither of which existed before.
  - **`configs/` directory created** (didn't exist before): 7 single-run
    example configs (one per scenario, on `cifar100`) + `configs/
    matrix_full.yaml` (9 methods × 7 scenarios × 3 seeds). Verified
    entirely offline via a new `tests/test_configs_shipped.py`
    (`clover preflight`/`clover inspect` + full matrix-cell resolution,
    using the existing `patched_cifar100` fixture — no real download).
  - **`README.md` and `docs/MIGRATION.md` rewritten wholesale** (both were
    entirely v1 content — `OverlapDataManager`, PILOT drop-in-compatibility
    claims, v0.2-era internal API — not what v2 actually is). New
    `CONTRIBUTING.md`.
  - **A real, repo-wide blocker was found while scoping this and
    deliberately NOT fixed here** — see "P10" below. Documented precisely
    in every shipped config's header comment, in `README.md`'s quickstart
    section, and as a new `PLAN.md` phase, per the user's explicit
    instruction ("mark about this issue, then complete the task, create
    new P10 for this fix").

## P10 — Backbone real-image support (new phase, added 2026-07-07)

**Not in SPEC.md's original phase table.** Found while scoping P9: no
method or backbone threads a channel count through at all. Every method's
`build()` calls `backbone_cls(input_size=stream_info.input_size)` only —
never a channel count — and both `TinyViT`/`TinyMLP` hard-default to 1
channel (sized for the 8x8 grayscale `synthetic` dataset).
`StreamInfo`/`DatasetInfo` have no field to even carry a channel count.
**This means a shipped config for any real (3-channel) dataset — CIFAR-100,
the 5 `ImageFolder` built-ins, `image_folder` — crashes on the first
training batch, on any hardware, cluster included.** P8 already hit exactly
this crash shape once (`image_folder` + SimpleCIL + `tiny_mlp`) and
reasoned it out of that phase's scope at the time; scoping P9 revealed it's
not `image_folder`-specific at all — it's a repo-wide gap across every
real dataset × every method.

`PLAN.md`'s P10 entry: `StreamInfo`/`DatasetInfo` gain a channel-count
field; `TinyViT`/`TinyMLP` and all 9 methods' `build()` thread it through;
at least one method wired to `clover/backbones/loader.py:resolve_base_model`
with a real timm base-model name (the config-selectable-backbone mechanism
SPEC §6.4 describes exists but no method actually calls it yet — every
method calls `get_backbone(...)` directly instead). Gate: a real dataset
trains one experience without a shape/channel error, on CPU, for at least
one method — no cluster needed for this gate, only real per-method
accuracy tuning stays cluster-side (same deferral pattern as every
dataset's CIFAR-100-vs-published-accuracy verification throughout this
project).

**Once P10 lands**, return to P9's outstanding piece: the user runs
`configs/matrix_full.yaml` on their own Snellius access
(`sbatch scripts/slurm_run.sh configs/matrix_full.yaml`), then `clover
report runs/matrix_full/*` — at that point P9's DONE criterion is
actually met and its checkbox can be ticked. Branch/tag mechanics (`v1`
branch creation, merging `v2` into `main`, RC tag) were explicitly deferred
by the user this session — confirm exact mechanics with them before doing
any of that, don't infer it from SPEC wording alone.

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (every phase from P3
  onward) — blocked on both P10 and the user's own cluster access now.
- **Per-method typed config schemas** — still deferred; confirmed again
  this session while writing `configs/`: no method's `build()` reads any
  `cfg.get(...)` key except `"backbone"`, so SPEC §9's own illustrative
  `prompt_pool_size`/`prompt_length` example keys are aspirational, not
  implemented. Shipped configs deliberately don't include fake keys.
- **`training.amp` is validated but still inert** — no autocast wrapping
  exists anywhere, even now that L2P/DualPrompt/CODA-Prompt are real
  gradient-trained methods (P5/P6). Shipped configs use `amp: none`, not
  `bf16`, to avoid implying a speedup that doesn't exist. Worth fixing
  before/alongside P10 if real cluster runs care about mixed-precision
  speed — same "never actually prepared for a GPU" family as P10 itself,
  but not folded into P10's scope (P10 is about *correctness* — real
  images training at all — not performance).
- **CODA-Prompt's gradient masking**, **L2P's auxiliary key-pulling loss**
  (P5); **no dropout in `Adapter`** (P6); **MOS's entropy self-refinement →
  plain logit average**, **TUNA's per-sample routing → EMR-merged adapter
  directly** (P6); **EASE's `beta`/`use_init_ptm`/`use_diagonal`, MOS's orth
  regularizer + two-param-group optimizer, TUNA's `use_orth`/`decay`/dead
  `r`** (P6) — none reimplemented, see `docs/methods.md`'s per-method tables.
- **`clover report`'s scenario-name grouping relies on the run directory
  naming convention**, not `config_resolved.yaml` (P7).
- **`run-matrix` GPU-pool parallelism** (P7) — sequential dispatch only.

## Judgment calls worth knowing about

- **When a DONE criterion's literal wording would require touching another
  phase's territory, verify as far as the current phase's actual
  responsibility goes and say precisely where and why it stops — but if
  the gap is big enough to block the *next* phase's entire premise, don't
  just document it and move on; ask the user how they want to scope the
  fix.** P8 found the backbone/channel gap for `image_folder` specifically
  and correctly scoped around it (dataset-layer verification, not
  training). P9 re-found the *same* gap while writing cluster-workflow
  docs and configs, this time as a repo-wide blocker on P9's entire
  premise — different enough in severity/scope to warrant asking rather
  than silently repeating P8's same call. The user's answer became P10.
- **A hardcoded value that was never wrong *yet* (because nothing ever
  exercised the alternative) can silently invalidate an entire documented
  workflow.** `device="cpu"` in `_build_run_config` was fine for every
  local CPU-only smoke test and every synthetic-dataset unit test ever
  written — it only became a bug the moment someone tried to document "and
  then submit to a real GPU cluster" as something that actually works.
  Same lesson as P8's transform-composition gap: a path nothing has ever
  really exercised can hide a real bug indefinitely; writing the docs that
  describe a workflow is often what finally exercises it.
- **Don't assume a git branch mentioned in `CLAUDE.md`/`SPEC.md` actually
  exists — check `git branch -a` before writing docs that reference it.**
  `CLAUDE.md` talks about "the `v1` branch" as if it already exists;
  it doesn't yet (only `main`, which will become `v1` at release). Both
  `README.md` and `docs/MIGRATION.md` initially said "on the `v1` branch"
  and were caught and fixed to say "currently on `main`, to become `v1` at
  release" before commit.
- **This repo's own `v1`/`main` branch is a legitimate, sanctioned
  reference from P8 onward** (read via `git show main:<path>`, never
  checked out) — SPEC says `CLDataset` ABC "≈ v1" explicitly, so
  porting/adapting v1's own dataset/scenario/docs code (including for
  `docs/MIGRATION.md`'s real `OverlapDataManager`/`OverlapSpec` mapping)
  is fundamentally different from the LAMDA-PILOT "cite, never copy" rule.
- **A pipeline stage that's never been exercised for real can hide a
  totally silent gap indefinitely** (P8's transform-composition gap,
  P9's device-hardcoding gap — same family). The only reliable check is
  actually running/documenting the full path for real, once, not just
  unit-testing each piece in isolation.
- **A dataclass's `to_dict()` output must stay a valid `from_dict()` input
  for every field, or `config_resolved.yaml` round-tripping breaks** (P8) —
  any new `StreamSpec`/`StreamSection`/`TrainingSection` field needs both
  directions checked together (this session's `cudnn_benchmark` field was
  added to both `_ALLOWED_KEYS` and `to_dict`/`from_dict` together from the
  start, having learned this from P8's `dataset_num_classes` near-miss).
- **Any per-experience *structural* growth that must be reflected before
  `load_state_dict` runs belongs in `before_experience`, never
  `train_experience`/`after_experience`** (P6: EASE/MOS/TUNA).
- **A per-run artifact written *before* its corresponding checkpoint can go
  stale on a crash** (P7): `per_task.csv` filters read-back rows to
  `task_idx < resume_from` on resume for this reason.
- **`torch.distributions.MultivariateNormal.sample()` draws from the
  global RNG** (P6) — `classifier_alignment.py` samples by hand instead.
- **A method's `backbone.parameters()` is what the safety gate's
  finiteness check actually walks** — closed-form-solved head weights must
  be `nn.Parameter`, never a plain buffer.
- **mypy's `clover/core clover/config` gate transitively pulls in whatever
  those modules import** — third-party stub gaps (torchvision, P8) and
  `ClassVar` subtleties (P8) both surfaced this way.
- **`config_resolved.yaml` doesn't carry the scenario *name*** (P7) —
  `clover report` parses it from the run directory name instead.
- **The registry-driven safety gate's `optimizer_lr=3e-2`, `epochs=40`**
  (tuned for CODA-Prompt, P5) needed zero retuning for any of the 9 methods.
- **Never let per-experience randomness draw from the global torch RNG in
  `Trainer`** — `_build_loader` seeds each experience's `DataLoader` from
  `(run.seed, task_label)`.
- **Global torch backend flags (`cudnn.deterministic`/`cudnn.benchmark`)
  mutated in a test need explicit cleanup** (this session,
  `test_trainer.py`) — `_seed_everything` sets them as a side effect on
  every `Trainer.run()`, so a test that sets `cudnn_benchmark=True` restores
  the default (`False`) in a `finally` block afterward, since these are
  process-global, not test-scoped.
- **Git-Bash-on-Windows path footgun, not a product bug**: a bare
  `/tmp/...`-style path can silently diverge from what Python resolves.
  Use a relative or drive-lettered path when manually testing CLI commands.

## Quick commands

```
cd D:\Dev\Research\clover-cl
.venv/Scripts/python.exe -m pytest tests/ -v                     # full suite, ~9 min
.venv/Scripts/python.exe -m ruff check clover tests
.venv/Scripts/python.exe -m mypy clover/core clover/config
.venv/Scripts/python.exe -m clover.cli smoke                     # all 9 registered methods
.venv/Scripts/python.exe -m clover.cli preflight configs/<name>.yaml
git checkout v2 && git pull                                      # resume from here
```

Update this file in place at the end of each future session.
