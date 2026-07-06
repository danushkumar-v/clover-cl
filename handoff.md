# Handoff — clover-cl v2 (2026-07-06: P7 complete)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **P6 (4 commits)
  and P7 (3 commits) are committed locally but not yet pushed** — push
  only if the user asks.
- `main` is untouched at `a01eaf7` (v1, preserved as-is until the P9 release
  per SPEC.md — do not merge or push to `main` before then).
- `PLAN.md`: **P0-P7 all done and ticked. P8 (Datasets + extras + docs) is
  the next unchecked phase.**
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.
- Current test suite: **387 tests**. Full `pytest tests/ -v` run takes
  **~5-10 minutes** (varies) — the registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`, 54 cases: 9 methods × 6
  scenarios) and `tests/test_cli_run_matrix.py` (real subprocess spawns)
  are the expensive parts. Budget for this when working — don't assume
  the suite is fast.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`, implement the *first unchecked* phase only (P8 next).
2. Read that phase's `SPEC.md` section(s) first.
3. **Work in plan mode first** — research (a background Agent for
   reference-repo research has worked well for P5/P6/P7), then present a
   plan via `ExitPlanMode` before writing code. Every phase so far has had
   at least one real scope/design judgment call worth surfacing before
   implementing.
4. Implement, then verify the phase's DONE criteria actually pass locally
   (`ruff check clover tests`, `mypy clover/core clover/config`, `pytest
   tests/ -v`, `clover smoke`).
5. Tick the box in `PLAN.md`, log deviations under `PLAN.md`'s `## Log`
   section (dated), commit on `v2`. **Never** add an AI-attribution
   watermark to commit messages (no "Co-Authored-By: Claude", no
   "Generated with Claude Code" line) — explicit standing instruction.
6. Don't push unless the user explicitly asks (they did, once, at the end
   of the P5 session — that's why `v2` is on GitHub through P5).
7. Large phases span multiple sessions — deliver in sub-passes (P5: one
   method then two more; P6: one method per session; P7: metrics library,
   then report, then run-matrix), logging progress in `PLAN.md`'s Log each
   time, ticking the phase's box only once *all* of its DONE criteria are
   met.

## What's done (P0-P7) — one line each

- **P0 — Scaffold**: package layout, decorator registries, pyproject, CI.
- **P1 — Stream model**: `StreamSpec`/planner/`Experience`/`Stream`/
  `Benchmark`, the 6 core scenario factories.
- **P2 — Label space + loss policies**: `LabelSpaceView`, `methods/
  losses.py` (`new_class_ce`/`seen_class_ce`/`masked_logits`/
  `angular_margin_ce`), synthetic dataset + `tiny_mlp`, the revisit-safety
  gate.
- **P3 — CLMethod + Trainer + SimpleCIL**: `CLMethod` ABC, `Trainer`
  (seeding/checkpoint-resume/status.json), `PerClassEvaluator`/`RMatrix`.
- **P4 — Config + CLI**: typed config schema, `clover run/smoke/inspect/
  preflight`, `scripts/slurm_run.sh`.
- **P5 — Backbones + prompt trio**: backbone registry, `TinyViT`
  (`prefix_kv` hook), L2P/DualPrompt/CODA-Prompt (`PromptMethodBase`).
- **P6 — Remaining 5 methods, adapter family**: new `adapter` hook
  (parallel branch off the pre-MLP residual). APER-Adapter+RanPAC share
  `AdapterMethodBase` (train once, freeze, closed-form head). EASE: growing
  adapter list + growing-dim cosine head. MOS: one continuously-EMA
  -blended adapter + `classifier_alignment.py` (Gaussian-resample CA,
  reused by TUNA). TUNA: growing adapter list combined via EMR-merge into
  one consensus adapter. All 9 methods pass the safety gate + `clover
  smoke` at the existing tuned hyperparameters; `docs/methods.md` covers
  all of them with hyperparameter provenance.
- **P7 — Metrics + reporting + matrix**: `clover/evaluation/history.py:
  PerClassHistory` persists per-class accuracy the Trainer already
  computed but previously discarded. `clover/evaluation/metrics/{standard,
  overlap}.py`: 5 standard metrics (pure R-matrix reductions) + echo-aware
  CLOVER metrics (RAG, Repetition Gain, Anchor/Long-Range Retention).
  Trainer writes `runs/<name>/per_task.csv` every experience.
  `clover/reporting.py` + `clover report`: rebuild-from-scratch aggregate
  CSV + summary pivot, no pandas. `clover/matrix.py` + `clover run-matrix`:
  grid enumeration, `status.json`-based resume/retry/stale detection,
  subprocess dispatch (crash isolation) — sequential only, GPU-pool
  parallelism deferred (no GPU to test against). See `PLAN.md`'s three
  2026-07-06 P7 log entries for full design detail.

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (every phase from P3
  onward) — no GPU/Snellius access this session, CIFAR-100 doesn't exist
  as a dataset yet (P8). User runs these on the cluster once real datasets
  land.
- **Per-method typed config schemas** — still deferred; no method-level
  config knob has needed validation badly enough yet to justify it.
- **CODA-Prompt's gradient masking**, **L2P's auxiliary key-pulling loss**
  (P5); **no dropout in `Adapter`** (P6, dead code given how
  `PerClassEvaluator` is constructed — see judgment calls); **MOS's
  entropy self-refinement → plain logit average**, **TUNA's per-sample
  routing → EMR-merged adapter directly** (P6); **EASE's `beta`/
  `use_init_ptm`/`use_diagonal`, MOS's orth regularizer + two-param-group
  optimizer, TUNA's `use_orth`/`decay`/dead `r`** (P6) — none reimplemented,
  see `docs/methods.md`'s per-method tables.
- **`clover report`'s scenario-name grouping relies on the run directory
  naming convention**, not `config_resolved.yaml` (P7) — see judgment
  calls below.
- **`run-matrix` GPU-pool parallelism** (P7) — sequential dispatch only;
  a real feature but untestable/inoperative on this GPU-less machine
  either way. Adding it later is a pure enhancement (swap the dispatch
  loop for a thread pool), not a redesign.

## Judgment calls worth knowing about (for P8 and beyond)

- **Don't assume a shared base class before building at least two things
  that actually need the same lifecycle** — P6's 5 adapter methods only
  shared `AdapterMethodBase` between 2 of them; the other 3 were
  standalone. Applies equally to P8's dataset wrappers (CUB-200/ImageNet-R/
  ImageNet-A/OmniBenchmark/VTAB) — don't force a common base until you've
  actually built two and confirmed they overlap.
- **Any per-experience *structural* growth that must be reflected before
  `load_state_dict` runs belongs in `before_experience`, never
  `train_experience`/`after_experience`** — the Trainer's resume-replay
  only re-invokes `before_experience` for already-completed experiences
  (P6: EASE/MOS/TUNA all hit this). **Value-only recomputation** (a
  fixed-shape accumulator's contents changing, not a structure growing) is
  fine anywhere, including `train_experience` (P7: `per_task.csv`'s
  metric values, TUNA's `recompute_merge()`). Ask "does this change
  *shape* or just *values*?" before deciding where an update call belongs.
- **A per-run artifact written *before* its corresponding checkpoint can go
  stale on a crash** (P7): `per_task.csv` is written before
  `ckpt_task*.pt` each round, so a crash between the two writes leaves a
  row for a task the checkpoint doesn't yet confirm as done. Fixed by
  filtering read-back rows to `task_idx < resume_from` on resume — only
  what the checkpoint confirms is trusted. Check for this pattern with any
  new pre-checkpoint artifact write.
- **Recomputing a historical value from today's fuller state can leak
  future information into it** (P7): some CLOVER metrics (`RAG_mean`)
  aggregate over the *whole* run's history rather than being strictly
  task-scoped; `per_task.csv` rows are read back verbatim on resume, never
  recomputed, to avoid an earlier task's row silently reflecting a revisit
  that hadn't happened yet when it was first written.
- **`torch.distributions.MultivariateNormal.sample()` draws from the
  global RNG and doesn't accept an explicit generator** (P6) — same class
  of bug as P3's DataLoader-RNG fix (anything inside `train_experience`
  that resume-replay skips must never touch the global RNG).
  `classifier_alignment.py` samples by hand (Cholesky + seeded
  `torch.randn`) instead. Keep this in mind for any future
  Gaussian/multivariate sampling in library code.
- **A method's `backbone.parameters()` is what the safety gate's
  finiteness check actually walks** — keep any closed-form-solved head
  weight as an `nn.Parameter` (RanPAC's ridge weight, e.g.), never a plain
  buffer, or the check silently skips it.
- **mypy's `clover/core clover/config` gate transitively pulls in whatever
  those modules import** — adding anything reachable from there means its
  type errors *will* surface in this gate. `nn.ModuleList` iteration
  yields plain `nn.Module` per mypy's stubs — use `typing.cast` at the
  access site (see `adapter_ease.py`/`adapter_mos.py`/`adapter_tuna.py`),
  never blanket `# type: ignore`.
- **`config_resolved.yaml` doesn't carry the scenario *name*** (found
  while designing `clover report`, P7) — `ResolvedConfig.stream_spec` is
  post-resolution, so only the concrete `revisits` are recorded. Rather
  than extend tested P4 code for a P7-only need, `clover report` parses
  `{dataset}__{method}__{scenario}__seed{seed}` from the run *directory
  name* instead (matching the bench's own convention exactly);
  `run-matrix` names every dir it creates this way by construction. An
  ad-hoc `clover run` whose dir name doesn't parse this way still gets its
  rows aggregated, just grouped as `"unknown"` in the summary pivot.
- **The registry-driven safety gate's `optimizer_lr=3e-2`, `epochs=40`**
  (tuned for CODA-Prompt in P5) needed **zero retuning** for any of the 9
  methods — if a future method needs different values, re-verify the
  *whole* gate, not just the new one.
- **Never let per-experience randomness draw from the global torch RNG in
  `Trainer`** — `Trainer._build_loader` seeds each experience's
  `DataLoader` from `(run.seed, task_label)` specifically to avoid
  desyncing resumed runs; keep doing that for any new per-experience
  randomness.
- **Git-Bash-on-Windows path footgun, not a product bug**: testing
  anything through Git Bash with a bare `/tmp/...`-style path can silently
  diverge from what Python resolves (Git Bash's `/tmp` and Windows
  Python's `/tmp`, i.e. `D:\tmp` for a D: cwd, are *different
  directories*). Use a relative or drive-lettered path when manually
  testing CLI commands on this machine.

## Next up: P8 — Datasets + extras + docs (SPEC §7, §8, §13)

CUB-200, ImageNet-R/A, OmniBenchmark, VTAB dataset wrappers + staging
docs; `image_folder` config-only dataset; scenario extras
(`hierarchical`/`near_miss`/`distribution_shift`/`symmetric_pair`, ported
opportunistically per SPEC — their absence must not block v2.0);
`docs/concepts.md`, `docs/extending.md` (method/dataset/scenario/backbone
worked examples).

DONE: built-in metadata smoke tests green; a tutorial-followed custom
dataset runs a full smoke benchmark without touching core.

Suggested first step: read SPEC §7-§8 and `clover/datasets/base.py` +
`clover/datasets/synthetic.py` (the only existing dataset, P2) to see the
`CLDataset` ABC shape this phase extends. The 5 new datasets need real
download/staging (CIFAR-100 itself is still not built either — check
whether it's in scope here or genuinely P8-adjacent per SPEC's exact
wording before assuming). This phase is the first one that will actually
need to *download* real image data — confirm with the user before
attempting any real download on their behalf, and remember this machine's
CPU-only / no-GPU constraint doesn't block dataset *staging* work, just
real training on it.

## Quick commands

```
cd D:\Dev\Research\clover-cl
.venv/Scripts/python.exe -m pytest tests/ -v                     # full suite, ~5-10 min
.venv/Scripts/python.exe -m ruff check clover tests
.venv/Scripts/python.exe -m mypy clover/core clover/config
.venv/Scripts/python.exe -m clover.cli smoke                     # all 9 registered methods
git checkout v2 && git pull                                      # resume from here
```

Delete this file once P8 is underway and this content is stale, or update
it in place at the end of each future session — whichever the next
session prefers.
