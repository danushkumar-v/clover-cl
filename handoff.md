# Handoff — clover-cl v2 (2026-07-07: P8 done, P9 next)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **P6, P7, and P8
  are committed locally but not yet pushed** — push only if the user asks.
- `main` is untouched (v1, preserved as-is until the P9 release per
  SPEC.md — do not merge or push to `main` before then). **v1 was a
  genuinely useful read-only reference for P8** — read via `git show
  main:<path>`, never checked out.
- `PLAN.md`: **P0-P8 all done and ticked. P9 (Release candidate) is next
  and last.**
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.
- Current test suite: **437 tests**. Full `pytest tests/ -v` run takes
  **~8-9 minutes** — the registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`, 54 cases: 9 methods × 6
  scenarios) and `tests/test_cli_run_matrix.py` (real subprocess spawns)
  are the expensive parts. Budget for this when working — don't assume
  the suite is fast.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`, implement the *first unchecked* phase only (P9 next).
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
7. Large phases span multiple sessions — deliver in sub-passes, logging
   progress in `PLAN.md`'s Log each time, ticking the phase's box only
   once *all* of its DONE criteria are met.

## What's done (P0-P8) — one line each

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
- **P8 — Datasets + extras + docs**: `clover/datasets/cifar100.py`
  (torchvision-backed, `download=True`) + `clover/datasets/
  image_folder_base.py:ImageFolderCLDataset` (shared base — v1 had 5
  structurally-identical `ImageFolder`-backed wrappers; extracted into one
  base since all 5 already existed and matched exactly) with 5 thin
  subclasses (CUB-200, ImageNet-R, ImageNet-A, OmniBenchmark, VTAB), plus
  `image_folder` — a config-only dataset (`dataset: {type: image_folder,
  root:, num_classes:}`) needing no new Python at all. New
  `distribution_shift` scenario (same-id revisit, once, end-of-stream);
  `symmetric_pair`/`near_miss`/`hierarchical` documented as out-of-scope
  in `docs/CONCEPTS.md` §8 (SPEC explicitly permits this). Two real,
  pre-existing gaps surfaced and fixed while proving `image_folder` works
  end-to-end (see Judgment calls below): `data_root` was never actually
  passed to a dataset constructor, and no code anywhere ever composed a
  dataset's declared `train_trsf`/`test_trsf`/`common_trsf` into an actual
  transform. `docs/CONCEPTS.md` rewritten for v2 (was still v1's
  `OverlapSpec` framing); new `docs/extending.md` with one worked example
  each for dataset/scenario/method/backbone registration. No real dataset
  downloads or network access used anywhere — CIFAR-100 tests monkeypatch
  `torchvision.datasets.CIFAR100`; the `ImageFolder`-based ones (including
  `image_folder` itself) use real tiny hand-built fixture directories.

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (every phase from P3
  onward) — no GPU/Snellius access this session. CIFAR-100 exists as a
  real dataset now (P8); still needs the user's own cluster access to
  actually run.
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
- **Real (non-synthetic) dataset + real backbone training, end to end**
  (found in P8, belongs to P9/cluster scope) — every registered backbone
  (`tiny_mlp`/`tiny_vit`) is a documented, synthetic-only stand-in
  (1-channel, sized for the 8x8 synthetic dataset); no method currently
  threads `in_chans` or a real timm base model through `build()`. The
  infrastructure exists (`clover/backbones/loader.py:resolve_base_model`,
  registry-first then timm fallback) but no method calls it yet — every
  method's `build()` calls `get_backbone(...)` directly instead. Fine for
  now (CLAUDE.md: no real training locally, ever — real runs are cluster
  jobs), but worth revisiting whether P9's full matrix run needs a method
  wired to a real timm ViT, or whether the existing toy backbones are
  accepted as permanently synthetic-only.

## Judgment calls worth knowing about

- **Don't assume a shared base class before confirming at least two things
  actually need the same lifecycle — but do extract one once several
  already-built examples turn out identical.** P6's 5 adapter methods only
  shared `AdapterMethodBase` between 2 of them (3 stayed standalone) — a
  case *against* premature abstraction. P8's 5 `ImageFolder`-backed
  datasets were the opposite: v1 already had all 5 built, and reading them
  confirmed they were structurally identical apart from 3 values — a case
  *for* extracting `ImageFolderCLDataset`, since the evidence (5 real,
  already-written examples) came before the abstraction, not the other
  way around. The rule isn't "never share code," it's "let confirmed
  duplication justify the abstraction, don't guess ahead of it."
- **This repo's own `v1`/`main` branch is a legitimate, sanctioned
  reference from P8 onward** (read via `git show main:<path>`, never
  checked out) — SPEC says `CLDataset` ABC "≈ v1" explicitly, so porting/
  adapting v1's own dataset/scenario/docs code is fundamentally different
  from the LAMDA-PILOT "cite, never copy" rule: it's evolving this same
  project's prior version, not reusing a third party's.
- **A "prove it end-to-end" DONE criterion doesn't require chasing every
  downstream layer if a boundary belongs to a different, already-shipped
  phase** (P8, `image_folder`'s literal SPEC acceptance criterion): the
  honest scope call was to run the real dataset+transform+benchmark
  pipeline all the way through a real `DataLoader` batch (proving the
  *dataset layer*, P8's actual scope, works), and stop at the
  pre-existing, documented backbone/channel-count gap (P5/P6's scope,
  already logged in P5's own log entry: "real cluster configs against an
  actual pretrained timm ViT would tune these separately") rather than
  quietly expanding scope to fix a different phase's gap under this one's
  banner. When a DONE criterion's literal wording would require touching
  another phase's territory, verify as far as the current phase's actual
  responsibility goes and say precisely where and why it stops.
- **A pipeline stage that's never been exercised for real can hide a
  totally silent gap indefinitely** (P8's biggest find): `CLDataset`
  subclasses declare `train_trsf`/`test_trsf`/`common_trsf`, but *nothing*
  in `cli.py`/`Trainer` ever composed them into an actual `transform` and
  applied it — v1's `DataManager` did this centrally
  (`transforms.Compose([*train_trsf, *common_trsf])`); v2 never got an
  equivalent. Invisible for 6 real datasets across P3-P8 because every
  dataset test either passed `transform=` manually or used `synthetic`
  (empty transform lists) — only surfaced when this session ran the
  *first* real (non-synthetic) dataset through a real `clover run`. Fixed
  with `cli.py:_apply_default_transforms(dataset)`. **Lesson: a "some day
  another dataset/method will exercise this" gap can sit latent through
  several phases' tests all passing — the only reliable check is actually
  running the full path for real, once, not just unit-testing each piece
  in isolation.** Same session also found `StreamSpec.data_root` was
  parsed/validated/serialized end-to-end but never passed to any dataset
  constructor — same root cause, same fix opportunity.
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
  never blanket `# type: ignore`. **P8 hit the same lesson with a
  third-party stub gap instead of an internal type error**: adding any
  dataset that imports `torchvision` surfaced `import-untyped` in this
  gate (`clover/config/loader.py` imports `clover.datasets`, pulling in
  every registered dataset module transitively) — fixed with a
  `[[tool.mypy.overrides]] module = ["torchvision.*"]
  ignore_missing_imports = true` block in `pyproject.toml` (torchvision
  ships no `py.typed` marker; there was nothing to fix on our side). Also
  hit a `ClassVar` subtlety: mypy rejects instance-attribute assignment to
  a name annotated `ClassVar` in a base class — `ImageFolderCLDataset
  ._num_classes` needed to support both class-level (5 named datasets)
  and instance-level (`image_folder`) assignment, so it's a plain `int`
  annotation, not `ClassVar[int]`.
- **A dataclass's `to_dict()` output must stay a valid `from_dict()` input
  for every field, or `config_resolved.yaml` round-tripping breaks** (P8):
  adding `StreamSpec.dataset_num_classes` and wiring it into `to_dict()`
  without also adding it to `StreamSection._ALLOWED_KEYS` broke
  re-resolving a saved resolved config — caught by the full test suite,
  not by the narrower schema-only tests. Any new `StreamSpec`/
  `StreamSection` field needs both directions checked together.
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

## Next up: P9 — Release candidate (SPEC §13-§14, last phase)

- `docs/MIGRATION.md` (legacy API → v2 configs) — check what's already
  there (`docs/MIGRATION.md` exists; likely needs a pass to reflect P6-P8
  additions rather than being written from scratch).
- README: quickstart, smoke→SLURM workflow, scenario table, results,
  citations + no-copied-code attribution note.
- Full 9-method × 6-scenario (or 7, now that `distribution_shift` exists —
  confirm with the user whether the matrix should include it) matrix run
  on the Snellius cluster — needs the user's own cluster access, not
  runnable from this machine.
- Tag RC; `main` preserved as `v1` branch (confirm exact mechanics with
  the user before touching `main` at all — SPEC says preserve it, don't
  guess the git operations that means).
- Worth resolving first: whether any deferred CIFAR-100-vs-published
  -accuracy sanity runs (P3/P5/P6) should happen as part of this phase's
  cluster work, since CIFAR-100 now exists as a real dataset (P8).

DONE (whole phase): full 9-method × 6-scenario matrix reproduced on
Snellius from shipped configs; README workflow verified end-to-end; `main`
preserved as `v1` branch.

## Quick commands

```
cd D:\Dev\Research\clover-cl
.venv/Scripts/python.exe -m pytest tests/ -v                     # full suite, ~8-9 min
.venv/Scripts/python.exe -m ruff check clover tests
.venv/Scripts/python.exe -m mypy clover/core clover/config
.venv/Scripts/python.exe -m clover.cli smoke                     # all 9 registered methods
git checkout v2 && git pull                                      # resume from here
```

Update this file in place at the end of each future session.
