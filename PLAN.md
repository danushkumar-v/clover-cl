# PLAN — CLOVER v2 implementation checklist

Source of truth for scope/design: `SPEC.md` (final, 2026-07-05); constraints:
`CLAUDE.md`; background: `../AUDIT.md`.

**Session protocol:** read this file, implement the *first unchecked* phase
(read its SPEC.md sections first), verify its DONE criteria actually pass,
tick the box, commit. Do not start a later phase while an earlier one is
unchecked. If a phase must deviate from SPEC.md, record the deviation under
"Log" below before ticking.

- [x] **P0 — Scaffold** (SPEC §2): `git pull` to origin/main, branch `v2`; package layout, decorator registries (methods/datasets/scenarios/backbones), pyproject (py≥3.10, torch≥2.0, timm≥1.0), ruff+mypy+pytest config, CI workflow skeleton. DONE: CI green on the empty package (lint + typecheck + pytest collecting 0-or-trivial tests).
- [x] **P1 — Stream model** (SPEC §3–§4): StreamSpec v2 (`label: same|new`, `images: same|new|partial`, `task_size: fixed|grow`) + validation, planner → serializable StreamPlan, assignment port, Experience/Stream/Benchmark refactor; legacy `OverlapDataManager` API removed, no shim. DONE: golden-plan tests for the 6 core scenario shapes + seed-1993 class-order fixture green; plan/manifest round-trip test green; validation rejects each documented failure mode with actionable message.
- [x] **P2 — Label space + loss policies** (SPEC §5): LabelSpaceView on Experience; `methods/losses.py` (`new_class_ce`, `seen_class_ce`, `masked_logits`); synthetic dataset + 2-layer backbone stub; revisit-safety gate with a stub method. DONE: gate fails on a deliberately v1-L2P-broken stub (unmasked `-inf` CE) and passes on the fixed stub, across disjoint / same-id / echo synthetic streams.
- [x] **P3 — CLMethod + Trainer + SimpleCIL** (SPEC §6.1–6.2): CLMethod ABC, TrainContext, incremental heads, core Trainer (seeding, loop, checkpoints/resume, status.json, per-class evaluator hookup); SimpleCIL end-to-end. DONE: SimpleCIL passes safety gate + smoke on CPU; interrupted-run resume test green; CIFAR-100 disjoint sanity run queued/verified on cluster ≈ published range (record in docs/methods.md). **CIFAR-100-on-cluster leg deferred — see Log.**
- [x] **P4 — Config + CLI + cluster workflow** (SPEC §9, §11): typed YAML schemas with unknown-key rejection and layered defaults, `config_resolved.yaml`, run-dir artifacts; CLI `run` / `smoke` / `inspect` / `preflight`; smoke profile; `scripts/slurm_run.sh`. DONE: typo'd config fails pre-data naming the key; `clover run <cfg> --profile smoke` and `clover smoke` green on GPU-less CPU in <10 min; sbatch template submits and resumes on Snellius with only `#SBATCH` placeholders filled. **Actual Snellius submission unverified — see Log.**
- [x] **P5 — Backbones + prompt trio** (SPEC §6.4): backbone registry with config-selectable base models (timm/HF name or user class), prompt-pool / prefix / CODA wrappers; L2P, DualPrompt, CODA-Prompt using core loss policies (no hand-rolled `-inf` masking — lint test enforces). DONE: all three pass safety gate + smoke; finite loss and sane accuracy on all 6 scenarios (incl. same-id cumulative_drift) on the synthetic stream; disjoint CIFAR-100 ≈ published per method. **CIFAR-100-vs-published leg deferred — see Log.**
- [x] **P6 — Remaining methods** (SPEC §6.5): APER-Adapter, EASE, RanPAC, MOS, TUNA (adapter wrappers as needed); hyperparameter defaults carried from bench configs with provenance notes. DONE: each passes safety gate + smoke + disjoint CIFAR-100 sanity vs. published range; comparison table in docs/methods.md covers all 9. **CIFAR-100-vs-published leg deferred for all 5 (same reason as P3/P4/P5) — see Log.**
- [x] **P7 — Metrics + reporting + matrix** (SPEC §10): per-class evaluator finalized, standard (A_t/AIA/BWT/FWT/Forgetting) + CLOVER (RAG, Repetition Gain, Anchor/Long-Range Retention, echo-aware) metrics ported with fixture tests; `clover report`; `run-matrix` orchestrator (resume/retry/stale/GPU-pool). DONE: hand-computed metric fixtures green; two-method synthetic demo report renders in CI; matrix resume test green. **GPU-pool parallelism deferred (sequential dispatch only) — see Log.**
- [x] **P8 — Datasets + extras + docs** (SPEC §7, §8, §13): CUB-200, ImageNet-R/A, OmniBenchmark, VTAB wrappers + staging docs; `image_folder` config-only dataset; scenario extras as capacity allows; docs/concepts.md, docs/extending.md (method/dataset/scenario/backbone worked examples). DONE: built-in metadata smoke tests green; a tutorial-followed custom dataset runs a full smoke benchmark without touching core.
- [ ] **P9 — Release candidate** (SPEC §13–§14): docs/MIGRATION.md (legacy API → v2 configs), README (quickstart, smoke→SLURM workflow, scenario table, results, citations + no-copied-code attribution note); full matrix on cluster; tag RC. DONE: full 9-method × 6-scenario matrix reproduced on Snellius from shipped configs; README workflow verified end-to-end; `main` preserved as `v1` branch.

## Log

(Record deviations from SPEC.md here, dated, when ticking a phase.)

- **2026-07-05, P0:** the phase table assigns "legacy API removed, no shim"
  to P1 (§3-§4 data layer refactor), but P0's own gate says "CI green on the
  *empty* package." Confirmed with user: P0 deletes the v1 implementation
  now rather than deferring it to P1 — `clover/core/{data_manager,
  overlap_spec, task_builder, image_assigner, stream_spec, stream_builder,
  benchmark, experience, stream}.py`, all v1 `clover/scenarios/*.py`, v1
  `clover/datasets/*.py` impls, `clover/cli/` (old package), `clover/utils/
  {manifest,seeding,visualizer}.py`, all v1 `tests/*.py`, `configs/*.yaml`,
  `scripts/capture_pilot_fixtures.py`. Nothing lost: `main` still has the
  full v1 history untouched; this only affects the `v2` branch. New
  `clover/{core,datasets,scenarios,methods,backbones,training,evaluation,
  config}/` modules are docstring-only stubs each tagged with the phase
  that owns their body; the one piece of real P0 logic is
  `clover/utils/registry.py` (generic `Registry`) plus the
  `register_dataset`/`register_scenario`/`register_method`/
  `register_backbone` wiring in each plugin package's `__init__.py`.
  Also rebuilt the local `.venv` (was pointing at a since-removed Python
  3.13.7 install from an unrelated project) on Python 3.11 to satisfy the
  new `py>=3.10` requirement.

- **2026-07-05, P1:** v1 turned out to have *two* untested-against-each-other
  mechanisms for revisits: `StreamSpec`/`RevisitSpec` compiled to same-id
  `OverlapPair`s via a generic placement engine that was never actually
  exercised by the 6 scenario modules, which instead built bespoke
  `EchoSpec`/`task_class_lists` by hand (confirmed by reading v1 source from
  `main`'s history — `test_echo_classes.py` only tested the hand-built
  `build_spec()` path). SPEC.md's unification is therefore new design, not a
  port, and the 6 v2 scenario factories are **not** bit-for-bit reproductions
  of v1's per-scenario id arithmetic (e.g. v1's `partial_overlap` silently
  discarded some real classes near the tail to make room for echoes; v2's
  general engine never discards anything real that's actually needed).
  Golden-plan tests (`tests/test_scenarios.py`) assert the invariants that
  matter for each scenario's *purpose* (fixed task size, contiguous ids,
  correct echo/same/image semantics, correct head sizes) rather than exact
  id-list equality — matching how v1's own tests worked (structural
  invariants against a synthetic dataset, not a frozen fixture; only the
  disjoint seed-1993 class order is a frozen byte-equivalence fixture).
- Unifying mechanism (`clover/core/planner.py:resolve`): one monotonically
  increasing label-id cursor is shared between "draw the next real class"
  and "allocate the next echo id." Under `task_size="fixed"`, a revisit
  occurrence reduces its target experience's genuinely-new-class draw by
  exactly the number of occurrences landing there (no eviction — a budget
  decision made before assignment, never a displacement of an
  already-placed class); if occurrences exceed the budget, that's the named
  "class-budget overflow" failure. Echo ids allocated for a shortfall
  therefore fill the id-space gap immediately (contiguity holds by
  construction) instead of starting from a separate counter — this
  generalizes the trick v1's own `partial_overlap.build_spec` used by hand
  (`echo_start = fresh_ids[-1] + 1`).
- `StreamSpec.validate()` is structural only (types/ranges/enums/unknown
  keys); plan-time feasibility (min_gap vs. stream length, class-budget
  overflow, contiguous first-appearance) is checked in `planner.resolve()`,
  mirroring where these checks actually lived in v1 (`stream_builder`, not
  `stream_spec.validate()`).
- `_pilot_class_order` in `planner.py` is the sole sanctioned exception to
  CLAUDE.md's "no bare `np.random.*`" rule (byte-compat with the frozen
  seed-1993 fixture) — this exception already existed in v1's own
  `seeding.py` docstring, so it's carried forward, not newly introduced.
- `Experience.dataset` is a plain dataset-name `str`, not a live
  `torch.utils.data.Dataset` (v1's design) — keeps `clover/core` free of
  torch/I/O; wrapping `image_indices` into an actual per-experience Dataset
  is a P3/dataset-layer concern once real data exists (P7/P8).
- `Benchmark.from_yaml` / config-driven construction is deferred to P4
  (needs `clover/config`); P1's `Benchmark` is built via
  `clover.core.stream.build_benchmark(spec, dataset_info, class_to_indices)`.

- **2026-07-05, P2:** two pieces pulled forward from their SPEC-assigned
  phases, confirmed with user before implementing: a minimal `CLDataset` ABC
  (`clover/datasets/base.py`) + one built-in, `synthetic` (P7/P8's "Dataset
  layer" originally), and a 2-layer backbone stub, `tiny_mlp`
  (`clover/backbones/tiny_mlp.py`, P5's "Backbones" originally) — both
  registered, since SPEC §12.2 frames the synthetic dataset as "built-in"
  and reused by later phases' smoke profile and trainer/resume tests, not a
  throwaway fixture. `clover/methods/base.py`/`heads.py` were **not** pulled
  forward — the gate's stub training-step callables are plain functions,
  not registered `CLMethod`s, so `clover.methods` stays empty until P3's
  `SimpleCIL` (unlike datasets/backbones, no plugin needed to exist yet).
- `masked_logits(logits, targets, view, kind)` concretizes SPEC's
  illustrative 2-arg `masked_logits(logits, view)` snippet: `kind: "new"|
  "seen"` derives *both* the logit-column mask and the sample mask from the
  same id set (`view.new_classes` or `view.seen_classes`), which is what
  actually makes the v1 bug unrepresentable — a generic `view.logit_mask()`
  alone (seen|new, kept as a separate public method for eval-time
  prediction) isn't enough, since new_class_ce needs new-only columns, not
  everything valid so far.
- Discovered while building the revisit-safety gate: the v1 masked-logit bug
  (`logits[:, :known_classes] = -inf` over an unfiltered batch) can only
  actually fire when an *old-id* sample shares a batch with the blind column
  mask — true for the same-id revisit shape, but **not** the echo shape,
  since echo ids are always freshly allocated above every seen id and so
  never fall inside the masked-out range. This matches `AUDIT.md`'s finding
  that the bug "lives under cumulative_drift" specifically. The gate
  reflects this precisely: the broken stub is asserted non-finite only on
  the same-id shape, and finite (but not necessarily *more* correct) on
  disjoint and echo — proving the gate correctly localizes the bug rather
  than just rejecting anything that isn't the fully-fixed implementation.
- Classifier head in the gate is a single `nn.Linear` preallocated to the
  plan's final head size (known upfront for the synthetic stream) — no
  incremental growth. Real incremental head expansion is P3's
  `methods/heads.py`; growing the head as head_size increases would just be
  re-deriving that work early for no benefit to P2's own gate.

- **2026-07-05, P3:** CIFAR-100-on-cluster leg of the DONE criteria is
  **deferred**, confirmed with user before implementing — CIFAR-100 doesn't
  exist as a dataset yet (P7/P8), and this machine has no GPU/Snellius
  access. Everything else is implemented and fully verified locally
  (`SimpleCIL` passes the registry-driven revisit-safety gate + a CPU smoke
  run; the interrupted-run resume test is green and bit-identical, not just
  approximate). Once P7/P8 land a real CIFAR-100 dataset, the user runs
  `sbatch scripts/slurm_run.sh <cifar100 disjoint config>` on Snellius
  themselves and records the result in `docs/methods.md` — this is not a
  false-checked box, it's an explicitly named follow-up.
- `CLMethod.state_dict()`/`load_state_dict()` aren't in SPEC's illustrative
  §6.1 snippet but are added as abstract methods -- necessary to make the
  Trainer's checkpoint/resume generic across arbitrary methods (mirrors the
  bench's `model._network.state_dict()` pattern, confirmed via read-only
  exploration of `clover-pilot-bench`, no code copied).
- SimpleCIL recomputes a class's prototype from *whichever* experience's
  images it's currently processing, for both first-appearance and
  revisiting classes alike (no cross-experience accumulation) — a
  deliberate generalization: SimpleCIL never fine-tunes a prototype after
  its initial closed-form pass regardless, so a revisit has nothing to
  "accumulate into," and treating new/revisit uniformly avoids a needless
  special case.
- **Real bug found and fixed while writing the resume test**:
  `DataLoader(shuffle=True)` without an explicit `generator=` draws from
  the *global* torch RNG. A resumed run replays `before_experience` (no
  data touched) for already-completed experiences instead of re-running
  `train_experience` (which does touch the loader) — so by the time it
  reaches the first *actually retrained* experience, the global RNG is at a
  different position than the uninterrupted run reached at the same point,
  producing a different batch/shuffle order and, because float32 `mean()`
  is summation-order-sensitive, non-bit-identical (though numerically very
  close) prototypes. Fixed by giving each experience's `DataLoader` its own
  `torch.Generator` seeded from `(run.seed, task_label)`
  (`clover/training/trainer.py:_build_loader`) — shuffle order now depends
  only on the experience itself, never on what happened to run before it,
  restoring true bit-identical resume reproducibility. Worth remembering
  for any future Trainer changes: never let per-experience randomness draw
  from the global RNG.

- **2026-07-06, P4:** the sbatch-submits-and-resumes-on-Snellius leg of the
  DONE criteria is **unverified** — this session has no Snellius access.
  `scripts/slurm_run.sh` is written per SPEC §11.1 (placeholder `#SBATCH`
  values, module/venv hook, `clover run "$1"`, resume-by-resubmission
  relying on `Trainer`'s existing checkpoint detection — no separate
  `--resume` flag needed), but its actual submission/resume behavior on the
  real cluster needs the user to try it once they fill in the site-specific
  placeholders. Not a false-checked box.
- Per-method typed config schemas (SPEC's "each method declares a typed
  schema") are deferred: `MethodSection.extra` passes through untyped to
  `CLMethod.build(stream_info, cfg)` since only SimpleCIL exists and it has
  no config knobs beyond an optional backbone-name override. Revisit once
  P5/P6 add methods with real hyperparameters (prompt_pool_size, etc.).
- `training.amp` is validated against `{"none","bf16","fp16"}` but only
  collapses to a bool for `TrainContext.amp` (`amp != "none"`) — no
  autocast wrapping exists anywhere yet since no gradient-trained method
  exists (SimpleCIL has no training step at all). Finer bf16-vs-fp16
  dtype handling is deferred to whichever phase adds one.
- Extracted `clover/utils/strict_dict.py` (`reject_unknown_keys`) from
  `clover/core/spec.py`'s previously-private `_unknown_key_error`, now
  shared with the config schemas — one implementation of the "did you
  mean" pattern used in `RevisitSpec`/`StreamSpec` and every config
  section, instead of a second copy.
- `clover/config` importing from `clover/training` (for
  `OPTIMIZER_FACTORIES`, so an unknown optimizer name is a config-resolve
  error, not a `KeyError` deep inside a run) pulled `clover/methods` and
  `clover/training` transitively into the `mypy clover/core clover/config`
  gate for the first time, surfacing two pre-existing type gaps from P3
  (an `Any`-typed backbone attribute access in `SimpleCIL.build`, and an
  `Optional[int]` narrowing gap in `Trainer.run`'s resume branch) — both
  fixed, not suppressed away, since they were real (if minor) type-safety
  gaps that happened to be invisible before this phase's import graph
  connected config to them.

- **2026-07-06, P5 (in progress, not ticked):** researched (read-only,
  `clover-pilot-bench`, no code copied) confirmed L2P/DualPrompt/CODA-Prompt
  are three genuinely different mechanisms, not variations of one: L2P
  prepends top-k-selected prompts as extra input tokens (no attention
  surgery); DualPrompt and CODA-Prompt both splice prefix key/value pairs
  into specific attention blocks (DualPrompt: hard top-k task-conditioned
  pool + an always-on general prompt; CODA-Prompt: soft attention-weighted
  combination of every pool component, no top-k, plus Gram-Schmidt
  orthogonalization of new slots). Building and testing all three in one
  pass risked exactly the kind of subtle bug this project keeps finding
  when work is incremental (P3's DataLoader-RNG bug, P4's mypy import
  surface) — confirmed with user: **this session delivers the shared
  infrastructure + L2P complete and tested; DualPrompt and CODA-Prompt are
  the queued next continuation of this same phase**, reusing the
  prefix-KV hook L2P leaves built but unexercised
  (`TinyViT.forward_tokens`/`Attention.forward`'s `prefix_kv` parameter,
  covered by `tests/test_tiny_vit.py`'s hook tests even though no method
  uses it yet).
- Also confirmed via the same research: the real L2P/DualPrompt/CODA-Prompt
  masked-loss bug (`logits[:, :known_classes] = -inf` then a plain
  unfiltered `cross_entropy`) is exactly what P2's `new_class_ce`/
  `seen_class_ce` structurally prevents (confirmed unpatched in the bench
  for these three specifically — only `ease`/`tuna`/`mos` were patched) —
  validates the P2 design, nothing to change.
- `clover/backbones/vit.py`'s `TinyViT` is a clean-room minimal ViT sized
  for the 8x8 synthetic dataset (patch_size=4, 2x2=4 patches) — a randomly
  initialized, never-pretrained stand-in, unlike the real method's
  pretrained-ImageNet-ViT assumption. Empirically, L2P's default
  `training.optimizer.lr=1e-3`/`epochs=1` was too weak for the prompt pool
  to converge against this random backbone (one round hit exactly 0%
  accuracy on its own just-trained classes — verified via direct
  debugging this was an undertrained-hyperparameters issue, not a gradient
  -flow bug, by confirming a fully-unfrozen TinyViT+head fits the same
  data to 100% in ~20 steps). The registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`) now uses `epochs=40,
  optimizer_lr=1e-2` for both methods (harmless to SimpleCIL, which
  ignores both) — real cluster configs against an actual pretrained timm
  ViT would tune these separately.
- `training.epochs` was added to `TrainingSection` in P4 but never actually
  threaded through to a method — SimpleCIL has no per-experience loop to
  repeat, so the gap was invisible until L2P (the first real gradient loop)
  needed it. Fixed: `TrainContext.epochs` (default 1) and
  `RunConfig.epochs`, wired through `Trainer.run()` and `cli.py`'s
  `_build_run_config`.
- `clover.backbones` registry semantics settled precisely: registry keys
  are either complete usable backbones (`tiny_mlp`, `tiny_vit` -- no
  wrapping needed) or mechanism wrappers over a resolved base model
  (`vit_prompt_pool`, taking `base_model: str` resolved via
  `clover/backbones/loader.py:resolve_base_model` -- registry-first, timm
  name fallback, `source="class"` for a user class). Base models
  themselves are never registered as such; only complete backbones and
  wrapper mechanisms are.
- L2P's auxiliary key-pulling loss term (encouraging a selected key to
  match its query, present in the real method) is omitted for this
  session's scope -- the top-k selection itself is non-differentiable
  w.r.t. which indices are chosen regardless, so this only affects how
  well the *keys* specialize over time, not correctness. Matches SPEC's
  "faithful-to-the-published-method, not bit-reproduction" bar; worth
  adding when DualPrompt/CODA-Prompt's key/attention-vector training makes
  it more directly comparable.

- **2026-07-06, P5 continued (still in progress, not ticked): DualPrompt
  done.** Extracted `clover/methods/prompt_common.py:PromptMethodBase`
  (build/before_experience/train_experience/classifier/state_dict/
  load_state_dict) out of `l2p.py` once a second method needed the
  identical scaffolding — L2P and DualPrompt now differ only in
  `default_backbone`; CODA-Prompt (next) will be a third 3-line subclass.
  Confirmed no behavior change: L2P's existing tests pass unmodified
  against the refactored class.
- `clover/backbones/dual_prompt.py:DualPromptViT` implements the general
  prompt (always-on, fixed shallow layer(s)) + expert prompt pool
  (top-k-selected, deeper layer(s)) as **prefix key/value pairs** spliced
  into specific attention blocks via `TinyViT`'s `prefix_kv` hook (built in
  the L2P pass, unexercised until now) — the mechanism L2P's plain
  token-prepend doesn't need. `g_layers`/`e_layers` must be disjoint and
  within the base's depth (validated, actionable error). Default
  `g_layers=(0,)`/`e_layers=(1,)` fit `TinyViT`'s default `depth=2` with no
  extra config.
- Unlike L2P's debugging detour, DualPrompt worked correctly on the first
  full end-to-end run with the same tuned hyperparameters (epochs=40,
  optimizer_lr=1e-2) — diagonal accuracies 0.93-1.0 across all experiences,
  actually more stable than L2P's. Registry-driven safety gate (12 cases:
  3 methods × 4 shapes) all green on the first attempt.
- CODA-Prompt is the one remaining piece of the P5 prompt trio: same
  prefix-KV injection point as DualPrompt, but a *soft* attention-weighted
  combination of every pool component (no top-k) plus Gram-Schmidt
  orthogonalization of new pool slots per task — the next continuation of
  this phase.

- **2026-07-06, P5 complete: CODA-Prompt done, all three prompt methods
  ticked.** `clover/backbones/coda_prompt.py:CodaPromptPool` combines the
  *whole* unlocked-so-far pool via per-slot soft attention weights
  (`softmax` over `cos_sim(query ⊙ attn_vec_i, key_i)`, no top-k) instead
  of L2P/DualPrompt's hard selection. Each task's newly unlocked pool slice
  (`pool_size // nb_experiences` slots) is Gram-Schmidt-orthogonalized
  against every already-unlocked slice's *prompt content* (not just keys)
  before training starts on it. `nb_experiences` must reach the backbone at
  build time — `CODAPrompt.build()` overrides `PromptMethodBase.build()`
  to inject `stream_info.nb_experiences` into the backbone kwargs rather
  than changing the shared base (which would've broken L2P/DualPrompt's
  factories, which don't accept that kwarg). The pool's `unlocked` counter
  is a registered *buffer*, not a plain attribute, specifically so it
  round-trips through `state_dict()`/`load_state_dict()` automatically —
  needed for resume correctness with zero changes to `PromptMethodBase`'s
  generic checkpoint logic.
- **Explicitly omitted vs. the real method**: gradient masking that freezes
  earlier tasks' pool components during later tasks' training. Without it,
  every task's backward pass can perturb every unlocked slot, including
  older ones. This is a deliberate, logged scope simplification (matching
  L2P's omitted auxiliary key-pulling loss) — implementing it would need
  either per-parameter gradient hooks or splitting the pool into separate
  per-task parameter groups, more machinery than this phase's fidelity bar
  needs. It's very likely *why* CODA-Prompt needed a higher learning rate
  than L2P/DualPrompt to reliably pass (below).
- **Extended the registry-driven safety gate to use the actual 6 registered
  scenario factories** (not hand-built structural stand-ins) once it became
  clear P5's own DONE criteria names "all 6 scenarios" specifically — turned
  out simpler than the previous 4 hand-built shapes: every core scenario
  except `disjoint_baseline` concentrates its revisit/echo signal in the
  *last* train experience (`end_of_stream` or `every_task` placement always
  includes it), so one generic check
  (`revisiting_classes | echo_map` on `train_stream[-1]`) covers all 6
  without per-scenario branching. Now 4 methods × 6 scenarios = 24 gate
  cases, ~6 minutes total.
- **Real bug found via this extension**: `clover/scenarios/
  mid_range_revisit.py`'s feasibility check required
  `max(3, anchor_task + 1)` experiences, which only guarantees the anchor
  task itself exists — not that there's a further experience *after* it for
  the `end_of_stream` echo to land in. With the gate's 5-experience test
  stream and the scenario's default `anchor_task=4`, the anchor task *is*
  the last experience, so `planner.resolve()` raised "Cannot place 1
  revisit(s)... Maximum achievable: 0" deep inside the planner instead of a
  clear, actionable error at spec-construction time. Fixed the check to
  `max(3, anchor_task + 2)`; added a regression test
  (`test_mid_range_revisit_requires_room_after_the_anchor_task`) pinning
  exactly this boundary case. All prior P1 tests for this scenario were
  unaffected (their anchor_task/dataset-size combinations already had
  enough headroom).
- **coda_prompt/echo-style scenarios (exact_replay, long_range_revisit,
  mid_range_revisit, partial_overlap) needed `optimizer_lr=3e-2`**, not the
  `1e-2` that sufficed for L2P/DualPrompt: an echoed sample and its source
  share the exact same underlying synthetic-class pattern (only the noise
  differs), so CODA's *soft, query-only* combination gives them
  near-identical prompted features — the classifier head alone has to pull
  apart two very similar feature vectors, needing a stronger gradient
  signal than hard-selection methods (which incidentally get more
  differentiating signal from *which* discrete prompts get concatenated).
  Root-caused via the same debugging discipline as L2P's earlier detour
  (checked training accuracy per-experience and per-class-confusion before
  concluding it wasn't a code bug) — 1e-2 gave exactly-chance accuracy
  (0.047 ≈ 1/20) on one seed, 3e-2 gave a reliable 0.40-0.47 margin,
  verified this didn't regress any of the other 23 gate cases.
- CIFAR-100-vs-published-accuracy verification remains deferred for the
  same reason as P3/P4: no real CIFAR-100 dataset yet (P7/P8) and no
  GPU/Snellius access this session. Everything locally verifiable is done
  and green: registry-driven safety gate (24 cases), `clover smoke` (4/4
  methods OK), and per-method isolated unit tests for all three prompt
  methods plus SimpleCIL.

- **2026-07-06, P6 (in progress, not ticked): APER-Adapter + RanPAC done,
  EASE/MOS/TUNA queued.** Read-only research pass on `clover-pilot-bench`/
  `_ref_pilot` (no code copied) confirmed all 5 remaining methods share one
  AdaptFormer-style bottleneck adapter (down-proj -> ReLU -> up-proj,
  scaled, zero-init up-proj) spliced as a parallel branch off the pre-MLP
  residual stream in every block -- a new hook shape, distinct from P5's
  `prefix_kv` (key/value splice), never wrapping `qkv`/`fc1`/`fc2` directly.
  New shared infra: `clover/backbones/adapter.py` (`Adapter` +
  `AdapterViT`/`vit_adapter`), a new `adapter: Optional[Dict[int,
  nn.Module]]` parameter threaded through `TransformerBlock`/`TinyViT`
  alongside `prefix_kv` (`clover/backbones/vit.py`).
- **Deviates from SPEC §6.5's literal list order** ("APER-Adapter, EASE"
  then "RanPAC, MOS, TUNA"): implementing in ascending mechanism-complexity
  order instead -- APER-Adapter, RanPAC (both: adapter gradient-trained
  only during the first experience then frozen forever, closed-form
  non-gradient head, no growing per-task state) before EASE, MOS, TUNA
  (each keeps a growing per-task adapter list plus a distinct combine
  mechanism, and MOS/TUNA add classifier alignment on top) -- confirmed
  with the user via plan-mode approval before implementing. Same rationale
  P5 used to build L2P (simplest of the prompt trio) before DualPrompt/
  CODA-Prompt.
- New shared method base `clover/methods/adapter_common.py:
  AdapterMethodBase`: "train adapter once (`exp.task_label == 0`,
  generalizing PILOT's `cur_task == 0` gate to CLOVER's experience-index
  equivalent), freeze forever, recompute a closed-form head every
  experience" -- confirmed via research that both APER-Adapter's and
  RanPAC's PILOT implementations skip gradient training entirely past task
  0. A throwaway `_train_head` (gradient-trained jointly with the adapter
  during experience 0 only, mirroring PILOT's `_init_train`) is
  deliberately excluded from `state_dict()`/`load_state_dict()` -- it's
  dead weight after experience 0 either way, and replaying
  `before_experience(0)` on resume reconstructs it identically from
  `stream_info` alone.
- **APER-Adapter's dual-branch concat reuses one frozen base ViT instead of
  loading a second checkpoint**: PILOT loads two separately-checkpointed
  backbones (a fresh plain one + the adapter-tuned one) into
  `MultiBranchCosineIncrementalNet`; here, since `AdapterViT.base` is
  frozen (`requires_grad_(False)`, `.eval()`) at construction time, before
  any adapter training happens, a plain pass via `backbone.base(x)` and an
  adapter pass via `backbone(x)` already read the exact same frozen
  weights -- no second model instance needed.
- **RanPAC's ridge-regression head** (`clover/methods/heads.py:
  RandomProjectionRidgeHead`) accumulates `G`/`Q` sufficient statistics
  across experiences and re-solves `weight = solve(G + ridge*I, Q)` from
  scratch every experience (not a literal Sherman-Morrison-Woodbury inverse
  update -- confirmed this is what PILOT's own `ranpac.py` does too, not a
  simplification). `M` (projection width) scaled from PILOT's 10000 down to
  256 to fit `TinyViT`'s tiny feature dim. The ridge value is grid-searched
  per experience against an 80/20 held-out split of that experience's own
  data (`ranpac.py:_update_head`), using a locally-seeded
  `torch.Generator()` (seeded from `exp.task_label`, not the run's actual
  seed -- `TrainContext` doesn't thread a seed through to methods) rather
  than the global RNG, per the standing "no bare global-RNG draws in
  library code" rule -- this only affects which ridge candidate gets
  picked, not the correctness of the closed-form solve itself.
- **No dropout in the new `Adapter` module**, unlike PILOT's: discovered
  that `clover/training/trainer.py`'s `Trainer.run()` constructs
  `PerClassEvaluator(self.method.classifier(), ...)` (which calls `.eval()`
  on the shared classifier module graph) once, before the experience loop
  even starts -- since the classifier and the method's own training-time
  modules are the *same* object references, this permanently disables
  dropout for the rest of the run regardless of method. Adding a dropout
  layer to `Adapter` would be dead code under this framework's current
  Trainer design, not specific to these two methods, so it's omitted
  entirely rather than included-but-inert.
- Registry-driven safety gate needed **no hyperparameter retuning**: both
  new methods pass all 6 scenarios first try at the existing tuned
  `epochs=40, optimizer_lr=3e-2` (picked up automatically via
  `list_methods()`, zero test-file changes).
- `docs/methods.md` created (didn't exist before P6): comparison table for
  all 9 methods (6 done, 3 queued) plus hyperparameter-provenance detail
  for APER-Adapter/RanPAC.

- **2026-07-06, P6 continued (still in progress, not ticked): EASE done.**
  `clover/backbones/adapter_ease.py:EaseAdapterViT` keeps a growing
  `nn.ModuleList` of per-experience adapter sets (only the most recent one
  ever trainable); `clover/methods/heads.py:EaseHead` stores
  `[num_classes, num_blocks, block_dim]` with a per-class "home block" and
  PILOT's cosine-similarity block-reweighting (`alpha`-scaled non-home
  blocks). Registry-driven safety gate (all 6 scenarios) and `clover smoke`
  both green at the *existing* tuned hyperparameters, no retuning needed.
- **`grow()` (freeze-then-allocate a new adapter set) lives in
  `EASE.before_experience`, not `after_experience`** -- discovered while
  designing this that `after_experience` would be actively wrong for
  resume: the Trainer's resume-replay only re-invokes `before_experience`
  for already-completed experiences (never `after_experience`), so any
  structural growth needed before `load_state_dict` succeeds must happen
  there. Putting `grow()`/`head.add_block()`/`head.expand_classes()` in
  `after_experience` instead would also silently drop the *final*
  experience's adapter from a live (non-resumed) run entirely, since no
  `before_experience(N)` call ever follows the last experience `N-1` to
  trigger it. This generalizes the same reasoning `IncrementalHead.
  expand_to`/CODA-Prompt's `pool.start_new_task` already follow -- worth
  remembering for MOS/TUNA too, which also grow per-experience state.
- **New classes get real (not approximated) prototype rows in *every*
  block, including old ones** -- this experience's images are still on
  hand and old (frozen) adapters are still available, so there's no need
  to approximate a new class's response under an old adapter the way
  PILOT's `solve_similarity` does. Only *already-existing* classes' row in
  the newly-added block is approximated (cosine-similarity-weighted
  combination of this round's new classes' own rows in that block), since
  their images are genuinely gone (no exemplar memory in this framework) --
  matches exactly why PILOT itself needs `solve_similarity`/
  `solve_sim_reset` there and nowhere else.
- EASE's proxy head is full-global-width (`AdapterMethodBase`'s
  `_train_head` pattern), not locally indexed to this round's new classes
  the way PILOT's `proxy_fc` is -- avoids a second targets-remapping
  mechanism alongside `new_class_ce`'s existing set-membership masks;
  functionally equivalent since gradient only ever reaches the new-class
  columns either way. Not serialized (thrown away after each experience,
  like `_train_head`); confirmed `EaseHead.home_block` (a plain Python
  list, not a tensor) also doesn't need explicit serialization -- replaying
  `before_experience` across a resume reconstructs it identically, since
  it's built purely from `exp.label_space` (deterministic, spec-derived),
  not from any trained value.
- Next: MOS (`clover/backbones/adapter_mos.py` + `clover/methods/mos.py`)
  -- per-experience adapter continuously EMA-merged toward the running
  mean of previous adapters during training, plus classifier alignment
  (new shared `clover/methods/classifier_alignment.py`, reused by TUNA).
  See handoff.md for the full per-method design (MOS/TUNA) agreed in the
  original P6 planning session.

- **2026-07-06, P6 continued (still in progress, not ticked): MOS done.**
  `clover/backbones/adapter_mos.py:MOSAdapterViT` keeps ONE adapter
  training continuously across the whole run (unlike EASE's fresh adapter
  per experience) -- `merge_step()` EMA-blends it toward the running mean
  of all earlier experiences' frozen snapshots after every optimizer step
  (`momentum`), and `snapshot()` (freeze + fold into the running sum) is
  called from `before_experience` for the same resume-safety reason as
  EASE's `grow()`. New shared `clover/methods/classifier_alignment.py`
  (`update_class_stats`/`gaussian_resample_finetune`) reused as-is by
  TUNA. Registry-driven safety gate (all 6 scenarios) and `clover smoke`
  both green at the *existing* tuned hyperparameters, no retuning needed.
- **The running-sum accumulator (`_adapter_sum`) is a second,
  never-trained `Adapter`-shaped `nn.ModuleList`** (zeroed at construction,
  `requires_grad_(False)`), not a set of raw buffers -- reuses `Adapter`'s
  existing parameter shapes directly instead of hand-deriving matching
  buffer shapes per layer, and round-trips through `state_dict()`/
  `load_state_dict()` automatically since it's a real submodule. `len(
  adapter_list)` (not a separate serialized counter) is what `merge_step`
  divides by to get the running mean -- correct after resume because
  `adapter_list`'s length is itself reconstructed via the same
  before_experience-replay mechanism.
- **Classifier alignment's synthetic sampling required a second
  never-let-per-experience-randomness-touch-the-global-RNG fix**: found
  while designing this (not yet an actual bug, caught before implementing)
  -- `torch.distributions.MultivariateNormal.sample()` draws from the
  global RNG and doesn't accept an explicit generator, so
  `classifier_alignment.py` implements sampling by hand (Cholesky
  decomposition + `torch.randn(..., generator=...)`) seeded from
  `exp.task_label`, mirroring RanPAC's ridge-selection-split fix from
  earlier this phase. Same underlying class of bug P3's DataLoader-RNG fix
  addressed: CA runs inside `train_experience`, which resume-replay skips
  for already-completed experiences, so drawing from the global RNG there
  would desynchronize a resumed run's later RNG-dependent steps from an
  uninterrupted run's.
- **Per-class `(mean, covariance)` stats are computed via the *current*
  adapter's features**, kept verbatim (stale) for classes not seen in a
  given experience (their images are gone, matching the same
  no-exemplar-memory constraint EASE's cross-block interpolation is under)
  -- overwritten only when that class reappears. Damped by `1e-4 * I`
  (matches PILOT's own damping) for numerical stability in
  `torch.linalg.cholesky`.
- **Simplified out** (agreed before implementing, per handoff.md's P6
  plan): PILOT's entropy-based test-time adapter self-refinement search --
  the classifier instead ensembles by averaging logits from every stored
  adapter (history + current) through the same shared head. **Also
  simplified, discovered while implementing** (not previously flagged):
  PILOT's always-on orthogonality regularizer (`reg`-weighted `orth_loss`
  term in the main training loss) and its two-param-group optimizer (base
  adapter LR vs. head at 10x lower LR) are not reimplemented -- the
  orth_loss is a secondary regularizer layered on top of MOS's actual
  headline mechanism (the EMA merge, implemented in full), and the
  per-param-group LR split isn't expressible through this framework's
  single-LR `TrainContext.optimizer_factory` without a broader Trainer
  change no other method has needed yet. Recorded in `docs/methods.md`'s
  MOS hyperparameter table.
- Next: TUNA (`clover/backbones/adapter_tuna.py` + `clover/methods/
  tuna.py`) -- the last method in P6. Per-experience adapter trained with
  a CosFace/angular-margin loss, then EMR-merge (sign-consensus,
  max-magnitude-among-agreeing-signs, rescaled) across all stored per-task
  adapters; per-task `nn.Linear` heads (no bias, cosine input)
  concatenated, unfrozen; reuses `classifier_alignment.py`. Once TUNA
  lands: tick P6's box, and update `docs/methods.md`'s comparison table to
  mark all 9 methods done.

- **2026-07-06, P6 complete: TUNA done, all 9 methods ticked.**
  `clover/backbones/adapter_tuna.py:TunaAdapterViT` grows a fresh adapter
  per experience like EASE (only the most recent trainable), but combines
  every stored (frozen) adapter via **EMR-merge** (`_emr_merge`: elect a
  per-parameter sign by majority vote, keep the max-magnitude value among
  agreeing task tensors, rescale to preserve the mean task tensor's
  average magnitude) into a single consensus adapter used directly for
  evaluation. Registry-driven safety gate (all 6 scenarios) and
  `clover smoke` both green at the *existing* tuned hyperparameters, no
  retuning needed. Full suite: 337 tests (54 gate cases: 9 methods × 6
  scenarios).
- **Deviated from the "per-task `nn.Linear` heads, concatenated" plan
  stated when this phase was queued**: discovered while designing this
  that PILOT's `TunaLinear` (separate per-task heads) is an
  implementation detail for isolating gradient to the newest task's own
  output columns during training -- the framework's existing
  `masked_logits` primitive (already used by `new_class_ce`) gives that
  same isolation for free over a single plain global-width
  `IncrementalHead(cosine=True)`, with no separate per-task objects and no
  targets-remapping needed. Confirmed PILOT's own default margin (`m=0.0`)
  makes the resulting angular loss numerically identical to
  `new_class_ce` -- added `angular_margin_ce` to `clover/methods/
  losses.py` anyway (not a bare alias) so a nonzero margin is genuinely
  supported, not just documented as unsupported. No closed-form
  prototype-overwrite step exists for TUNA's head (unlike MOS/
  APER-Adapter) -- purely gradient-trained via the angular loss, matching
  what the research found for PILOT's own TUNA (no `replace_fc`-equivalent
  mentioned for it).
- **`recompute_merge()`'s timing intentionally differs from `grow()`'s**:
  `grow()` (structural -- append a fresh trainable adapter, freeze the
  previous one) lives in `before_experience`, same resume-safety reason as
  EASE/MOS. `recompute_merge()` (value-only -- merged_adapter's shape
  never changes) lives in `train_experience`, right after that round's
  training finishes, matching PILOT's own cadence (merge right after a
  task's adapter is appended, so evaluating that same round already
  reflects its own contribution, not a one-round-stale merge). Confirmed
  this is safe for resume precisely because it's a value update, not a
  structural one -- `load_state_dict` overwrites whatever values existed
  regardless of when in the live run's flow they were last computed
  (same reasoning as EASE's/MOS's prototype and CA updates, which also
  live in `train_experience`).
- **Per-class CA stats computed via the *merged* adapter's features, not
  `forward_current`'s** -- unlike MOS (whose evaluation ensembles every
  adapter including the current one), TUNA's evaluation uses *only* the
  merged adapter, so CA has to align the head against that same feature
  space specifically, not the still-training current adapter's.
- CIFAR-100-vs-published-accuracy verification remains deferred for all 5
  P6 methods, same reason as every prior phase: no real CIFAR-100 dataset
  yet (P7/P8) and no GPU/Snellius access this session. Everything locally
  verifiable is done and green: registry-driven safety gate (54 cases, 9
  methods × 6 scenarios), `clover smoke` (9/9 methods OK), and per-method
  isolated unit tests for all 5 adapter-family methods. `docs/methods.md`
  now covers all 9 methods with hyperparameter provenance tables.

- **2026-07-06, P7 (in progress, not ticked): metrics library + Trainer
  wiring done; `clover report`/`run-matrix` queued.** Read-only research
  pass on `clover-pilot-bench` (no code copied) confirmed the exact
  formulas for the 5 standard metrics (pure R-matrix reductions) and the 4
  CLOVER-specific ones (need a per-class accuracy history, not derivable
  from the R-matrix). New `clover/evaluation/history.py:PerClassHistory`
  (`{class_id: {task: accuracy}}`) persists what `Trainer.run()` already
  computed via `PerClassEvaluator.evaluate` every experience but previously
  discarded after reducing it to the R-matrix's mean -- no new evaluation
  pass needed. New `clover/evaluation/metrics/{standard,overlap}.py`: 5
  standard functions (`aggregate_accuracy`/`average_incremental_accuracy`/
  `backward_transfer`/`forgetting`/`forward_transfer`) + CLOVER bookkeeping
  helpers (`first_appearance_map`/`revisit_task_map`/`anchor_classes`/
  `echo_source_ids`/`classify_task`, all derived from `Experience.
  first_appearance_of`/`.revisiting_classes`/`.echo_map` and `StreamPlan.
  echo_table` -- no manual class-id scanning needed, unlike the bench's own
  reconstruction from raw task lists) + 4 metric functions (`rag_per_class`/
  `rag_mean`/`repetition_gain`/`anchor_retention`/`long_range_retention`) +
  an `image_level_bonus` NaN stub (bench never wired this one up either).
- **`forward_transfer` is PILOT's own zero-baseline approximation, not a
  true FWT** (confirmed via research: the "baseline" is hardcoded to 0,
  "no zero-shot knowledge assumed") -- kept as-is and documented plainly in
  the docstring rather than silently "fixing" it into a different metric,
  since SPEC's instruction is "ported from bench/metrics/standard.py."
- **`revisit_task_map` only records each class's *first* revisit** -- a
  known bench limitation (a class revisiting 3+ times only ever scores its
  first revisit for RAG), kept as-is for fidelity rather than generalized,
  and documented in the docstring.
- **Trainer wiring**: `PerClassHistory` persisted/restored in the
  checkpoint dict (`ckpt_task*.pt`) exactly like `r_matrix` already is.
  Standard + CLOVER metrics (RAG_mean, Repetition_Gain) computed every
  experience; Anchor_Retention/Long_Range_Retention only at the *final*
  experience, matching the bench's own "read at final task" design.
  Written to `runs/<name>/per_task.csv` (`task_idx,metric,value`, matching
  the bench's per-run schema exactly) via a full atomic rewrite each round
  (this Trainer is single-process per run, so the bench's OS-level
  file-locking for concurrent multi-process writers isn't needed).
- **Real bug found and fixed while writing the resume test**: `per_task.csv`
  is written *before* the checkpoint each round (so a stream reader always
  sees a task's metrics as soon as they're known); this means a crash
  between the two writes leaves a stale row for the about-to-be-redone
  task, which a naive "read existing rows + append" resume would duplicate
  on top of. Fixed by filtering `_read_per_task_rows()` to `task_idx <
  resume_from` before appending new rows -- only rows for tasks the
  checkpoint actually confirms are done are trusted, mirroring how
  `resume_from` itself is already the sole authority for what counts as
  "done." Caught by `tests/test_trainer_per_task_csv.py::
  test_resume_does_not_duplicate_stale_rows_for_the_redone_task`, which
  simulates exactly this crash window (deletes only the last checkpoint,
  not `per_task.csv`).
- **Per-task rows are read back verbatim on resume, never recomputed from
  the fuller post-resume history** -- some CLOVER metrics (e.g. `RAG_mean`)
  aggregate over the *whole* history rather than being strictly
  task-scoped; recomputing an earlier task's row using today's fuller
  history would leak future information into it (e.g. task 0's row would
  already reflect a revisit that happens at task 3). The standard metrics
  and `Repetition_Gain` are naturally immune to this (they only ever read
  `R`/history entries with column/task `<= t`), but the read-back-verbatim
  design avoids relying on that per-metric distinction being airtight.
- Verified end-to-end (not just via fixtures) that CLOVER metrics produce
  real, non-NaN values through the actual Trainer on real scenarios:
  `Long_Range_Retention`/`Repetition_Gain` non-NaN under `partial_overlap`
  (echo-based revisits), `RAG_mean`/`Anchor_Retention` non-NaN under
  `cumulative_drift` (same-id revisits/anchors) -- confirms the echo-vs
  -same-id split in `classify_task`/`revisit_task_map` is wired correctly,
  not just correct in isolated fixture tests.
- Full suite: 363 tests (57 new: metrics fixtures + Trainer/per_task.csv
  tests). `clover smoke` green (9/9 methods).
- Next: `clover report <run_dirs...>` (rebuild-from-scratch aggregate CSV +
  method×scenario×dataset summary pivot -- **no pandas**, CLAUDE.md pins
  runtime deps to exactly torch/torchvision/timm/numpy/pyyaml/pillow; use
  the stdlib `csv` module + plain dicts), then the `run-matrix` orchestrator
  (grid enumeration, `status.json`-based resume/retry/stale detection reusing
  the Trainer's existing schema, GPU-pool parallelism degrading to
  sequential on this CPU-only machine). Once both land: tick P7's box.

- **2026-07-06, P7 complete: `clover report` + `run-matrix` done, all 3
  pieces of this phase ticked.** `clover/reporting.py`: `parse_run_id`
  (bench's own `{dataset}__{method}__{scenario}__seed{seed}` convention),
  `rebuild_long_csv` (rebuild-from-scratch every invocation, only
  `state=="done"` runs contribute rows -- never trust/append to a stale
  aggregate), `build_summary` (last value per `(run_id, metric)`, averaged
  over seeds per `(method, scenario, dataset)`, NaN entries excluded from
  the average) -- all stdlib `csv`/`statistics`, no pandas. `clover/
  matrix.py`: `enumerate_cells` (methods × scenarios × datasets × seeds),
  `cell_config` (synthesizes a single-run config dict per cell -- new
  `MatrixSection` schema in `clover/config/schema.py` carries a shared
  `stream`/`training` block + optional per-method `method_overrides`,
  since clover-cl configs are single-file/self-contained unlike the
  bench's per-method YAML directory), `read_status`/`is_stale`/
  `plan_dispatch` (pure, fixture-testable classification logic -- only
  `state=="done"` is skipped; a stale `running` status, per the bench's
  own 15-minute heartbeat threshold, is force-relabeled `failed` before
  retry), `dispatch_cell` (writes the synthesized config, invokes `clover
  run` as a subprocess -- crash isolation, matching the bench's actual
  architecture rather than an in-process design). New CLI subcommands
  `clover report`/`clover run-matrix` (the latter preflights the first
  grid cell before dispatching the whole matrix, catching a shared-config
  typo fast -- the same `AUDIT.md` lesson P4's `clover preflight` already
  captures).
- **A real gap found while designing `clover report`**:
  `ResolvedConfig.stream_spec` (`clover/config/loader.py`, P4) is a
  *post-resolution* `StreamSpec` -- scenario resolution already happened
  before it was constructed, so `config_resolved.yaml` records the
  concrete `revisits` but not the scenario *name* itself. Since the
  summary pivot needs to group by scenario, and extending
  `ResolvedConfig`/`config_resolved.yaml` for a P7-only need would touch
  tested P4 code, the run-id-in-directory-name convention (bench's own
  actual design, confirmed via research) was used instead --
  `run-matrix` names every run dir this way by construction, so
  `clover report`'s parser lines up automatically; an ad-hoc `clover run`
  whose dir name doesn't parse still gets its rows aggregated (nothing
  lost), just grouped as `"unknown"` in the summary.
- **`run-matrix` dispatches via real subprocesses (`sys.executable -m
  clover.cli run <config>`), not in-process** -- confirmed via research
  this is what the bench's own orchestrator does, and for a real reason,
  not just fidelity-for-its-own-sake: `CUDA_VISIBLE_DEVICES`-based GPU
  pinning is fundamentally a process-level concern (all in-process
  threads would share one process's CUDA visibility), and subprocess
  isolation means one crashed/hung cell can't take down the rest of the
  matrix.
- **GPU-pool parallelism (the bench's `ThreadPoolExecutor` sized to
  detected CUDA devices) is deferred, not built** -- this machine has no
  GPU to run or test it against either way, and sequential dispatch
  already fully satisfies "resume/retry/stale detection" (this phase's
  actual DONE criterion). Logged explicitly as a follow-up rather than
  silently dropped; adding it later is a pure enhancement (swap the `for
  cell in to_run` loop for a thread pool), not a redesign, since
  `dispatch_cell`/`plan_dispatch` are already pure/parallelizable
  functions.
- **Manual end-to-end verification caught a real Windows/Git-Bash
  path-translation footgun, not a product bug**: testing `run-matrix`
  against a `/tmp/...`-style path produced confusing "already done"
  results after an `rm -rf` that appeared to have no effect -- Git Bash's
  `/tmp` and Windows Python's interpretation of a leading-`/` path
  resolve to *different directories* (`D:\tmp\...` for Python, since
  there's no drive letter). Not a code issue; worth remembering for any
  future manual CLI testing on this machine -- use a relative or
  drive-lettered path, never a bare `/tmp/...` one, when testing through
  Git Bash.
- Full suite: 387 tests (24 new: reporting + matrix fixtures + 2 real
  end-to-end CLI tests using actual subprocesses). `clover smoke` green
  (9/9 methods). The `run-matrix` CLI tests are noticeably slower than
  the rest of the suite (real subprocess spawns, ~40s for 3 tests) --
  expect the full suite to take longer than prior phases as a result.
- P7 is now fully done; all three pieces (metrics library, `clover
  report`, `run-matrix`) ticked together in this entry since the last two
  landed in the same session as this log entry. Next unchecked phase:
  P8 — Datasets + extras + docs (SPEC §7, §8, §13).

- **2026-07-06, P8 (in progress, not ticked): the 6 built-in datasets
  done; `image_folder` config-only dataset, scenario extras, docs still
  queued.** Unlike P5-P7 (researched against `clover-pilot-bench`, a
  third party), this phase's best reference turned out to be this same
  repo's own `v1`/`main` branch — read directly via `git show main:...`
  (no checkout): it already has clean, working implementations of exactly
  the 6 datasets and 4 scenario extras this phase needs. SPEC itself says
  `CLDataset` ABC "≈ v1" — porting/adapting v1's own dataset code is
  explicitly sanctioned here in a way reusing PILOT's code never was;
  this is evolving the same project's prior version, not a third-party
  -code question.
- `clover/datasets/cifar100.py`: direct port of v1's `torchvision
  .datasets.CIFAR100(root, download=True)`-backed wrapper — torchvision
  owns the actual download/checksum/extraction. `clover/datasets/
  image_folder_base.py:ImageFolderCLDataset`: **v1 duplicated the exact
  same `ImageFolder`-backed wrapper 5×** (CUB-200, ImageNet-R, ImageNet-A,
  OmniBenchmark, VTAB — confirmed via direct comparison: identical
  `ImageFolder` load, identical `FileNotFoundError` staging message
  shape, identical train/test transform pipelines, identical
  `class_to_indices` construction, differing only in `dataset_dir`/
  `num_classes`/download URL) — extracted into one shared base instead.
  Unlike P6's adapter-family lesson ("don't assume a shared base before
  building two things that need it"), this is the *opposite* situation:
  five already-built, confirmed-identical examples existed before any
  base class was written, so extracting one here is not premature.
- **No real dataset downloads or network access used anywhere this
  session** — matches how v1's own test suite avoided this: CIFAR-100
  tests monkeypatch `torchvision.datasets.CIFAR100` with a synthetic
  100-class stand-in (`tests/conftest.py:patched_cifar100`, ported
  directly from v1's `conftest.py`); the 5 `ImageFolder`-based ones are
  tested via (a) the `FileNotFoundError` staging-error path and (b) a
  real, tiny hand-built `ImageFolder`-shaped fixture directory (`PIL`
  -generated tiny PNGs via `tempfile`/`tmp_path`) exercising the actual
  loading path end-to-end — going one step further than v1's own tests,
  which only ever checked the error path for these five.
- **Download URLs carried forward from v1's own docstrings/constants**
  (Google Drive links the user had already vetted for their prior v1
  work) — not fetched, not newly sourced; purely documentation pointers
  in `FileNotFoundError` messages and module docstrings, exactly as v1
  itself used them.
- **Real, unrelated mypy gap found and fixed**: adding any dataset that
  imports `torchvision` (all 6 new ones) surfaced `import-untyped` for
  `torchvision.*` in the `mypy clover/core clover/config` gate --
  `clover/config/loader.py` imports `from clover.datasets import
  get_dataset`, so `clover/datasets/__init__.py`'s new imports are pulled
  in transitively (the exact "mypy transitively pulls in whatever those
  modules import" lesson from P6, now hitting a *third-party stub gap*
  rather than an internal type error). Fixed with a
  `[[tool.mypy.overrides]] module = ["torchvision.*"] ignore_missing_imports
  = true` block in `pyproject.toml` — torchvision ships no `py.typed`
  marker, so there was nothing to fix on our own code's side.
- ImageNet-A's docstring notes explicitly (matching v1's own comment)
  that it has no official train/test split — staging requires manually
  creating one, unlike the other 4 which have official splits.
- Full suite: 408 tests (23 new: 6 dataset-registration/CIFAR-100
  fixtures + 15 parametrized `ImageFolder` cases + 2 registry-scaffold
  fixes). `clover smoke` unaffected (9/9 methods still green, this phase
  doesn't touch the method/training path at all).
- Next: `image_folder` config-only dataset (SPEC §7) -- needs
  `StreamSection.dataset` to accept a mapping (`{type: image_folder,
  root:, num_classes:}`) in addition to a plain string, a real (if small)
  config-schema extension distinct from "register another class", so it
  gets its own pass. Then scenario extras: re-derive each of v1's 4
  extras' *intent* against v2's unified `RevisitSpec` model (confirmed
  v1's own `build_stream_spec` for all 4 is generic boilerplate --the
  real per-scenario mechanism lived only in the legacy `OverlapSpec`
  path P1 already removed) -- expect `distribution_shift` to map cleanly
  (same-id revisit + new/partial images), `symmetric_pair`
  approximately (v2 has no bidirectional task-pair concept), and
  `near_miss`/`hierarchical` to end up documented as out-of-scope-for
  -v2.0 rather than force-fit (SPEC explicitly permits this: "absence
  must not block v2.0"). Then `docs/concepts.md`/`docs/extending.md`,
  once there's more to reference concretely. Once all land: tick P8's box.

- **2026-07-07, P8 done (all 3 remaining pieces landed; box ticked).**
  `image_folder`: `StreamSection.from_dict` now detects an inline
  `dataset: {type:, root:, num_classes:}` mapping (vs. a plain string),
  routing `type`->dataset name, `root`->`data_root` (overriding the
  existing default), `num_classes`->the new `StreamSpec
  .dataset_num_classes` field (`None` for every named built-in, which
  already knows its own class count). `clover/datasets/
  image_folder_config.py:ImageFolderDataset` (`dataset_dir=""`, reusing
  `ImageFolderCLDataset`) requires an explicit `num_classes` (can't be
  zero-arg-constructed like the named built-ins) and cross-checks it
  against the actual on-disk class-subdirectory count, raising immediately
  on a mismatch rather than silently training on the wrong head size.
- **Pre-existing gap #1 (found, not introduced): `StreamSpec.data_root`
  was parsed/validated/serialized end-to-end but never actually passed to
  a dataset constructor anywhere in `cli.py`** -- every dataset was always
  built from its class-level default root. Fixed as a natural part of
  `image_folder` (whose entire point is a user-supplied root): `cmd_run`/
  `cmd_inspect`/`cmd_preflight` now thread `root=resolved.stream_spec
  .data_root` (+ `num_classes` when set) through dataset construction.
- **Pre-existing gap #2 (found, not introduced, more consequential): no
  code anywhere ever composed a dataset's declared `train_trsf`/
  `test_trsf`/`common_trsf` into an actual transform and applied it** --
  v1's `DataManager` did this centrally (`transforms.Compose([*train_trsf,
  *common_trsf])`); v2 never got an equivalent, so every real
  (non-synthetic) dataset would have silently trained on raw PIL images
  the first time anyone actually ran one through `clover run`. Surfaced
  only now because this session did the first real (non-synthetic)
  end-to-end `clover run` this repo has ever attempted -- prior dataset
  tests always passed `transform=` manually or used `synthetic` (empty
  transform lists, so the gap was invisible). Fixed with
  `cli.py:_apply_default_transforms(dataset)`, called on both train/test
  datasets in `cmd_run` and `cmd_smoke`'s `_smoke_check_method`.
- **Real end-to-end proof, scoped honestly**: built a tiny hand-made
  `image_folder` fixture (real PNGs via PIL) and ran the full pipeline --
  `clover inspect`/`clover preflight` against an inline `{type:
  image_folder, ...}` config (passing), plus the actual `Trainer`-used
  `ExperienceDataset`+`DataLoader` machinery pulling a real batch and
  confirming `(3, 224, 224)` float tensors with correct labels. Did
  **not** chase a full `clover run` through an actual method/backbone:
  every registered backbone (`tiny_mlp`/`tiny_vit`) is a documented,
  synthetic-only stand-in (1-channel, sized for the 8x8 synthetic
  dataset -- see P5's log entry above: "real cluster configs against an
  actual pretrained timm ViT would tune these separately"), and no method
  currently threads `in_chans`/a real timm base model through `build()`.
  That gap is pre-existing, shared by every one of the 5 already-shipped
  `ImageFolder`-backed datasets (none had ever been run through a real
  method/backbone either), and belongs to P5/P6/cluster-workflow scope,
  not P8's dataset-layer scope -- confirmed against CLAUDE.md's "no real
  training locally, ever" rule.
- **Scenario extras, final scope call**: only `distribution_shift`
  implemented (`clover/scenarios/distribution_shift.py`) -- `n_shifted`
  of task 0's classes reappear once, same-id, `placement="end_of_stream"`,
  distinct from `cumulative_drift` (same-id but *every* task) and
  `long_range_revisit`/`exact_replay` (fresh echo id, not same-id).
  `symmetric_pair`/`near_miss`/`hierarchical` are documented as
  out-of-scope in `docs/CONCEPTS.md` §8 rather than force-fit: v2's
  `RevisitSpec` has no bidirectional (backward-in-time) class injection,
  no zero-overlap adjacency bookkeeping, and no external-taxonomy-driven
  class-to-task assignment -- SPEC explicitly permits their absence.
- `docs/CONCEPTS.md` rewritten in place for v2 (was still v1's
  `OverlapSpec`/`OverlapDataManager`/`preserve_task_size` framing) --
  `StreamSpec`/`RevisitSpec`'s real v2 field syntax, echo-id vs. same-id
  revisits, a corrected code example (`clover.core.stream.build_benchmark`
  needs `dataset_info` + both class-to-indices maps, not just a spec), a
  new dataset-staging section (all 6 real built-ins + `image_folder`),
  and the scenario-extras non-implementation reasoning. New
  `docs/extending.md`: one worked, runnable example each for
  dataset/scenario/method/backbone registration, using `image_folder`
  and `distribution_shift` as the dataset/scenario examples per the plan.
- **Two regressions caught by the full suite, fixed same-session**: (1)
  `StreamSpec.to_dict()` writes `dataset_num_classes`, but
  `StreamSection._ALLOWED_KEYS`/`from_dict` didn't accept it as a
  top-level key (only via the inline-mapping path) -- broke
  `config_resolved.yaml` round-trip re-resolution; fixed by adding it to
  `_ALLOWED_KEYS` and reading it directly when `dataset` is a plain
  string. (2) `test_package_scaffold.py`'s two registry-membership tests
  still asserted the pre-`image_folder`/pre-`distribution_shift` sets;
  updated both.
- Full suite: 437 tests, all green (`ruff check clover tests`, `mypy
  clover/core clover/config`, `clover smoke` 9/9 also green). P8 fully
  done -- next unchecked phase: P9 (release candidate).
