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
- [ ] **P6 — Remaining methods** (SPEC §6.5): APER-Adapter, EASE, RanPAC, MOS, TUNA (adapter wrappers as needed); hyperparameter defaults carried from bench configs with provenance notes. DONE: each passes safety gate + smoke + disjoint CIFAR-100 sanity vs. published range; comparison table in docs/methods.md covers all 9.
- [ ] **P7 — Metrics + reporting + matrix** (SPEC §10): per-class evaluator finalized, standard (A_t/AIA/BWT/FWT/Forgetting) + CLOVER (RAG, Repetition Gain, Anchor/Long-Range Retention, echo-aware) metrics ported with fixture tests; `clover report`; `run-matrix` orchestrator (resume/retry/stale/GPU-pool). DONE: hand-computed metric fixtures green; two-method synthetic demo report renders in CI; matrix resume test green.
- [ ] **P8 — Datasets + extras + docs** (SPEC §7, §8, §13): CUB-200, ImageNet-R/A, OmniBenchmark, VTAB wrappers + staging docs; `image_folder` config-only dataset; scenario extras as capacity allows; docs/concepts.md, docs/extending.md (method/dataset/scenario/backbone worked examples). DONE: built-in metadata smoke tests green; a tutorial-followed custom dataset runs a full smoke benchmark without touching core.
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
- Next: EASE (`clover/backbones/adapter_ease.py` + `clover/methods/
  ease.py`) -- growing per-experience frozen adapter list, growing-dim head
  with cosine-similarity cross-block reweighting. See handoff.md for the
  full per-method design (EASE/MOS/TUNA) agreed in this session's plan.
