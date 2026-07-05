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
- [ ] **P4 — Config + CLI + cluster workflow** (SPEC §9, §11): typed YAML schemas with unknown-key rejection and layered defaults, `config_resolved.yaml`, run-dir artifacts; CLI `run` / `smoke` / `inspect` / `preflight`; smoke profile; `scripts/slurm_run.sh`. DONE: typo'd config fails pre-data naming the key; `clover run <cfg> --profile smoke` and `clover smoke` green on GPU-less CPU in <10 min; sbatch template submits and resumes on Snellius with only `#SBATCH` placeholders filled.
- [ ] **P5 — Backbones + prompt trio** (SPEC §6.4): backbone registry with config-selectable base models (timm/HF name or user class), prompt-pool / prefix / CODA wrappers; L2P, DualPrompt, CODA-Prompt using core loss policies (no hand-rolled `-inf` masking — lint test enforces). DONE: all three pass safety gate + smoke; finite loss and sane accuracy on all 6 scenarios (incl. same-id cumulative_drift) on the synthetic stream; disjoint CIFAR-100 ≈ published per method.
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
