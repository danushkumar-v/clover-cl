# PLAN — CLOVER v2 implementation checklist

Source of truth for scope/design: `SPEC.md` (final, 2026-07-05); constraints:
`CLAUDE.md`; background: `../AUDIT.md`.

**Session protocol:** read this file, implement the *first unchecked* phase
(read its SPEC.md sections first), verify its DONE criteria actually pass,
tick the box, commit. Do not start a later phase while an earlier one is
unchecked. If a phase must deviate from SPEC.md, record the deviation under
"Log" below before ticking.

- [x] **P0 — Scaffold** (SPEC §2): `git pull` to origin/main, branch `v2`; package layout, decorator registries (methods/datasets/scenarios/backbones), pyproject (py≥3.10, torch≥2.0, timm≥1.0), ruff+mypy+pytest config, CI workflow skeleton. DONE: CI green on the empty package (lint + typecheck + pytest collecting 0-or-trivial tests).
- [ ] **P1 — Stream model** (SPEC §3–§4): StreamSpec v2 (`label: same|new`, `images: same|new|partial`, `task_size: fixed|grow`) + validation, planner → serializable StreamPlan, assignment port, Experience/Stream/Benchmark refactor; legacy `OverlapDataManager` API removed, no shim. DONE: golden-plan tests for the 6 core scenario shapes + seed-1993 class-order fixture green; plan/manifest round-trip test green; validation rejects each documented failure mode with actionable message.
- [ ] **P2 — Label space + loss policies** (SPEC §5): LabelSpaceView on Experience; `methods/losses.py` (`new_class_ce`, `seen_class_ce`, `masked_logits`); synthetic dataset + 2-layer backbone stub; revisit-safety gate with a stub method. DONE: gate fails on a deliberately v1-L2P-broken stub (unmasked `-inf` CE) and passes on the fixed stub, across disjoint / same-id / echo synthetic streams.
- [ ] **P3 — CLMethod + Trainer + SimpleCIL** (SPEC §6.1–6.2): CLMethod ABC, TrainContext, incremental heads, core Trainer (seeding, loop, checkpoints/resume, status.json, per-class evaluator hookup); SimpleCIL end-to-end. DONE: SimpleCIL passes safety gate + smoke on CPU; interrupted-run resume test green; CIFAR-100 disjoint sanity run queued/verified on cluster ≈ published range (record in docs/methods.md).
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
