# CLAUDE.md — CLOVER (clover-cl)

Public continual-learning benchmark for class-revisit scenarios.
v2 is being built per `SPEC.md` — final, all design decisions confirmed
2026-07-05. **Start every session with `PLAN.md`**: implement the first
unchecked phase (read its SPEC.md sections first), verify its DONE criteria,
tick it off. Audit context lives in `../AUDIT.md`. v1 stays on the `v1`
branch; v2 work happens on the `v2` branch. v2 is a clean breaking release:
no v1 compat shim, migration covered by `docs/MIGRATION.md`.

## Workspace map (D:\Dev\Research)

- `clover-cl/` — THIS repo. The only repo you modify.
- `clover-pilot-bench/` — private bench harness. **READ-ONLY** reference for
  lessons learned (adapter, metrics, orchestrator, configs).
- `_ref_pilot/` + `clover-pilot-bench/third_party/pilot` — LAMDA-PILOT.
  **READ-ONLY**. Inspiration/citation only.
- Before starting v2 work: `git pull` here (local clone can lag origin/main).

## v2 package layout (see SPEC.md §2)

`clover/core` (spec→plan→experiences) · `clover/datasets` · `clover/scenarios`
· `clover/methods` (incl. `losses.py` — the only place loss masking lives) ·
`clover/backbones` · `clover/training` · `clover/evaluation` · `clover/config`
· `clover/cli.py` · `tests/` · `configs/` · `docs/`

## Tech stack

- Python ≥ 3.10, torch ≥ 2.0, torchvision, timm ≥ 1.0, numpy, pyyaml, pillow.
- Dev: pytest, pytest-cov, ruff, mypy. Keep runtime deps to exactly this
  list — no Hydra/OmegaConf/pydantic/lightning.

## Coding standards

- Dataclasses + explicit `validate()`; reject unknown config keys with
  actionable messages ("did you mean").
- Pure functions for planning/assignment: rng and data in, plan out. No I/O
  or global state in `clover/core`.
- Google-style docstrings on public API; type hints everywhere; ruff + mypy
  clean before commit.
- Label-space logic uses set membership via `LabelSpaceView`
  (`view.new_mask(targets)`), NEVER range arithmetic
  (`targets >= known_classes`, `logits[:, :k] = -inf`).
- Registries via decorators (`@register_method`, `@register_dataset`,
  `@register_scenario`, `@register_backbone`); core never imports a plugin
  by name. Backbone base models are config-selectable (timm/HF names or a
  user class) — never hard-code a checkpoint name inside a method.
- Every stochastic choice flows through a seeded generator passed in
  explicitly; no bare `np.random.*` / `random.*` in library code.

## Environment: NO GPU locally

- This machine has no GPU. Never launch real training locally — it will
  hang or crawl. Everything local runs on the synthetic dataset via the
  smoke profile: `clover run cfg.yaml --profile smoke` / `clover smoke`.
- Real runs are SLURM jobs on the Snellius cluster:
  `sbatch scripts/slurm_run.sh <config.yaml>`. Workflow: smoke locally →
  sbatch on cluster → `clover report` on synced results.

## Testing rules

- Run: `pytest tests/ -v` (all CPU; no GPU or dataset downloads in tests).
- Lint/type: `ruff check clover tests` and `mypy clover/core clover/config`.
- New method/dataset/scenario ⇒ registered in tests so the parametrized
  suites (incl. the revisit-safety gate) pick it up automatically.
- The revisit-safety gate (every method × disjoint/same-id/echo synthetic
  streams ⇒ finite loss, finite params) must stay green; never mark it skip.
- Golden fixtures (seed-1993 class order, per-scenario plans) are contracts:
  if a change breaks one, stop and justify — don't regenerate silently.
- Coverage ≥ 85% on core/config/losses.

## What NOT to do

- NO imports from LAMDA-PILOT / `third_party/*`; no copied PILOT code, ever.
  Cite it in docs instead.
- NO per-method loss patches: revisit handling comes from
  `clover/methods/losses.py` + `LabelSpaceView` only. A method writing its
  own `-inf` logit masking is a bug even if tests pass.
- NO per-method `_known_classes` bookkeeping — the stream/experience is the
  single source of truth for label-space state and head sizes.
- NO config sprawl: one YAML per run, one schema per plugin, layered
  defaults; no interpolation languages, no scattered ad-hoc keys, no
  duplicated dataset metadata (num_classes lives on the dataset class only).
- NO edits to `clover-pilot-bench/` or any pilot checkout.
- NO v1 compat layer: do not create `clover/compat`, do not carry the
  legacy `OverlapDataManager` API, `int|list` unions, or PILOT calling
  conventions anywhere — legacy users get `docs/MIGRATION.md` and the `v1`
  branch.
- NO reintroduction of removed v1 traps: `preserve_task_size` eviction,
  label-hash-only feature cache keys (keys must include the manifest hash).
- Don't commit datasets, weights (`*.pt`), or run outputs; don't push or tag
  releases unless asked.

## Useful commands

- Local smoke (the only training this machine does):
  `clover run configs/<name>.yaml --profile smoke` · all methods: `clover smoke`
- Cluster run: `sbatch scripts/slurm_run.sh configs/<name>.yaml`
- Inspect a stream (no training): `clover inspect configs/<name>.yaml`
- Preflight before long runs: `clover preflight configs/<matrix>.yaml`
- (While the CLI is unbuilt: `python -m clover.cli ...`)
- v1 reference tests on the `v1` branch: `pytest tests/ --cov=clover`
