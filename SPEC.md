# CLOVER v2 — Specification

Status: **final — ready for implementation.** Written 2026-07-05 from the
Phase-1 audit (`../AUDIT.md`); all open questions in AUDIT.md §6 were answered
by the maintainer on 2026-07-05 and the decisions are baked in below. Sections
are ordered for section-by-section implementation; each ends with acceptance
criteria.

---

## 1. Vision and product requirements

CLOVER v2 is a **self-contained, public, extensible continual-learning
benchmark framework** for class-revisit scenarios. One simple config file +
one command runs a full experiment: dataset → overlap scenario → CL method →
training → metrics → report.

Hard requirements:

- **R1 — Self-contained.** Zero code dependency on LAMDA-PILOT. PILOT is
  cited as inspiration for method implementations; no vendored/forked code.
- **R2 — Config-driven.** A user edits one human-readable YAML (dataset,
  method, scenario, training params) and runs `clover run config.yaml`.
- **R3 — Plugin architecture.** New methods, datasets, and overlap mechanisms
  are added by implementing a small documented interface and registering it —
  never by editing core code.
- **R4 — Correctness by construction.** Label-space and loss handling for
  revisited classes is defined once in the core and inherited by all methods.
  A method that would NaN on a revisit cannot pass CI.
- **R5 — Benchmark hygiene.** Seeds, logged resolved configs, manifests,
  unit + integration tests, GitHub Actions CI, results reporting, clear README
  and contributor docs.
- **R6 — Ship with**: 6 core scenarios, 6 prebuilt datasets + custom dataset
  registration, and all 9 methods — SimpleCIL, APER-Adapter, L2P, DualPrompt,
  CODA-Prompt, EASE, RanPAC, MOS, TUNA — with L2P/DualPrompt/CODA *properly
  fixed* via the core loss policy. Methods **and backbones** are fully
  pluggable: documented `CLMethod` interface + decorator registry, backbone
  selectable from config (timm/HF model names or a user-provided class) —
  users add arbitrary methods/models without touching core.
- **R7 — CPU-first development workflow.** A `smoke` run profile executes any
  config end-to-end on a tiny synthetic dataset, CPU-only, in minutes (finite
  loss required). Real runs are SLURM batch jobs on a GPU cluster
  (`scripts/slurm_run.sh`, parameterized by config path). The README
  documents the smoke-locally → submit-to-cluster workflow. Development
  machines are assumed to have **no GPU**.

Non-goals for v2.0: rehearsal/replay-memory methods, non-vision modalities,
distributed training, hyperparameter search, v1 API compatibility (see §4).

---

## 2. Repository layout

Developed on a `v2` branch of `clover-cl`; current `main` is preserved as a
`v1` branch at release time. Before branching: `git pull` the standalone
clone (it lags origin/main by the echo-class commits).

```
clover/
├── core/            stream model: specs, task/image assignment, experiences
│   ├── spec.py          StreamSpec v2 + RevisitSpec + validation (§3)
│   ├── plan.py          StreamPlan: resolved, concrete, serializable (§3.3)
│   ├── planner.py       spec → plan (placement, echo-id allocation, image split)
│   ├── assignment.py    pure image→experience assignment (v1 image_assigner)
│   ├── experience.py    Experience + LabelSpaceView (§5)
│   ├── stream.py        Stream, Benchmark
│   └── manifest.py      reproducibility manifests
├── datasets/        CLDataset ABC, registry, 6 built-ins (§7)
├── scenarios/       scenario plugins: spec factories (§8)
├── methods/         method plugins (§6)
│   ├── base.py          CLMethod ABC + lifecycle contract
│   ├── heads.py         incremental classifier heads
│   ├── losses.py        THE revisit-safe loss policies (§5)
│   └── <method>.py      one module per method
├── backbones/       registry + ViT wrappers: prompt-pool, prefix, adapter (§6.4)
├── training/        core-owned Trainer: loop, AMP, seeding, checkpoints (§6.2)
├── evaluation/      per-class evaluator, R-matrix, metrics (§10)
├── config/          schema, loading, validation, resolution (§9)
├── cli.py           `clover` entry point: run / smoke / inspect / preflight / report
└── utils/
configs/             shipped example configs (one per scenario × dataset)
scripts/slurm_run.sh SLURM template: `sbatch scripts/slurm_run.sh <config>` (§11.1)
tests/               unit + integration + revisit-safety gate (§12)
docs/                concepts, plugin how-tos, method provenance, MIGRATION.md (§13)
.github/workflows/   CI (§12.4)
```

---

## 3. Unified stream model

v1 has two drifting overlap mechanisms (same-id `OverlapPair` vs. echo
classes) reachable through two different APIs. v2 unifies them: **one spec, in
which "what label does a returning category get" is an axis, not a mechanism.**

### 3.1 `StreamSpec` (the only user-facing spec)

```yaml
stream:
  dataset: cifar100
  init_cls: 10
  increment: 10
  task_size: fixed            # fixed | grow
  revisits:
    - classes: task0           # task0 | [ids] | {random: N}
      placement: end_of_stream # random | spaced | end_of_stream | clustered | every_task
      label: new               # new (echo id) | same (original id)
      images: new              # same | new | partial:<pct>
      min_gap: 3
  shuffle_seed: 1993
  stream_seed: 42
```

Semantics:

- `label: new` → returning category re-presented under a fresh label id
  (v1 "echo"). Compatible with `task_size: fixed` — the revisit occupies
  ordinary new-class slots.
- `label: same` → returning category keeps its id (v1 `OverlapPair`;
  cumulative-drift anchors). With `task_size: fixed`, same-id revisits *count
  toward* the task's class budget (no eviction — the v1
  `preserve_task_size=True` eviction behaviour is removed; it produced
  out-of-range labels and is unusable with CIL heads).
- `images: same | new | partial:<pct>` replaces v1's
  `duplicate / disjoint / partial_duplicate` ImageSplit, per-revisit instead
  of global.
- `placement: every_task` covers cumulative-drift anchors.

The six core scenarios are expressible in this one schema; scenario plugins
(§8) are just named factories that emit a `StreamSpec`.

### 3.2 Validation

`StreamSpec.validate()` must reject, with actionable messages: unknown keys
(typo protection), unsatisfiable placements (min_gap vs. stream length —
carry over v1's feasibility checks), class-budget overflows for fixed-size
streams, empty experiences, and any experience whose labels would exceed the
head size implied by first appearances (the invariant the bench adapter
enforced defensively).

### 3.3 `StreamPlan` — the resolved artifact

`planner.resolve(spec) -> StreamPlan`: a fully concrete, seed-resolved,
JSON-serializable object: per-experience class lists, echo table
`{echo_id: (source_id, image_relation)}`, per-class image-index assignments,
head-size schedule. Properties:

- Deterministic given (spec, dataset metadata).
- The manifest **is** the serialized plan + header (version, spec, seeds).
- `StreamPlan.from_manifest(path)` reconstructs a run's data exactly —
  reviewers can re-run a published experiment from its manifest alone.

The v1 engine internals (`task_builder`, `image_assigner`) survive as the
implementation of `planner`/`assignment`, refactored to emit a plan instead
of feeding a manager object directly.

Acceptance: every v1 paper scenario reproduced as a spec; plan serialization
round-trips; validation rejects each documented failure mode with a clear
message; disjoint spec reproduces v1's PILOT-equivalent class order
(seed-1993 fixture test — keep the frozen fixtures even though PILOT itself
is gone).

---

## 4. Data layer

Carries over from v1 with refactors, not a rewrite:

- `Experience` keeps v1's metadata and gains `label_space` (§5) and
  `echo_map`. Frozen dataclass, invariant-checked in `__post_init__`.
- `Stream`/`Benchmark` as in v1. **No compat shim**: the legacy
  `OverlapDataManager` API is dropped entirely. v2 is a clean breaking
  release — v1 stays available on the `v1` branch, and `docs/MIGRATION.md`
  maps every legacy API/config construct to its v2 equivalent (§13). Optimize
  the API for new public users, not legacy callers.
- Datasets own their input pipeline: a dataset declares `input_size` and
  transform presets, so ViT methods request `224` and CIFAR provides it —
  the bench's `_Resize224Dataset` wrapper disappears.
- Single source of dataset metadata: `num_classes` etc. come from the
  registered dataset class only (v1's duplicated `_n_classes_map` in
  `stream_builder` is removed).

Acceptance: all v1 core tests ported and green; no module anywhere mentions
PILOT calling conventions (`np.arange` class-range queries, `int|list`
unions, `get_task_size` semantics).

---

## 5. Label space and revisit-safe loss (the centerpiece)

### 5.1 `LabelSpaceView`

Every `Experience` carries a `LabelSpaceView` computed by the core:

```python
view.new_classes        # first-appearance ids in this experience (incl. echo ids)
view.seen_classes       # ids seen in experiences 0..t-1
view.revisiting_classes # same-id returns present in this experience
view.head_size          # required classifier width after this experience
view.new_mask(targets)  # bool tensor: label ∈ new_classes
view.old_mask(targets)  # bool tensor: label ∈ seen_classes
view.logit_mask(width)  # bool [width]: which logit columns are valid *now*
```

Set-membership, never range arithmetic: `targets >= known_classes` is
forbidden everywhere, because with same-id revisits "old" ids appear in
current batches and class ids are not guaranteed contiguous per task.

### 5.2 Loss policies (`methods/losses.py`) — defined once, inherited by all

- `new_class_ce(logits, targets, view)` — CE restricted to new-class
  *samples* AND new-class *logit columns*, consistently; returns a zero-loss
  connected to the graph when the batch has no new-class samples. This is
  the generalization of the EASE/TUNA/MOS patches and the correct form of
  the L2P/DualPrompt/CODA "mask old logits" intent.
- `seen_class_ce(logits, targets, view)` — CE over all seen classes
  (methods that legitimately train on revisits, e.g. drift anchors).
- `masked_logits(logits, view)` — returns logits with invalid columns at
  `-inf` *and* an accompanying sample mask; the two can only be obtained
  together, so the v1 bug (mask columns, keep old-labelled samples) is
  unrepresentable through the API.

Methods never write `logits[:, :k] = -inf` by hand — enforced by convention,
code review, and a grep-based lint test in CI.

### 5.3 The revisit-safety gate (CI, structural guarantee)

A parametrized integration test runs **every registered method** on a tiny
synthetic stream (§12.2) with three shapes: disjoint, same-id revisit
mid-stream, echo revisit at end. Asserts after each task: loss is finite,
parameters are finite, and accuracy on the revisited classes is above chance.
A new method plugin inherits this test automatically via the registry — R4 is
enforced by CI, not by documentation.

Acceptance: gate red when a deliberately broken method (upstream-style L2P
loss) is registered; green for all shipped methods.

---

## 6. Method layer

### 6.1 `CLMethod` interface (plugin surface)

```python
@register_method("l2p")
class L2P(CLMethod):
    name = "l2p"
    default_backbone = "vit_prompt_pool/vit_base_patch16_224"  # overridable in config
    cacheable_features = False       # declared, not guessed (bench lesson)

    def build(self, stream_info, cfg): ...          # networks, heads
    def before_experience(self, exp, ctx): ...     # expand head, new prompts...
    def train_experience(self, exp, loader, ctx): ...  # inner epochs; uses losses.*
    def after_experience(self, exp, ctx): ...      # prototypes, merges, freezes
    def classifier(self) -> nn.Module: ...          # for the shared evaluator
```

- `ctx` (`TrainContext`) provides device, AMP policy, optimizer/scheduler
  factories from config, logging, and `exp.label_space` — methods do not
  track `_known_classes` themselves; the stream is the source of truth.
- The head-size schedule comes from `exp.label_space.head_size`; incremental
  head expansion is implemented once in `methods/heads.py`.
- `cacheable_features` + a `frozen_backbone` property drive the feature
  cache (§6.3) explicitly instead of monkey-patching.

### 6.2 Core `Trainer` (owned by the framework, not by methods)

Replaces PILOT's per-method self-contained loops. Responsibilities: seeding
(torch/numpy/random from config), experience iteration, dataloader
construction, AMP (BF16 autocast as a config flag — no forward-wrapping),
per-experience checkpoint/resume (bench-proven design: `status.json`
heartbeat, atomic checkpoint files), calling the evaluator, and writing run
artifacts (§11). Methods keep their *inner* epoch loop in
`train_experience` — CL methods differ too much to share one epoch loop —
but everything around it is core.

### 6.3 Feature cache

Optional, for frozen-backbone methods only (`cacheable_features=True`).
Key = (backbone id, dataset, **manifest hash**, split) — the manifest hash
scopes the cache to the exact image assignment, structurally fixing the
cross-scenario pollution bug.

### 6.4 Backbones (`clover/backbones/`) — pluggable, config-selectable

Backbones are a plugin surface of their own (R6), split into two orthogonal
parts:

1. **Base model** — any timm (or HF-hub-via-timm) model name, or a
   user-provided class:

   ```yaml
   method:
     name: l2p
     backbone:
       source: timm                      # timm | class
       model: vit_base_patch16_224       # any timm/HF name with compatible geometry
       # or: source: class, path: mypkg.models.MyViT
   ```

2. **Mechanism wrapper** — what the method does to the base model, registered
   via `@register_backbone("vit_prompt_pool")`: identity/frozen (SimpleCIL,
   RanPAC, APER base), prompt-pool (L2P), prefix/dual-prompt (DualPrompt),
   CODA prompt attention, adapter (APER-Adapter, EASE, MOS, TUNA variants).
   A wrapper declares the hooks it needs (e.g. per-block attention access);
   incompatibility with a chosen base model fails at config validation, not
   mid-run.

Methods declare a `default_backbone` but any config can override the base
model, so users can benchmark new ViT checkpoints (or their own class)
without touching core. Clean-room implementation on timm ≥ 1.0; non-strict
checkpoint loading where heads are dropped (audit §3.1). Each wrapper
documents which paper/official repo defines the mechanism; cite LAMDA-PILOT +
original method papers, copy no code (R1).

### 6.5 Method implementation order

1. **SimpleCIL** (frozen ViT + prototype head — smallest; validates trainer,
   evaluator, heads end-to-end)
2. **L2P**, then **DualPrompt**, **CODA-Prompt** (prompt family with the
   §5 loss policy — the headline fix)
3. **APER-Adapter**, **EASE** (adapter family)
4. **RanPAC**, **MOS**, **TUNA**
Each lands with: unit tests, revisit-safety gate green, a passing `smoke`
profile run (CPU, minutes — R7), and a disjoint CIFAR-100 sanity run (on the
cluster) within the published accuracy range. The validation bar is
*faithful-to-the-published-method*, not bit-reproduction of the v1 bench
numbers; the comparison against published accuracies is recorded in a table
in `docs/methods.md`. Hyperparameter defaults carried from the bench's
verified `configs/methods/*.yaml` (provenance: PILOT `exps/*.json`), recorded
in `docs/methods.md`.

---

## 7. Dataset layer

`CLDataset` ABC ≈ v1 (`num_classes`, `class_to_indices`, `use_path`,
transform presets, `input_size`) plus explicit download/staging docs.
Built-ins: CIFAR-100, CUB-200, ImageNet-R, ImageNet-A, OmniBenchmark, VTAB.

Custom registration:

```python
@register_dataset("my_plants")
class MyPlants(CLDataset): ...
```

plus a config-only path for folder datasets
(`dataset: {type: image_folder, root: ..., num_classes: ...}`). Registries
(datasets, methods, scenarios) share one implementation with decorator
registration; third-party packages can expose plugins via the
`clover.plugins` entry-point group.

Acceptance: a tutorial-followed custom dataset runs a full benchmark without
touching clover source; built-ins pass a metadata smoke test (counts, shapes).

---

## 8. Scenario layer

A scenario plugin is a named factory returning a `StreamSpec`:

```python
@register_scenario("exact_replay")
def exact_replay(dataset_info, init_cls, increment, seed, **params) -> StreamSpec: ...
```

Core six (paper set, confirmed): `disjoint_baseline`, `exact_replay`,
`partial_overlap` (fraction param), `long_range_revisit`,
`mid_range_revisit`, `cumulative_drift`. Cumulative-drift identity is
**confirmed by design**: anchors keep their same label ids with disjoint
images per task — identity persistence *is* the scenario's purpose, and the
core loss policy (§5) is what makes it trainable for every method, including
the prompt family. v1's extra scenarios (`hierarchical`, `near_miss`,
`distribution_shift`, `symmetric_pair`) port to `clover/scenarios/extras/`
opportunistically; their absence must not block v2.0.

Because scenarios emit specs, a new *overlap mechanism* (not just a new
combination) is added by extending `RevisitSpec`'s axes in core — documented
in `docs/extending.md` with the invariants (§3.2) any extension must keep.

Acceptance: each core scenario has a golden-plan test (spec → plan → expected
class lists/echo table for a small synthetic dataset) and an overlap-matrix
visual in docs.

---

## 9. Configuration system

**Decision (confirmed): plain YAML + typed dataclass schema validation,
unknown-key rejection. No Hydra/OmegaConf/CLI-DSL.** v1's instinct
(dataclasses + `from_yaml`) was right; v2 adds rigor.

One run = one file:

```yaml
# configs/cifar100_exact_replay_l2p.yaml
run:
  name: l2p_exact_replay        # optional; defaults to file stem
  seed: 1993                    # master seed (torch/numpy/random)
  output_dir: runs/
stream:                         # §3.1 — dataset + scenario
  dataset: cifar100
  scenario: exact_replay        # or an inline revisits: block
  init_cls: 10
  increment: 10
  stream_seed: 42
method:
  name: l2p                     # registry key
  # method-specific keys below are validated against the method's schema
  prompt_pool_size: 10
  prompt_length: 5
training:
  epochs: 5
  batch_size: 16
  optimizer: {name: adam, lr: 1.875e-3}
  amp: bf16
```

Rules:

- **Unknown keys are errors** (bench lost 30 cluster runs to guessed keys).
  Each method/dataset/scenario declares a typed schema (dataclass fields);
  the loader validates section by section and reports all errors at once
  with "did you mean" suggestions.
- **Layered defaults, visible resolution**: framework defaults ← method
  defaults ← user file. The fully resolved config is printed on start and
  written to the run dir (`config_resolved.yaml`). No includes, no
  interpolation language; at most a documented `defaults: <method>` pull-in.
- Matrix runs (`clover run-matrix matrix.yaml`) reuse the same schema with
  lists over methods/scenarios/datasets/seeds — the bench's orchestrator
  (resume, retry, stale detection, GPU pool) is ported, not reinvented.
- **Smoke profile (R7)**: `clover run config.yaml --profile smoke` overrides
  the stream onto the built-in synthetic dataset (§12.2) with a minimal
  training budget while keeping the chosen method, scenario shape, and code
  paths. `clover smoke` (no config) runs *every registered method* through
  the smoke profile — the pre-cluster gate. Both are CPU-only and finish in
  minutes; a smoke run failing (non-finite loss, crash) blocks submission.

Acceptance: a config with one typoed key fails before any data loads, naming
the key; `config_resolved.yaml` from any run re-runs identically; `clover
smoke` passes on a GPU-less machine in under ~10 minutes.

---

## 10. Evaluation and metrics

- Evaluator is core-owned and **per-class-native**: predictions over the test
  stream → per-class accuracy → R-matrix cells as means over each
  experience's first-appearance classes (bench-proven; exact for non-uniform
  tasks).
- Standard metrics: A_t, AIA (average incremental accuracy), BWT, FWT,
  Forgetting — ported from `bench/metrics/standard.py` with their tests.
- CLOVER metrics: Repetition Gain, RAG, Anchor Retention, Long-Range
  Retention — ported from `bench/metrics/overlap.py`; echo-aware (returning =
  echo ids + same-id revisits; retention read on echo *source* ids).
- `clover report <runs...>`: aggregates per-run artifacts (authoritative)
  into the long CSV + a summary table; aggregate files are always
  rebuildable from per-run data (bench lesson).

Acceptance: metric unit tests with hand-computed fixtures; a two-method demo
report renders in CI.

---

## 11. Run artifacts and reproducibility

Per run directory (bench-proven layout):
`manifest.json` (= serialized StreamPlan + header), `config_resolved.yaml`,
`status.json` (state/heartbeat/last task), `per_task.csv`, `R_matrix.npy`,
`ckpt_task*.pt` (resumable), `log.txt`.

Seeding: one master `run.seed` fans out to torch/cuda/numpy/random;
`stream_seed`/`shuffle_seed` remain independent so data streams and training
stochasticity vary independently (keep v1's two-seed design). Determinism
flags (`cudnn.deterministic`) on by default; `cudnn_benchmark` opt-in and
recorded.

### 11.1 Cluster workflow (SLURM)

Development machines have no GPU; every real run is a SLURM batch job.
`scripts/slurm_run.sh` is a shipped template parameterized by config path:

```
sbatch scripts/slurm_run.sh configs/cifar100_exact_replay_l2p.yaml
sbatch scripts/slurm_run.sh configs/matrix_full.yaml   # run-matrix mode
```

Template properties: account/partition/time as `#SBATCH` variables at the top
(placeholders in the public repo; site-specific values documented in a
comment), module/venv activation hook, resumability via the run dir's
checkpoints (a resubmitted job continues, mirroring the bench's orchestrator
behaviour), and logs into the run directory. The README documents the
workflow explicitly: **1)** edit config → **2)** `clover run cfg.yaml
--profile smoke` locally (CPU) → **3)** `sbatch scripts/slurm_run.sh
cfg.yaml` on the cluster → **4)** `clover report runs/...` on the synced
results.

Acceptance: same config + seeds → identical manifest and (CPU) identical
metrics; a run interrupted mid-stream resumes to the same final numbers; the
SLURM template runs unmodified on Snellius apart from filling the `#SBATCH`
placeholders.

---

## 12. Testing and CI

1. **Unit**: spec validation, planner placement, image assignment, label-space
   views, loss policies, heads, metrics. Pure functions, CPU, fast.
2. **Synthetic integration**: a built-in `synthetic` dataset (e.g. 20 classes
   × 32 samples of 8×8 noise-with-signal, a 2-layer backbone stub) runs every
   method × three stream shapes on CPU in seconds — hosts the revisit-safety
   gate (§5.3), the `smoke` profile (§9), and trainer/resume tests. CI's
   integration stage is literally `clover smoke` plus assertions.
3. **Golden fixtures**: seed-1993 class order and per-scenario plans frozen as
   JSON; guards refactors (replaces the PILOT byte-equivalence tests after the
   dependency is gone).
4. **CI (GitHub Actions)**: lint (ruff) + type check (mypy, core modules) +
   unit + synthetic integration on CPU for pushes/PRs; a manual/nightly job
   may run one real CIFAR-100 smoke run if a GPU runner exists. Coverage
   gate ≥ 85% on `clover/core`, `clover/config`, `clover/methods/losses.py`.

---

## 13. Documentation

- `README.md`: what/why, quickstart (one config + one command), the
  smoke-locally → submit-to-cluster workflow (§11.1), scenario table with
  overlap-matrix pictures, results table, citation block.
- `docs/concepts.md` (from v1 CONCEPTS.md, updated to unified spec),
  `docs/extending.md` (add a method / dataset / scenario / backbone — each a
  complete worked example), `docs/methods.md` (per method: paper citation,
  implementation notes, hyperparameter provenance, published-accuracy
  comparison table, known deviations), `docs/MIGRATION.md` (required, since
  there is no compat shim: maps every legacy `OverlapDataManager` /
  `OverlapSpec` construct to its v2 config equivalent; v1 remains on the
  `v1` branch).
- **Attribution (confirmed)**: README credits CIR (conceptual parent),
  Avalanche (stream/experience abstraction), and LAMDA-PILOT as inspiration
  for method implementations and the PTM benchmark suite — with an explicit
  statement that v2 contains no code copied from any of them. An MIT
  attribution note in the README/docs is sufficient; no NOTICE file.
  CONTRIBUTING.md: plugin checklists, the no-per-method-loss-hacks rule,
  test requirements.

---

## 14. Implementation phases (each = one or more focused sessions)

| Phase | Sections | Gate |
|---|---|---|
| P0 | `git pull` clover-cl to origin/main; branch `v2`; scaffold layout, registries, CI skeleton | CI green on empty package |
| P1 | §3 spec/planner/plan + §4 data layer refactor (legacy API removed, no shim) | golden-plan + fixture tests green |
| P2 | §5 label space + loss policies + synthetic dataset + safety gate (with a stub method) | gate demonstrably catches the v1 L2P bug |
| P3 | §6.1–6.2 CLMethod + Trainer + heads; SimpleCIL end-to-end | SimpleCIL smoke-green locally; CIFAR-100 disjoint ≈ published (cluster) |
| P4 | §9 config system + smoke profile + §11 artifacts + CLI `run`/`smoke`/`preflight` + `scripts/slurm_run.sh` | one-command run from YAML; `clover smoke` green on CPU; sbatch template submits |
| P5 | §6.4 backbone registry (config-selectable base models) + L2P/DualPrompt/CODA with core loss policy | prompt trio finite + sane on all 6 scenarios |
| P6 | remaining methods (APER-Adapter, EASE, RanPAC, MOS, TUNA) | safety gate + smoke + disjoint sanity per method |
| P7 | §10 metrics/report + matrix orchestrator port | demo report in CI |
| P8 | §7 datasets beyond CIFAR + §8 scenario extras + docs polish | tutorial-tested custom dataset |
| P9 | MIGRATION.md, README results + workflow docs, release candidate | full matrix reproduced on cluster |

Dependencies are strictly forward; any phase can be paused without leaving
the repo broken.
