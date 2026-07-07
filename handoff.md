# Handoff — clover-cl v2 (2026-07-07: P10 done; P9 blocked only on user actions)

Session notes for picking this back up cold in a fresh Claude Code session.
Read this first, then `CLAUDE.md` → `PLAN.md` → the relevant `SPEC.md`
section for the next phase.

## State right now

- Repo: `D:\Dev\Research\clover-cl`. Work branch: `v2`.
- `v2` was pushed to `origin/v2` (`b444ec2`) through P5. **P6, P7, P8, P9,
  and P10 are committed locally but not yet pushed** — push only if the
  user asks.
- `main` is untouched (v1, preserved as-is until release per SPEC.md — do
  not merge or push to `main` before then, and don't create a `v1` branch
  either without asking; the user explicitly deferred all branch/tag
  mechanics — see below). **v1 is a legitimate, sanctioned read-only
  reference from P8 onward** — read via `git show main:<path>`, never
  checked out.
- `PLAN.md`: **P0-P8 and P10 all ticked. P9 (Release candidate) is the
  only unticked phase** — and there is genuinely **no more local
  implementation work left in it**. What remains is two things only the
  user can do: (1) run `configs/matrix_full.yaml` on their own Snellius
  access, (2) confirm/perform the `v1` branch + RC tag mechanics (this
  session was explicitly told "no git surgery"). If you start a session
  and PLAN.md still shows P9 unticked with nothing else queued, **don't
  invent new work for it** — ask the user whether they've run the cluster
  matrix yet, or whether there's a different next priority.
- Local `.venv` (Python 3.11) has everything installed (`pip install -e
  ".[dev]"` already run) — torch/torchvision/timm/pyyaml/pillow +
  pytest/ruff/mypy/types-PyYAML.
- Current test suite: **454 tests**. Full `pytest tests/ -v` run takes
  **~9 minutes** — the registry-driven safety gate
  (`tests/test_method_registry_safety_gate.py`, 54 cases: 9 methods × 6
  scenarios) and `tests/test_cli_run_matrix.py` (real subprocess spawns)
  are the expensive parts. Budget for this when working — don't assume
  the suite is fast.

## Session protocol (from PLAN.md, already established — just follow it)

1. Read `PLAN.md`. If P9 is still the only unticked phase and nothing new
   has been asked, this is a "wait for the user" state, not a "find work"
   state — see above.
2. For any *new* phase/task: read its `SPEC.md` section(s) first (P10
   wasn't in SPEC.md — it was added mid-session; a similar ad-hoc phase
   might happen again if scoping a phase reveals a repo-wide blocker).
3. **Work in plan mode first** — research, then present a plan via
   `ExitPlanMode` before writing code. If scoping reveals something that
   blocks more than the current phase, say so explicitly and let the user
   decide how to scope the fix (this is exactly how P10 came to exist) —
   don't silently absorb it or silently route around it.
4. Implement, then verify DONE criteria actually pass locally
   (`ruff check clover tests`, `mypy clover/core clover/config`, `pytest
   tests/ -v`, `clover smoke`) — and **actually run the real end-to-end
   path once**, not just unit tests in isolation, if the phase's claim is
   "X now works." Three separate real bugs this project has now found
   (P8's transform-composition gap, P9's device-hardcoding gap, P10's
   `Trainer.run()` hardcoded `{}` for method_cfg) were only caught this way.
5. Tick the box in `PLAN.md`, log deviations under `PLAN.md`'s `## Log`
   section (dated), commit on `v2`. **Never** add an AI-attribution
   watermark to commit messages. Don't push, don't touch `main`/branches/
   tags, unless the user explicitly asks *in that session* — a past
   approval doesn't carry forward.
6. Large phases span multiple sessions — deliver in sub-passes, logging
   progress each time, ticking the phase's box only once *all* of its
   DONE criteria are met.

## What's done (P0-P8, P10; P9's non-cluster work) — one line each

- **P0-P8**: scaffold → stream model → label space/losses → CLMethod+
  Trainer+SimpleCIL → config/CLI → backbones+prompt trio → adapter family
  (all 9 methods) → metrics/reporting/matrix → datasets+scenario extras+docs.
  See prior handoff versions (git history) for per-phase detail if needed;
  the summary below focuses on P9/P10, this session's actual work.
- **P9 (docs/configs/small-fixes; checkbox NOT ticked — cluster run +
  branch mechanics remain)**: fixed `device="cpu"` hardcoding (no config
  could ever select a GPU) and missing `manifest.json`/`log.txt`/
  `cudnn.deterministic`/`cudnn_benchmark` (SPEC §11's run-artifact layout
  was only half-implemented). Created `configs/` (didn't exist): 7
  single-run examples + `matrix_full.yaml`, verified offline. Rewrote
  `README.md`/`docs/MIGRATION.md` (both were still v1 content); added
  `CONTRIBUTING.md`.
- **P10 — Backbone real-image support (new phase, done, checkbox
  ticked)**: see below — the big piece of this session.

## P10 — what actually got fixed, and what's still a real limitation

Found while scoping P9: no method/backbone threaded a channel count
through at all (`TinyViT`/`TinyMLP` hard-defaulted to 1 channel). The user
said: document precisely, spin off a new phase, then **complete it in the
same session**. Three real, previously-invisible bugs surfaced by actually
writing an end-to-end verification test (not just unit-testing pieces):

1. **Channel-count plumbing** (the originally-scoped fix): `CLDataset
   .channels: int = 3` (new, `synthetic` overrides to 1), `StreamInfo
   .channels`, threaded through `Trainer.run()`, `TinyMLP`'s `Linear` size,
   and every method's `build()` (`backbone_kwargs.setdefault("in_chans",
   stream_info.channels)`) — a small, uniform edit at 6 call sites since 6
   of 9 methods already had a `backbone_kwargs` pass-through pattern, and
   every wrapper-mechanism backbone already forwarded `**base_kwargs` to
   `resolve_base_model` with zero changes needed.
2. **`get_backbone` → `resolve_base_model` swap at all 6 call sites** (not
   just "at least one," a strictly better outcome for the same work) — but
   this immediately surfaced two more bugs once actually exercised:
   - **timm models expose `.num_features`, not `.feature_dim`** (every
     backbone/method in this project reads `.feature_dim` — a pervasive
     convention). Fixed in `resolve_base_model` itself: `model
     .feature_dim = model.num_features` after timm construction.
   - **`timm.create_model` doesn't accept an `input_size` kwarg** (every
     method unconditionally sets one). Fixed by popping `input_size` out
     of kwargs specifically in the timm-fallback branch (the registry
     branch, for `TinyViT`/`TinyMLP`, still needs and gets it).
3. **The most serious find, and arguably this session's biggest**:
   `Trainer.run()` hardcoded `self.method.build(stream_info, {})` — an
   empty dict, always. **Every method-level config override (backbone
   selection included) has been silently discarded by every real `clover
   run` since P4/P5** — invisible because every prior test either called
   `method.build(info, cfg)` directly (bypassing `Trainer`) or never
   overrode a method-level key. `resolve_base_model`'s entire point —
   "backbone selectable from config" (SPEC §6.4) — was functionally dead
   code from `clover run`'s perspective until this session. Fixed:
   `Trainer.__init__` gains `method_cfg: Optional[Dict[str, Any]] = None`
   (defaults `{}`, so all ~15 existing call sites stay valid unchanged);
   `cli.py:cmd_run` passes `method_cfg=resolved.method_cfg`.

**Real, precise, still-standing limitation (documented, not silently
expanded into or papered over)**: the prompt/adapter wrapper mechanisms
(`vit_prompt_pool`, `vit_dual_prompt`, `vit_coda_prompt`, `vit_adapter*`)
need their `base_model` to implement CLOVER-specific hooks
(`query_features`, `forward_tokens`/`forward(x, adapter=...)`) that a raw
timm ViT doesn't have. `resolve_base_model` happily *constructs* a real
timm model as a wrapper's base (confirmed by hand), but the wrapper's
forward pass then raises `AttributeError`. **Real timm usage is only
fully wired end-to-end for SimpleCIL** (the one "complete backbone"
method with a plain `forward(x) -> features`, no extra hooks). The other
8 methods' shipped configs still use their tiny, untrained, TinyViT-family
stand-in backbone — fine for shape-correctness, not for real accuracy.
Splicing prompts/prefixes/adapters into real timm ViT internals is future
work — not proposed as a new phase yet; raise it with the user if a real
accuracy comparison for the other 8 methods becomes a priority.

Verification: `tests/test_backbones_real_image.py` (new) — a real `clover
run` (through `main(["run", ...])`, not a hand-wired `Trainer` call)
against a real, tiny, hand-built `image_folder` fixture (P8's zero
-network-access pattern): SimpleCIL + a real timm backbone selected purely
via config (exercises the `method_cfg` fix + channel threading + timm
fallback together), and L2P with its own default backbone (proves channel
threading works for the wrapper family too, without claiming real-timm
compatibility for it).

## Deferred items (logged in `PLAN.md`'s `## Log`, not forgotten)

- **CIFAR-100-vs-published-accuracy verification** (every phase from P3
  onward) — now genuinely just needs the user's own cluster access; P10
  removed the code-side blocker.
- **Per-method typed config schemas** — still deferred; no method's
  `build()` reads any `cfg.get(...)` key except `"backbone"`.
- **`training.amp` is validated but still inert** — no autocast wrapping
  exists anywhere. Shipped configs use `amp: none`.
- **Real timm ViT internals hooking for the 8 prompt/adapter-family
  methods** (P10, see above) — a real, precise, currently-unaddressed
  limitation, not a bug to silently work around.
- **CODA-Prompt's gradient masking**, **L2P's auxiliary key-pulling loss**
  (P5); **no dropout in `Adapter`**, **MOS's entropy self-refinement →
  plain logit average**, **TUNA's per-sample routing → EMR-merged adapter
  directly**, **EASE's `beta`/`use_init_ptm`/`use_diagonal`, MOS's orth
  regularizer + two-param-group optimizer, TUNA's `use_orth`/`decay`/dead
  `r`** (all P6) — none reimplemented, see `docs/methods.md`.
- **`clover report`'s scenario-name grouping relies on the run directory
  naming convention**, not `config_resolved.yaml` (P7).
- **`run-matrix` GPU-pool parallelism** (P7) — sequential dispatch only.

## Judgment calls worth knowing about

- **A hardcoded value that's never wrong *yet* (because nothing ever
  exercised the alternative) can silently invalidate an entire promised
  mechanism.** This is now a 3-for-3 pattern in this project: P8's
  transform-composition gap, P9's `device="cpu"` hardcoding, P10's
  `Trainer.run()` hardcoded `{}` for method_cfg. All three were only
  caught by actually writing a real end-to-end test/doc/workflow that
  exercised the path for the first time — never by unit-testing pieces in
  isolation. **When a phase's DONE claim is "X now works end-to-end,"
  write a test that goes through the real entry point (`clover run`/
  `main([...])`), not just a hand-constructed call to the function you
  changed** — the hand-constructed call is exactly what was already
  passing before the bug was fixed.
- **When scoping reveals a blocker big enough to invalidate the next
  phase's entire premise, ask the user how to scope the fix rather than
  silently absorbing it or silently routing around it.** P8 found the
  channel/backbone gap for `image_folder` specifically and correctly
  scoped around it (verified the dataset layer, stopped at the boundary).
  P9 re-found the *same* gap as a repo-wide blocker — different enough in
  severity to ask rather than repeat P8's call. The user's answer became
  P10, and later explicitly asked to complete it same-session.
- **A research finding can legitimately shrink an approved plan's scope
  significantly — say so plainly rather than silently doing "the plan"
  regardless.** P10's plan assumed 9 separate method edits; research
  before writing code found 6 of 9 already shared a `backbone_kwargs`
  pattern and every wrapper mechanism already forwarded kwargs correctly,
  shrinking the actual diff to 6 call sites with 2-3 lines each. Worth
  surfacing in the plan itself (this session did), not just discovering
  it silently mid-implementation.
- **Swapping one function for a safe superset (same behavior for existing
  inputs, more capability for new ones) is worth doing broadly rather than
  narrowly, when the phase's minimum bar only asked for "at least one."**
  `get_backbone` → `resolve_base_model` at all 6 call sites (not 1) cost
  the same amount of work and benefited all 9 methods instead of 1.
- **Don't assume a git branch mentioned in `CLAUDE.md`/`SPEC.md` actually
  exists — check `git branch -a` before writing docs that reference it.**
  Both `README.md` and `docs/MIGRATION.md` initially said "on the `v1`
  branch" (which doesn't exist yet, only `main` does) and were caught and
  fixed before commit.
- **A pipeline stage that's never been exercised for real can hide a
  totally silent gap indefinitely** — the umbrella lesson under the
  "hardcoded value" bullet above. The only reliable check is actually
  running the full path for real, once.
- **A dataclass's `to_dict()` output must stay a valid `from_dict()` input
  for every field, or `config_resolved.yaml` round-tripping breaks** (P8) —
  any new config field needs both directions checked together from the start.
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
  those modules import** — third-party stub gaps and `ClassVar` subtleties
  both surfaced this way (P8).
- **`config_resolved.yaml` doesn't carry the scenario *name*** (P7) —
  `clover report` parses it from the run directory name instead.
- **The registry-driven safety gate's `optimizer_lr=3e-2`, `epochs=40`**
  (tuned for CODA-Prompt, P5) needed zero retuning for any of the 9 methods.
- **Never let per-experience randomness draw from the global torch RNG in
  `Trainer`** — `_build_loader` seeds each experience's `DataLoader` from
  `(run.seed, task_label)`.
- **Global torch backend flags (`cudnn.deterministic`/`cudnn.benchmark`)
  mutated in a test need explicit cleanup in a `finally` block** — they're
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
