# Contributing to CLOVER

CLOVER is a plugin architecture (SPEC R3): core never imports a method,
dataset, scenario, or backbone by name. Everything is added by implementing
a small interface and registering it with a decorator. See
[`docs/extending.md`](docs/extending.md) for one complete, runnable worked
example of each kind of plugin before you start.

## Plugin checklists

**New method** (`@register_method`, `clover/methods/base.py:CLMethod`):
- [ ] Implements `build`/`train_experience`/`classifier`/`state_dict`/
      `load_state_dict`; only grow head/prompt/adapter structure in
      `before_experience` (never `train_experience`/`after_experience` —
      the Trainer's resume-replay only re-invokes `before_experience` for
      already-completed experiences).
- [ ] Uses `exp.label_space` (`LabelSpaceView`) for any revisit-aware
      masking — see the rule below.
- [ ] Declares `cacheable_features` truthfully (only `True` if the backbone
      never changes after `build()`).
- [ ] Registered in the revisit-safety gate
      (`tests/test_method_registry_safety_gate.py`) automatically, since it
      parametrizes over `list_methods()` — no manual registration step, but
      run the full suite once to confirm it passes at the default
      hyperparameters before assuming they need retuning.
- [ ] Passes `clover smoke`.

**New dataset** (`@register_dataset`, `clover/datasets/base.py:CLDataset`):
- [ ] Implements `num_classes`/`get_class_to_indices`/`__getitem__`/
      `__len__`; declares `input_size` and `train_trsf`/`test_trsf`/
      `common_trsf` (SPEC §7 — datasets own their input pipeline).
- [ ] If it's an `ImageFolder`-shaped `<root>/<dir>/{train,test}/<class>/`
      layout, subclass `clover.datasets.image_folder_base
      .ImageFolderCLDataset` instead of implementing `CLDataset` from
      scratch (see `clover/datasets/cub200.py` for the ~5-line pattern).
- [ ] A missing/misconfigured data directory raises an actionable
      `FileNotFoundError` (staging path + where to get the data), not a
      silent fallback.
- [ ] No real download or network access in its tests — monkeypatch the
      underlying loader (see `tests/conftest.py:patched_cifar100`) or use a
      real tiny hand-built fixture directory (see
      `tests/test_datasets_image_folder.py`).

**New scenario** (`@register_scenario`,
`(dataset_info, init_cls, increment, seed, **params) -> StreamSpec`):
- [ ] Built from `RevisitSpec`'s axes (`classes`/`placement`/`label`/
      `images`/`min_gap`/`times` — see `docs/CONCEPTS.md` §3), not a new
      core mechanism — if your scenario needs something `RevisitSpec` can't
      express (e.g. external taxonomy metadata, backward-in-time class
      injection), it's likely out of scope for a scenario alone; see
      `docs/CONCEPTS.md` §8 for three real examples of scenarios that don't
      fit this model and precisely why.
- [ ] A golden-plan test (`tests/test_scenarios.py`'s style): assert the
      resolved plan's `echo_table`/`revisit_ids`/`task_class_lists` match
      what the scenario claims to do, and that it's distinguishable from
      the closest existing scenario.

**New backbone** (`@register_backbone`, a complete `nn.Module`, or a
mechanism wrapper over `clover/backbones/loader.py:resolve_base_model`):
- [ ] Never hard-codes a checkpoint/model name inside a method — base
      models are config-selectable (`backbone:` key, resolved via
      `resolve_base_model`, registry-first then timm fallback).
- [ ] A mechanism wrapper (prompt/prefix/adapter-style) takes a `base_model:
      str` and resolves it via `resolve_base_model`, never importing timm
      directly itself.

## The no-per-method-loss-hacks rule

Revisit-safe label-space handling lives in **exactly one place**:
`clover/methods/losses.py` + `LabelSpaceView` (`clover/core/experience.py`).
A method that writes its own `-inf` logit masking, its own
`targets >= known_classes` range check, or its own `_known_classes`
bookkeeping is a bug, even if its tests pass — the stream/experience is the
single source of truth for label-space state and head sizes, and range
arithmetic silently breaks the moment a same-id revisit re-presents an
"old" (numerically small) id in a later batch. Always build masks from
`exp.label_space`'s materialized id sets (`new_mask`/`old_mask`/
`logit_mask`), never from arithmetic on raw target values.

## Test requirements

- CPU-only, no GPU or dataset downloads in tests (`tests/conftest.py` has
  the monkeypatch fixtures already in use — reuse or extend them, don't
  add a new real download).
- A new method/dataset/scenario must be picked up by the existing
  parametrized suites automatically (they iterate `list_methods()`/
  `list_datasets()`/`list_scenarios()`) — if you find yourself hand-adding
  a new plugin's name to a test's hardcoded list, check whether the test
  should instead parametrize over the registry.
- `ruff check clover tests` and `mypy clover/core clover/config` clean
  before any PR.
- Golden fixtures (seed-1993 class order, per-scenario plans) are
  contracts: if a change breaks one, that's a stop-and-justify moment, not
  a "regenerate the fixture" moment.
- Coverage ≥ 85% on `clover/core`, `clover/config`, `clover/methods/losses.py`.

## Everything else

`CLAUDE.md` has the full coding standards (dataclasses + explicit
`validate()`, typed unknown-key rejection with "did you mean" suggestions,
seeded-generator discipline — no bare `np.random`/`random` in library
code) and the complete list of removed v1 patterns that must not be
reintroduced. Read it before your first PR.
