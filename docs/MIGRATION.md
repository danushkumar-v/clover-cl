# Migration guide: v1 → v2

v2 is a clean break — **there is no compatibility shim** (`CLAUDE.md`: no
`clover/compat`, no legacy `OverlapDataManager`, no `int|list` unions, no
PILOT calling conventions anywhere in v2). `OverlapDataManager`/
`OverlapSpec` and the 8 scenario `build_spec()` functions still exist,
unchanged, on the **`main` branch** (to become the **`v1` branch** at
release, per `SPEC.md`) — check it out if you need to keep using them.
This guide maps each real v1 construct (read directly from `clover/core/
data_manager.py`/`clover/core/overlap_spec.py` on `main`, the same "read
the actual code, don't guess" approach used throughout this project) to
its v2 equivalent.

> v1 also had its own in-progress internal experiment at a `StreamSpec`/
> `RevisitSpec`-shaped API (`build_stream_spec()` functions alongside each
> scenario's `build_spec()`). That API is **not** what v2 continues — v2's
> `StreamSpec`/`RevisitSpec` (`clover/core/spec.py`) share the same names
> but a genuinely different, more general field syntax (see below). If you
> were using v1's `build_stream_spec()`, treat it as superseded by v2's
> config system, not as something with a 1:1 field mapping.

---

## Baseline (no overlap)

**v1** (`OverlapDataManager`, PILOT-compatible):
```python
from clover import OverlapDataManager

dm = OverlapDataManager("cifar100", init_cls=10, increment=10)
for t in range(dm.nb_tasks):
    ds = dm.get_dataset(t, source="train", mode="train")
    train(ds)
```

**v2** (one config, one command):
```yaml
# configs/my_run.yaml
stream: {dataset: cifar100, init_cls: 10, increment: 10}
method: {name: simplecil}
```
```
clover run configs/my_run.yaml
```

If you want to keep writing your own training loop instead of using
`clover run`, the closest v2 equivalent to iterating `dm.get_dataset(t,
...)` per task is iterating `Experience` objects from a `Benchmark`:
```python
from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets import get_dataset

dataset_cls = get_dataset("cifar100")
train_ds, test_ds = dataset_cls(train=True), dataset_cls(train=False)
info = DatasetInfo("cifar100", train_ds.num_classes)
spec = StreamSpec(dataset="cifar100", init_cls=10, increment=10)
benchmark = build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())

for exp in benchmark.train_stream:
    train(exp)  # exp.image_indices -> your own Dataset/DataLoader wrapping
```
(`clover/training/trainer.py:ExperienceDataset` is the wrapper `clover run`
itself uses, if you want to reuse it rather than writing your own.)

---

## With overlap: `OverlapSpec` → `scenario` or inline `revisits`

**v1** (explicit `OverlapPair`/`EchoSpec` declarations):
```python
from clover import OverlapDataManager
from clover.core.overlap_spec import EchoSpec, ImageSplit, OverlapSpec

spec = OverlapSpec(
    mode="exact_replay",
    task_class_lists=[[0,1,2,3,4], [5,6,7,8,9], [0,1,2,3,4]],  # echo ids picked by you
    echoes=[EchoSpec(new_id=10, source_id=0, image_relation="same"), ...],
    image_split=ImageSplit(strategy="duplicate"),
    seed=42,
)
dm = OverlapDataManager("cifar100", init_cls=5, increment=5, overlap_spec=spec)
```

**v2** (named scenario — the framework computes the echo ids and class
lists for you):
```yaml
stream:
  dataset: cifar100
  scenario: exact_replay
  init_cls: 5
  increment: 5
  stream_seed: 42
```

Or, for a bespoke revisit pattern with no named scenario, an inline
`revisits:` block (v2's `RevisitSpec`, see `docs/CONCEPTS.md` §3 for the
full axes):
```yaml
stream:
  dataset: cifar100
  init_cls: 5
  increment: 5
  revisits:
    - classes: [0, 1, 2, 3, 4]
      placement: end_of_stream
      label: new       # v1's EchoSpec.new_id path
      images: same     # v1's ImageSplit(strategy="duplicate")
```

Key differences, not just renames:

- **v1 required you to compute and hand over exact task-pair/echo-id
  assignments** (`OverlapPair(tasks=(0, 5), shared_classes=[...])`,
  `EchoSpec(new_id=10, source_id=0, ...)`). **v2 describes structure, not
  exact assignments** — `RevisitSpec` says *which classes, how placed, how
  many times*; the framework resolves concrete ids/placement from
  `(shuffle_seed, stream_seed)`. This is also what makes multi-seed error
  bars meaningful in v2 (a different `stream_seed` gives a different
  concrete realization of the same statistical structure) — v1's exact
  pair declarations were fixed regardless of seed.
- **v1's `image_split`/`ImageSplit` was one setting for the whole spec**
  (every shared class in the stream used the same strategy). **v2's
  `images` is per-`RevisitSpec`** — different revisit patterns in the same
  stream can use different image relations.
- **v1's `EchoSpec.image_relation: "same"|"new"`** maps to v2's
  `images: "same"|"new"`, plus v2 adds a third option, `"partial:<pct>"`,
  with no v1 equivalent.

## Mapping the 8 v1 scenario names

| v1 (`build_spec()` → `OverlapSpec`) | v2 (`scenario:`) |
|---|---|
| `exact_replay` | `exact_replay` — same name, same shape (fresh echo id, identical images) |
| `partial_overlap` | `partial_overlap` — same name; v1's `n_revisit_classes` int is now `overlap_fraction` (a ratio, not a count) |
| `long_range_revisit` | `long_range_revisit` — same name, same shape (fresh echo id, disjoint images) |
| `mid_range_revisit` | `mid_range_revisit` — same name, same shape (echoes an interior task, not task 0) |
| `cumulative_drift` | `cumulative_drift` — same name, same shape (anchor classes, same-id, every task) |
| `distribution_shift` | `distribution_shift` — same *name*, but v2's version (added in P8) was re-derived from v2's `RevisitSpec` model from scratch, not ported line-for-line: same classes, same label id, once, at the end of the stream, on a different image subset (`split_ratio`) |
| `hierarchical` | **Not implemented in v2.** Needs classes assigned to specific tasks by external taxonomy metadata; v2's class-to-task assignment is purely sequential/budget-driven (`init_cls`/`increment`), with no hook for this. See `docs/CONCEPTS.md` §8. |
| `near_miss` | **Not implemented in v2.** v1's version shares *no* classes between tasks — its only distinguishing feature was an `adjacency_map` recorded for downstream analysis only, which v2's `StreamSpec` has no field for. Indistinguishable from `disjoint_baseline` under v2's model without it. See `docs/CONCEPTS.md` §8. |
| `symmetric_pair` | **Not implemented in v2.** v1's real intent (two tasks *mutually* injecting 50% of their classes into each other) has no v2 equivalent — v2's model only expresses "a class revisits later in the stream," never "an earlier task retroactively gains classes from a later one" (that would leak future information backward). The only expressible direction is already exactly `partial_overlap(overlap_fraction=0.5)`. See `docs/CONCEPTS.md` §8. |

---

## Removed, not renamed

- **`preserve_task_size` (v1's eviction-on-overflow behavior) is gone.**
  v2's `task_size: "fixed"` raises a clear `ValueError` naming the
  offending experience instead of silently evicting a class to make room —
  a v1 trap CLAUDE.md explicitly calls out as intentionally not
  reintroduced. Use `task_size: "grow"` if you want experiences to grow
  past their nominal budget instead of erroring.
- **PILOT-compatible calling conventions are gone**: `get_dataset`'s
  `int|list` union (task id *or* class-id list), `ret_data`, `m_rate`,
  `appendent` — none exist in v2. `clover run` owns the whole loop; if you
  need direct dataset access, use the `Experience`/`ExperienceDataset`
  path shown above.
- **`OverlapDataManager.save_manifest(path)`** → automatic: every `clover
  run` writes `<run_dir>/manifest.json` itself (no manual call needed).

## Config keys carried over directly

| v1 constructor arg | v2 config key |
|---|---|
| `dataset_name` | `stream.dataset` |
| `init_cls` | `stream.init_cls` |
| `increment` | `stream.increment` |
| `shuffle_seed` (default 1993) | `stream.shuffle_seed` (default 1993 — same PILOT-compatible base class-order permutation) |
| `data_root` | `stream.data_root` |
| `overlap_spec.seed` | `stream.stream_seed` |

## Coming from LAMDA-PILOT directly (never used CLOVER v1)

Replace:
```python
from utils.data_manager import DataManager
dm = DataManager("cifar100", shuffle=True, seed=1993, init_cls=10, increment=10, args=args)
```

With a v2 config:
```yaml
stream: {dataset: cifar100, init_cls: 10, increment: 10, shuffle_seed: 1993}
method: {name: simplecil}
```
```
clover run configs/my_run.yaml
```

`shuffle_seed=1993` reproduces PILOT's class-order permutation exactly
(same seed-1993 fixture this project's own tests guard against as a
regression fixture — not a compatibility promise for other seeds).
