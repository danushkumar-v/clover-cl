# Migration Guide

> If you are coming from CIR or Avalanche directly, CLOVER's API is
> intentionally smaller and PILOT-shaped. For full-featured CL
> research tooling, those libraries remain the canonical choice.

For users coming from CLOVER v0.1 or from LAMDA-PILOT.

---

## Side-by-side: v0.1 `OverlapDataManager` → v0.2 `build_benchmark`

### Baseline (no overlap)

**v0.1 / PILOT-compatible:**
```python
from clover import OverlapDataManager

dm = OverlapDataManager("cifar100", init_cls=10, increment=10)
for t in range(dm.nb_tasks):
    ds = dm.get_dataset(t, source="train", mode="train")
    train(ds)
```

**v0.2 stream API:**
```python
from clover import build_benchmark, StreamSpec

spec = StreamSpec(dataset="cifar100", init_cls=10, increment=10)
bench = build_benchmark(spec)
for exp in bench.train_stream:
    train(exp.dataset)
```

### With overlap

**v0.1 explicit pair declaration:**
```python
from clover import OverlapDataManager
from clover.core.overlap_spec import OverlapSpec, OverlapPair, ImageSplit

spec = OverlapSpec(
    mode="partial",
    pairs=[OverlapPair(tasks=(0, 5), shared_classes=[3, 7, 22])],
    image_split=ImageSplit(strategy="disjoint"),
    seed=42,
)
dm = OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)
```

**v0.2 declarative:**
```python
from clover import build_benchmark, StreamSpec, RevisitSpec

spec = StreamSpec(
    dataset="cifar100",
    init_cls=10,
    increment=10,
    revisits=[RevisitSpec(classes=[3, 7, 22], times=1,
                          placement="random", min_gap=3)],
    stream_seed=42,
)
bench = build_benchmark(spec)
```

---

## When to keep using `OverlapDataManager`

- You need exact control over which task-pair shares which class IDs.
- You are integrating with a PILOT or TOSCA training loop that calls
  `dm.get_dataset(list_of_class_ids, source, mode)` directly.
- You are working with a custom dataset not in CLOVER's registry.
- You need `get_dataset_with_split` for validation splitting.

For these cases, `OverlapDataManager` is the right tool. The v0.2 stream API
is always available via `bench.engine` for power users who need both.

---

## Mapping the 8 v0.1 scenarios to their stream-spec helpers

| v0.1 (low-level) | v0.2 (stream) |
|---|---|
| `exact_replay.build_spec(...)` | `exact_replay.build_stream_spec(...)` |
| `partial_overlap.build_spec(...)` | `partial_overlap.build_stream_spec(...)` |
| `hierarchical.build_spec(...)` | `hierarchical.build_stream_spec(...)` |
| `distribution_shift.build_spec(...)` | `distribution_shift.build_stream_spec(...)` |
| `near_miss.build_spec(...)` | `near_miss.build_stream_spec(...)` |
| `long_range_revisit.build_spec(...)` | `long_range_revisit.build_stream_spec(...)` |
| `cumulative_drift.build_spec(...)` | `cumulative_drift.build_stream_spec(...)` |
| `symmetric_pair.build_spec(...)` | `symmetric_pair.build_stream_spec(...)` |

All `build_spec` functions remain unchanged. No v0.1 code needs to be modified.

---

## Backwards compatibility guarantees

- All `OverlapDataManager` constructor arguments from v0.1 are unchanged.
  The only addition is `preserve_task_size=True` (default), which fixes the
  task-size contract bug. Set `preserve_task_size=False` to restore v0.1
  behaviour.
- `dm.get_dataset(task_id, ...)` and `dm.get_dataset(list_of_class_ids, ...)`
  both still work.
- All 8 scenario `build_spec()` functions still work and return `OverlapSpec`.
- `OverlapSpec`, `OverlapPair`, `ImageSplit` are unchanged.
- The class-order produced by `shuffle_seed=1993` is identical to v0.1 (and
  to PILOT). No fixture re-capture is needed.

---

## Coming from LAMDA-PILOT

Replace:
```python
from utils.data_manager import DataManager
dm = DataManager("cifar100", shuffle=True, seed=1993, init_cls=10, increment=10, args=args)
```

With:
```python
from clover import OverlapDataManager
dm = OverlapDataManager("cifar100", init_cls=10, increment=10, shuffle_seed=1993)
```

The class order, task splits, and `get_dataset` calling convention are
byte-identical. The only difference is that CLOVER does not require an `args`
object; configuration is passed directly to the constructor.
