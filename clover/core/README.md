# Core engine and the Stream/Experience model

## Why this module exists

The `OverlapDataManager` (v0.1 engine) is a drop-in replacement for PILOT's `DataManager`, but it exposes a raw Dataset — no metadata, no stream context, no revisit information. The classes added in v0.2 (`Experience`, `Stream`, `Benchmark`, `StreamSpec`, `stream_builder`) form a self-describing temporal abstraction layer on top of it.

## What this module provides

This directory contains CLOVER's data-flow primitives:

- `data_manager.py` — PILOT-compatible engine (legacy API, preserved
  for drop-in compatibility with PILOT/TOSCA training loops).
- `experience.py`, `stream.py`, `benchmark.py` — modern user-facing
  API. These mirror the Stream/Experience abstraction popularised by
  [Avalanche](https://github.com/ContinualAI/avalanche). They are
  not novel; they exist here to keep CLOVER self-contained without
  pulling in the full Avalanche stack, which is heavy for users who
  only need PILOT-compatible CIR extensions.
- `stream_spec.py`, `stream_builder.py` — declarative spec layer
  that compiles a high-level `StreamSpec` into the underlying
  OverlapSpec. The control-parameter style is conceptually similar
  to [CIR's Gslot/Gsamp generators](https://github.com/HamedHemati/CIR);
  CLOVER's variant adds image-level overlap modes (disjoint /
  duplicate / partial_duplicate).

| If you want to... | Use |
|---|---|
| Run pure PILOT/TOSCA experiments unchanged | `OverlapDataManager(overlap_spec=None)` |
| Add controlled class revisits to PILOT/TOSCA pipelines | `OverlapDataManager(overlap_spec=...)` or `build_benchmark(StreamSpec(...))` |
| Run CIR experiments on CIFAR-100 / Tiny-ImageNet only | Use [CIR](https://github.com/HamedHemati/CIR) directly — that's its native habitat |
| Use the full Avalanche ecosystem (loggers, strategies, metrics) | Use [Avalanche](https://github.com/ContinualAI/avalanche) directly |
| Run CIR-style experiments on CUB-200, ImageNet-R, OmniBench, VTAB | CLOVER (this is the gap CLOVER fills) |
| Distinguish same-class-same-images from same-class-different-images | CLOVER's image-level overlap modes |

## Relation to prior work

See the main [README's "Relation to prior work" section](../../README.md#-relation-to-prior-work) for citations to CIR (Hemati et al. 2023), Avalanche (Carta et al. 2023), and i-Blurry/Si-Blurry.

## The Stream/Experience model

A bare `Dataset` is just the image. An `Experience` is the image plus EXIF data — *when*, *where*, and *in what context* within the stream this snapshot was taken.

```
Benchmark
   │
   ├── train_stream  ──►  [Exp 0] → [Exp 1] → ... → [Exp T-1]
   │                         │         │                  │
   │                         ▼         ▼                  ▼
   │                      dataset   dataset            dataset
   │                       + meta    + meta             + meta
   │
   ├── test_stream   ──►  [Exp 0] → [Exp 1] → ... → [Exp T-1]
   │
   └── stream_spec (declarative — describes structure, not exact placements)
```

A `Stream` is iterable; indexing `stream[k]` returns the k-th Experience. The stream also exposes aggregated metrics: `revisit_density()`, `class_appearance_count()`, `average_overlap()`.

## What an Experience carries

```python
@dataclass(frozen=True)
class Experience:
    task_label: int          # zero-based step index in the stream
    benchmark_name: str      # identifies the parent benchmark run

    dataset: Dataset         # PyTorch Dataset — yields (idx, image, remapped_label)

    # Class membership
    classes_in_this_experience: list[int]   # remapped class IDs present here
    classes_seen_so_far: list[int]          # union over experiences 0..task_label
    classes_in_future: list[int]            # for evaluation diagnostics only
    total_classes_in_stream: int

    # Revisit metadata — the core CLOVER contribution
    revisiting_classes: list[int]           # appeared in some EARLIER experience
    first_appearance_of: list[int]          # never seen before this experience
    overlap_with_previous: dict[int, int]   # {prev_task_label: n_shared_classes}

    # Reproducibility
    image_indices: dict[int, list[int]]     # {class_id: [indices]} for manifests
    n_samples: int
```

Invariant enforced at construction: `revisiting_classes ∪ first_appearance_of == classes_in_this_experience` and the two sets are disjoint.

## Why streams enable better metrics

**1. Forgetting on revisited classes vs. fresh classes** (one-liner):
```python
revisit_exps = [e for e in bench.train_stream if e.is_revisit_experience()]
# e.revisiting_classes tells you exactly which classes to measure forgetting on
```

**2. No accidental future-class contamination**:
```python
# Safety check during evaluation — never evaluate on future classes
assert set(eval_classes).issubset(set(exp.classes_seen_so_far))
```

**3. Per-experience overlap diagnostics**:
```python
for exp in bench.train_stream:
    if exp.overlap_with_previous:
        print(f"Exp {exp.task_label} shares with: {exp.overlap_with_previous}")
```

## Pointer to the deep concept doc

For the full motivation and contrast with PILOT/Avalanche, see [../docs/CONCEPTS.md](../docs/CONCEPTS.md).
