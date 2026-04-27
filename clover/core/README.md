# Core engine and the Stream/Experience model

## Why this module exists

This directory holds the data structures that make CLOVER novel. The `OverlapDataManager` (v0.1 engine) is a solid drop-in for PILOT, but it exposes a raw Dataset — no metadata, no stream context, no revisit information. The classes added in v0.2 (`Experience`, `Stream`, `Benchmark`, `StreamSpec`, `stream_builder`) form a self-describing temporal abstraction layer on top of it. Anyone reviewing the design should start here.

## The novelty in one paragraph

CLOVER v0.2 is the first CL benchmark library combining all three of: **(a)** configurable class revisits with controlled image-overlap strategies (disjoint / duplicate / partial-duplicate), **(b)** declarative stream specifications that describe statistical structure rather than exact task-pair assignments — enabling proper multi-seed error bars, and **(c)** self-describing Experience objects that carry revisit metadata so that overlap-aware metrics are one-liners. This combination does not exist in PILOT, Avalanche, PyCIL, or TOSCA.

| Feature | PILOT | PyCIL | Avalanche | CLOVER v0.2 |
|---|:---:|:---:|:---:|:---:|
| Class revisits | — | — | Partial | ✓ |
| Image-overlap strategies | — | — | — | ✓ |
| Declarative stream spec | — | — | — | ✓ |
| Multi-seed statistical structure | — | — | — | ✓ |
| Revisit metadata on experience | — | — | — | ✓ |

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
