# CLOVER Concepts: Streams, Experiences, and Declarative Overlap

This document explains the design of CLOVER v0.2's Stream/Experience model,
why it was built, and how it differs from existing CL benchmark libraries.

---

## 1. The problem with disjoint-task benchmarks

Standard class-incremental learning benchmarks divide a dataset into disjoint
task partitions. Once "cat" appears in Task 0, it never appears again:

```
Task 0: {cat, dog, ship, plane, frog}
Task 1: {car, horse, deer, bird, truck}
Task 2: {apple, mushroom, ...}
```

This is a reasonable simplification, but real-world streaming data does not
work that way. A content feed, a medical imaging pipeline, or a surveillance
system will encounter the same concept repeatedly across time — sometimes
with different images, sometimes with different co-occurring concepts.

Methods like GRAFT that explicitly exploit class revisits have no existing
benchmark infrastructure to evaluate against. CLOVER fills that gap.

---

## 2. Random-access task lookup vs. temporal stream

PILOT (and most CL libraries) expose tasks as a random-access lookup:

```python
dataset = dm.get_dataset([0, 1, 2, 3, 4], source="train", mode="train")
```

This is fine for the training loop, but it leaves all bookkeeping to the
researcher: which classes has the model seen so far? Which classes in this
task are revisiting from earlier? Is it safe to evaluate on these samples?

CLOVER v0.2 wraps the same engine with a temporal stream:

```python
for exp in bench.train_stream:
    # Everything is on the object:
    train_on(exp.dataset)
    track_revisiting_classes(exp.revisiting_classes)
    evaluate_only_on(exp.classes_seen_so_far)
```

The stream is iterable, indexable, and carries stream-level statistics
(`revisit_density`, `average_overlap`, `class_appearance_count`).

---

## 3. Task-pair declaration vs. declarative stream spec

CLOVER v0.1 required explicit task-pair declarations:

```yaml
# v0.1 OverlapSpec — exact and rigid
pairs:
  - tasks: [0, 7]
    shared_classes: [0, 1, 2, 3, 4]
```

This is powerful but inflexible for experimentation: every multi-seed run
produces the *exact same* overlap structure because the class IDs are
hardcoded.

CLOVER v0.2 introduces `StreamSpec` — a declarative description of
*statistical structure*:

```yaml
# v0.2 StreamSpec — structural, multi-seed-ready
revisits:
  - classes: 5         # pick 5 classes at random
    times: 1
    placement: random
    min_gap: 3
stream_seed: 42        # change this for a different concrete realisation
```

With `stream_seed=42` you get one concrete stream; with `stream_seed=99`
you get a different stream with the same statistical structure (5 classes
revisit once, randomly placed with gap ≥ 3). This is the essential
requirement for computing meaningful error bars.

---

## 4. The Experience object

An `Experience` is a frozen dataclass that bundles a Dataset with
self-describing metadata. The key fields and their use cases:

| Field | Use case |
|---|---|
| `task_label` | Loop index; identifies position in stream |
| `dataset` | Passed to DataLoader as normal |
| `classes_in_this_experience` | Classifier head sizing, logging |
| `classes_seen_so_far` | Safe evaluation boundary — never evaluate on future classes |
| `classes_in_future` | Diagnostic only — reveals what's coming |
| `revisiting_classes` | Which classes to measure forgetting on |
| `first_appearance_of` | Which classes are genuinely new |
| `overlap_with_previous` | `{prev_task: n_shared}` — overlap diagnostics |
| `image_indices` | Reproducibility manifest — records exact image assignments |
| `n_samples` | Convenience count |

The invariant `revisiting_classes ∪ first_appearance_of == classes_in_this_experience`
is enforced at construction time, making it impossible to construct an
inconsistent Experience.

---

## 5. How CLOVER differs from PILOT, Avalanche, and PyCIL

**PILOT / TOSCA** — Excellent CL toolboxes with clean training pipelines and
many reproduced methods. CLOVER's `OverlapDataManager` is API-compatible with
PILOT's `DataManager` (same class-ordering, same task splits when
`overlap_spec=None`). PILOT does not support class revisits, does not have a
stream abstraction, and does not expose revisit metadata.

**Avalanche** — Large, mature CL framework with a stream abstraction
(`AvalancheDataset`, `GenericExperienceStream`). Avalanche supports some forms
of data augmentation and class-incremental scenarios but does not provide
controlled class revisit with configurable image-overlap strategies or
declarative revisit specifications. Its experience objects do not carry
revisit-specific metadata.

**PyCIL** — Clean implementation of class-incremental algorithms. Data
management is similar to PILOT — task-based random access, no stream
abstraction, no revisit support.

**CLOVER v0.2** — Designed specifically to make class-revisit experiments
first-class: declarative specs, multi-seed support, and Experience objects
that make overlap-aware metrics trivial to compute.

---

## 6. Example: a multi-seed experiment with StreamSpec

```python
from clover import build_benchmark, StreamSpec, RevisitSpec

results = {"revisit_acc": [], "fresh_acc": []}

for seed in range(1, 6):
    spec = StreamSpec(
        dataset="cifar100",
        init_cls=10,
        increment=10,
        revisits=[
            RevisitSpec(classes=5, times=1, placement="random", min_gap=3)
        ],
        stream_seed=seed,      # vary this for different concrete streams
        shuffle_seed=1993,     # keep this fixed for PILOT-equivalent class order
    )
    bench = build_benchmark(spec)

    for exp in bench.train_stream:
        train_model(exp.dataset)  # your training loop

        if exp.is_revisit_experience():
            # Measure forgetting specifically on revisiting classes
            acc = evaluate_on_classes(exp.revisiting_classes)
            results["revisit_acc"].append(acc)
        else:
            acc = evaluate_on_classes(exp.first_appearance_of)
            results["fresh_acc"].append(acc)

print("Revisit accuracy:", mean(results["revisit_acc"]), "±", std(results["revisit_acc"]))
print("Fresh accuracy:", mean(results["fresh_acc"]), "±", std(results["fresh_acc"]))
```

The 6-experience sub-stream for stream_seed=42 might look like:

```
Exp 0: classes [0-9]      first_appearance=[0-9]    revisiting=[]
Exp 1: classes [10-19]    first_appearance=[10-19]  revisiting=[]
Exp 2: classes [20-29]    first_appearance=[20-29]  revisiting=[]
Exp 3: classes [3,30-38]  first_appearance=[30-38]  revisiting=[3]   ← revisit!
Exp 4: classes [40-49]    first_appearance=[40-49]  revisiting=[]
Exp 5: classes [7,22,50-57] first_appearance=[50-57] revisiting=[7,22] ← revisit!
```

With stream_seed=99 the same 5 classes revisit, but at different experience
indices. The statistical property (5 revisits, min_gap=3) is preserved.

---

## 7. Limitations and future work

- **Revisits must come after the first appearance.** CLOVER currently does not
  support placing a revisit *before* the natural first occurrence of a class.
  This covers the most common research scenario but excludes some prospective
  learning setups.

- **The `_n_classes_map` in `stream_builder.py` hardcodes class counts.** For
  custom datasets, use `OverlapDataManager` directly with an explicit `OverlapSpec`.

- **`preserve_task_size` evicts from the last non-shared class.** This is
  deterministic but may occasionally displace classes that a researcher
  intended to keep. Use `preserve_task_size=False` to opt out.

- **Image overlap strategies are per-OverlapSpec, not per-class.** All shared
  classes within a stream use the same `image_strategy`. Fine-grained per-class
  control is possible with the low-level `OverlapSpec` API.
