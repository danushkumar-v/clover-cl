# Overlap Scenarios

Each module here implements one overlap regime. Every module exports:

- `build_spec(...)` — returns an `OverlapSpec` (low-level / legacy API)
- `build_stream_spec(...)` — returns a `StreamSpec` (modern API)

Use `build_stream_spec` for new research where you need multi-seed
reproducibility. Use `build_spec` when you need exact task-pair control or
are integrating with the v0.1 engine directly.

For the full explanation of the Stream/Experience model, see
[../docs/CONCEPTS.md](../docs/CONCEPTS.md).

---

## Scenario reference

### `exact_replay`
Task 0's entire class set reappears verbatim at the final task. Tests whether
a model can recover from catastrophic forgetting given an exact repetition.
Use `build_stream_spec` for a declarative version that places all of Task 0's
classes at the end of the stream.

### `partial_overlap`
A configurable fraction of one task's classes reappear in another. The primary
scenario for studying controlled partial overlap. With `build_stream_spec`, the
specific classes and placement are resolved from `stream_seed`, enabling
multi-seed experiments.

### `hierarchical`
Classes sharing a taxonomic parent (e.g., all Finches, all Hawks) are
distributed across two tasks. Requires a `hierarchy_map` for the low-level
`build_spec`. With `build_stream_spec`, N random classes from the feasible
pool are chosen and placed according to the `placement` strategy.

### `distribution_shift`
The same classes appear in multiple tasks but each task receives a disjoint
image slice. Models the same concept encountered under different capture
conditions. The `build_spec` API gives exact control over the split ratio.

### `near_miss`
Tasks share no classes but contain visually adjacent classes (e.g., wolf /
husky, crocodile / alligator). Tests robustness to semantic proximity without
true class overlap. `shared_classes` is always empty; the proximity is
structural, not in the image assignment.

### `long_range_revisit`
Task 0 and the final task share classes; all intermediate tasks are unrelated.
Tests whether a model can re-activate knowledge from the distant past.

### `cumulative_drift`
Anchor classes appear in every consecutive task pair alongside new classes.
The class set grows cumulatively. Tests whether a model can maintain knowledge
of a stable core concept while the surrounding context changes.

### `symmetric_pair`
Tasks M and N share exactly 50% of their classes; each retains 50% unique
classes. Provides a clean controlled overlap baseline for ablation studies.
