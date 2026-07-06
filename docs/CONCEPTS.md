# CLOVER Concepts: Streams, Experiences, and Declarative Revisits

This document explains the design of CLOVER v2's Stream/Experience model,
why it was built, and how it differs from other CL benchmark libraries.
CLOVER v2 is a self-contained rewrite (SPEC.md R1) — LAMDA-PILOT is cited
as inspiration for method implementations, never vendored or depended on.

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

Methods that explicitly exploit class revisits have no standard benchmark
infrastructure to evaluate against. CLOVER fills that gap.

---

## 2. Random-access task lookup vs. temporal stream

Most CL libraries expose tasks as a random-access lookup: a training loop
asks for "dataset for classes X" by index, and does its own bookkeeping —
which classes has the model seen so far? Which classes in this task are
revisiting from earlier? Is it safe to evaluate on these samples?

CLOVER wraps a resolved benchmark in a temporal stream instead:

```python
for exp in benchmark.train_stream:
    train_on(exp)                          # exp carries everything needed
    track_revisiting_classes(exp.revisiting_classes)
    evaluate_only_on(exp.classes_seen_so_far)
```

The stream is iterable, indexable, and carries stream-level statistics
(`revisit_density()`, `average_overlap()`, `class_appearance_count()`).

---

## 3. `StreamSpec`/`RevisitSpec`: a declarative revisit description

`StreamSpec` describes a stream's *statistical structure*, not a hardcoded
list of task-to-class assignments. It resolves deterministically given
`(shuffle_seed, stream_seed)` — the same spec with a different `stream_seed`
gives a different *concrete* realization of the same structure, which is
what makes multi-seed error bars meaningful.

```yaml
stream:
  dataset: cifar100
  init_cls: 10
  increment: 10
  shuffle_seed: 1993   # drives the base class order (keep fixed across seeds)
  stream_seed: 42       # drives every other stochastic choice (vary this)
  revisits:
    - classes: {random: 5}    # pick 5 classes at random from what's appeared
      placement: random        # random | spaced | end_of_stream | clustered | every_task
      label: new                # new (fresh echo id) | same (keeps its original id)
      images: new                # same | new | partial:<pct>
      min_gap: 3                  # experiences between first appearance and any revisit
      times: 1                     # how many times each class revisits
```

`classes` also accepts the string shorthand `"task0"` (every class in the
first experience) or an explicit list of class ids. Whichever a revisit
picks, it can come back as:

- **an echo** (`label: new`) — a brand-new label id is minted for the
  reappearance, distinct from the class's original id. `echo_map`/
  `plan.echo_table` records `echo_id -> (source_id, image_relation)` so a
  method or metric can always trace an echo back to what it's really a
  repeat of.
- **a same-id revisit** (`label: same`) — the class keeps its original
  label id; no new head slot is ever allocated for it. This is the shape
  used by e.g. `cumulative_drift`'s anchor classes and
  `distribution_shift`'s shifted classes: the label space doesn't grow,
  only the image distribution behind that label does.

`images` controls whether the revisit's samples are the *same* images
(`same`, a pure duplicate), entirely *disjoint* images (`new`), or a
`partial:<pct>` split of both.

Rather than hand-writing `StreamSpec`s, most configs pick a named
**scenario** instead — a `@register_scenario` factory function that builds
one for you from `(dataset_info, init_cls, increment, seed, **params)`.
`clover/scenarios/*.py` ships 7: `disjoint_baseline`, `exact_replay`,
`partial_overlap`, `long_range_revisit`, `mid_range_revisit`,
`cumulative_drift`, `distribution_shift` (see `docs/extending.md` for how
to register your own).

---

## 4. The Experience object

An `Experience` (`clover/core/experience.py`) is a frozen, torch-free
dataclass that bundles per-step metadata — not a live `Dataset` (wrapping
`image_indices` into an actual per-experience `torch.utils.data.Dataset`
is `clover/training/trainer.py:ExperienceDataset`'s job, once real data
exists). Key fields:

| Field | Use case |
|---|---|
| `task_label` | Loop index; identifies position in stream |
| `classes_in_this_experience` | Classifier head sizing, logging |
| `classes_seen_so_far` | Safe evaluation boundary — never evaluate on future classes |
| `classes_in_future` | Diagnostic only — reveals what's coming |
| `revisiting_classes` | Which classes to measure forgetting/retention on |
| `first_appearance_of` | Which classes are genuinely new |
| `overlap_with_previous` | `{prev_task_label: n_shared}` — overlap diagnostics |
| `echo_map` | `{echo_id: EchoEntry(source_id, image_relation, ...)}` first introduced here |
| `label_space` | `LabelSpaceView` — set-membership masks for revisit-safe losses |
| `image_indices` | `{class_id: [image_indices]}` — the reproducibility manifest |
| `n_samples` | Convenience count |

The invariant `revisiting_classes ∪ first_appearance_of == classes_in_this_experience`
(and the two are disjoint) is enforced at construction time.

`label_space` (`LabelSpaceView`) is the one and only place revisit-safe
masking logic lives: `new_mask(targets)`/`old_mask(targets)`/
`logit_mask(width)` all build boolean masks from a materialized id *set*,
never from `targets >= known_classes` range arithmetic — that comparison
silently breaks the moment a same-id revisit re-presents an "old"
(numerically small) id in a later batch. `clover/methods/losses.py` is the
only place a method should ever touch this.

---

## 5. How CLOVER fits in the CL benchmark landscape

Continual-learning benchmarks have evolved through several waves:

1. **Disjoint class-incremental** (Split-MNIST, Split-CIFAR, PILOT).
   Each class appears in exactly one task. Most CL papers use this.
2. **Repetition-aware** (CIR, Hemati et al. 2023; i-Blurry, Si-Blurry).
   Classes can recur across the stream, controlled by interpretable
   parameters. This is the wave CLOVER lives in.
3. **Domain-incremental and online** (CORe50-NIC, CLAD, online OCL).
   Different axis; not CLOVER's focus.

CLOVER's specific niche within the repetition-aware wave is PTM-era
benchmark coverage (CIFAR-100, CUB-200, ImageNet-R, ImageNet-A,
OmniBenchmark, VTAB, plus arbitrary user datasets via `image_folder`) with
explicit, named revisit scenarios and CLOVER-specific overlap metrics
(Repetition Gain, Anchor/Long-Range Retention — see `docs/methods.md` and
`clover/evaluation/metrics/overlap.py`).

CLOVER is **not** the first library to support class revisits or stream-based
CL benchmarks. The canonical prior work is:

- **CIR** ([Hemati et al., CoLLAs 2023](https://arxiv.org/abs/2301.11396)) —
  stochastic CIR stream generators with interpretable control parameters,
  integrated with Avalanche. If your work targets CIFAR-100 or Tiny-ImageNet
  without needing CLOVER's PTM-era coverage, consider CIR directly.
- **Avalanche** ([Carta et al., JMLR 2023](https://www.jmlr.org/papers/v24/22-1280.html)) —
  the canonical Stream/Experience abstraction. CLOVER's stream model mirrors
  Avalanche's design but is self-contained (SPEC.md R1) to avoid the full
  Avalanche dependency.
- **i-Blurry / Si-Blurry** ([Koh et al., ICLR 2022](https://arxiv.org/abs/2110.10031);
  [Moon et al., ICCV 2023](https://arxiv.org/abs/2308.09303)) — class overlap
  via "blurry task boundary" scenarios.
- **LAMDA-PILOT** — inspiration for CLOVER's method implementations
  (SimpleCIL, L2P, DualPrompt, CODA-Prompt, APER-Adapter, RanPAC, EASE, MOS,
  TUNA); no code dependency (SPEC.md R1).

---

## 6. Example: a multi-seed experiment with `StreamSpec`

```python
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets import get_dataset

dataset_cls = get_dataset("cifar100")
train_dataset = dataset_cls(root="./data", train=True)
test_dataset = dataset_cls(root="./data", train=False)
info = DatasetInfo("cifar100", train_dataset.num_classes)

results = {"revisit_acc": [], "fresh_acc": []}

for seed in range(1, 6):
    spec = StreamSpec(
        dataset="cifar100",
        init_cls=10,
        increment=10,
        revisits=[RevisitSpec(classes={"random": 5}, times=1, placement="random", min_gap=3)],
        stream_seed=seed,      # vary this for different concrete streams
        shuffle_seed=1993,     # keep this fixed for a consistent base class order
    )
    benchmark = build_benchmark(
        spec, info, train_dataset.get_class_to_indices(), test_dataset.get_class_to_indices()
    )

    for exp in benchmark.train_stream:
        train_model(exp)  # your training loop

        if exp.is_revisit_experience():
            acc = evaluate_on_classes(exp.revisiting_classes)
            results["revisit_acc"].append(acc)
        else:
            acc = evaluate_on_classes(exp.first_appearance_of)
            results["fresh_acc"].append(acc)

print("Revisit accuracy:", mean(results["revisit_acc"]), "±", std(results["revisit_acc"]))
print("Fresh accuracy:", mean(results["fresh_acc"]), "±", std(results["fresh_acc"]))
```

The 6-experience sub-stream for `stream_seed=42` might look like:

```
Exp 0: classes [0-9]      first_appearance=[0-9]    revisiting=[]
Exp 1: classes [10-19]    first_appearance=[10-19]  revisiting=[]
Exp 2: classes [20-29]    first_appearance=[20-29]  revisiting=[]
Exp 3: classes [3,30-38]  first_appearance=[30-38]  revisiting=[3]   ← revisit!
Exp 4: classes [40-49]    first_appearance=[40-49]  revisiting=[]
Exp 5: classes [7,22,50-57] first_appearance=[50-57] revisiting=[7,22] ← revisit!
```

With `stream_seed=99` the same 5 classes revisit, but at different
experience indices. The statistical property (5 revisits, `min_gap=3`) is
preserved.

For a config-driven run instead of hand-writing a `StreamSpec`, use a named
scenario:

```yaml
stream:
  dataset: cifar100
  init_cls: 10
  increment: 10
  scenario: partial_overlap
  scenario_params: {overlap_fraction: 0.5}
method: {name: simplecil}
```

```
clover run configs/my_run.yaml
```

---

## 7. Dataset staging

CLOVER datasets own their own input pipeline (SPEC §7): each declares
`input_size` and `train_trsf`/`test_trsf`/`common_trsf` transform presets,
so a ViT-oriented method requests `224` and the dataset provides it —
`clover.cli._apply_default_transforms` composes `train_trsf + common_trsf`
(and `test_trsf + common_trsf`) into the actual `torchvision.transforms
.Compose` used at load time.

- **`synthetic`** — no staging; procedurally generated, used by `clover
  smoke` and every CPU-only local test.
- **`cifar100`** — auto-downloads via `torchvision.datasets.CIFAR100` on
  first use; no manual staging.
- **`cub200` / `imagenet_r` / `imagenet_a` / `omnibenchmark` / `vtab`** —
  manually staged `torchvision.datasets.ImageFolder`-shaped directories,
  each under its own subdirectory of `stream.data_root` (default
  `./data`): `<data_root>/<dataset_dir>/{train,test}/<class>/...`.

  | Dataset | `dataset_dir` | classes | download |
  |---|---|---|---|
  | `cub200` | `cub` | 200 | https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb |
  | `imagenet_r` | `imagenet-r` | 200 | https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR |
  | `imagenet_a` | `imagenet-a` | 200 | https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p |
  | `omnibenchmark` | `omnibenchmark` | 300 | https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u |
  | `vtab` | `vtab-cil/vtab` | 50 | https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_ |

  A missing directory raises `FileNotFoundError` naming the exact expected
  path and the download URL — nothing silently falls back to a different
  dataset.

- **`image_folder`** — a config-only dataset for arbitrary user data
  (SPEC §7): no `dataset_dir` nesting, no fixed class count. Point it at
  any `<root>/{train,test}/<class>/...` directory:

  ```yaml
  stream:
    dataset: {type: image_folder, root: /path/to/my_dataset, num_classes: 12}
    init_cls: 4
    increment: 4
  ```

  `num_classes` is required (the dataset can't be constructed with zero
  args to read it off a registry constant the way the 5 named built-ins
  can) and is cross-checked against the actual on-disk class-subdirectory
  count at load time — a mismatch raises immediately rather than silently
  training on the wrong number of classes.

---

## 8. Limitations

- **Revisits must come after the first appearance.** CLOVER does not
  support placing a revisit *before* the natural first occurrence of a
  class. This covers the common research scenarios but excludes some
  prospective-learning setups.
- **Image overlap strategies are per-`RevisitSpec`, not per-class.** All
  classes named by one `RevisitSpec` share the same `images` relation;
  finer per-class control means writing more than one `RevisitSpec`.
- **`hierarchical`/`near_miss`/`symmetric_pair`-style scenarios are out of
  scope for v2.0.** `RevisitSpec`'s class-to-task assignment is purely
  sequential/budget-driven (`init_cls`/`increment`); it has no hook for
  external taxonomy metadata (a `hierarchical` scenario's requirement) or
  adjacency bookkeeping with zero class sharing (`near_miss`'s v1 shape —
  indistinguishable from `disjoint_baseline` under v2's model without it).
  `symmetric_pair`'s only expressible direction (a later task echoing an
  earlier one, never the reverse — the reverse would leak future
  information backward) is already exactly `partial_overlap
  (overlap_fraction=0.5)`; it isn't separately registered because it would
  just be a redundant alias with no new mechanism. See `docs/extending.md`
  for how to add any of these yourself if your research needs them.
