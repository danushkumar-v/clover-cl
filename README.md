<p align="center">
  <h1 align="center">CLOVER — Continual Learning with OVERlap</h1>
</p>

<p align="center">
  <a href="#what-is-clover">What is CLOVER?</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#the-smoke--slurm-workflow">Workflow</a> •
  <a href="#scenarios">Scenarios</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#extending-clover">Extending</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/CLOVER-v2.0.0.dev0-darkcyan"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href=""><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-lightgrey"></a>
</p>

## What is CLOVER?

CLOVER is a config-driven continual-learning benchmark for **class-revisit
scenarios**: streams where a class can reappear later — under the same
label, under a fresh one, with the same images or different ones — instead
of the standard disjoint-task assumption where each class appears exactly
once. One YAML config plus one command runs a full experiment: dataset →
revisit scenario → CL method → training → metrics → report.

```
clover run configs/cifar100_exact_replay_l2p.yaml
```

CLOVER v2 is a **self-contained, clean-room rewrite** — zero code
dependency on [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT).
PILOT is cited as inspiration for method implementations and the
PTM-era benchmark suite; no code is vendored or copied from it, or from any
other project referenced below. See [`docs/CONCEPTS.md`](docs/CONCEPTS.md)
for the full design story, or the table of contents above for a quicker tour.

> Coming from v1 (the `OverlapDataManager`/`OverlapSpec` API, currently on
> the `main` branch, to become the `v1` branch at release)? See
> [`docs/MIGRATION.md`](docs/MIGRATION.md).

---

## Quickstart

```yaml
# configs/my_run.yaml
stream:
  dataset: cifar100
  scenario: exact_replay     # Task N repeats Task 0's classes, same images
  init_cls: 10
  increment: 10
method:
  name: simplecil            # frozen backbone + cosine-prototype head
training:
  epochs: 5
  batch_size: 16
```

```
pip install -e .
clover run configs/my_run.yaml
```

Every run writes a self-contained directory (`runs/my_run/` by default):
`config_resolved.yaml` (the fully layered config actually used —
reproduce the run from this file alone), `manifest.json` (the resolved
stream plan: exact class order and image assignment), `status.json`
(state/heartbeat, for resume), `per_task.csv` (every metric, every
experience), `R_matrix.npy`, `ckpt_task*.pt` (resumable checkpoints),
`log.txt`.

Other entry points:

```
clover inspect configs/my_run.yaml     # print the resolved stream plan, no training
clover preflight configs/my_run.yaml   # validate a config before a long run
clover smoke                           # every registered method, on a tiny CPU-only synthetic dataset
clover report runs/my_run runs/other   # aggregate one or more runs into a summary CSV
clover run-matrix configs/matrix_full.yaml   # sweep methods x scenarios x datasets x seeds
```

Real (3-channel) datasets — CIFAR-100, the `ImageFolder`-backed built-ins,
`image_folder` — train correctly through every method (P10 fixed channel
-count threading end-to-end). Every method still defaults to its own
tiny, untrained, CPU-only stand-in backbone (`tiny_mlp`/`tiny_vit`) unless
overridden — fine for shape-correctness dry runs, not for a real accuracy
comparison. **SimpleCIL is currently the only one of the 9 methods that
can be pointed at a real pretrained timm ViT directly**
(`method: {name: simplecil, backbone: vit_base_patch16_224, pretrained:
true}`); the other 8 methods' prompt/adapter wrapper mechanisms need their
base to implement CLOVER-specific hooks that a raw timm ViT doesn't have —
splicing prompts/prefixes/adapters into real timm ViT internals is future
work, not yet done (see `PLAN.md`'s P10 log entry).

---

## The smoke → SLURM workflow

Development machines are assumed to have **no GPU**. The whole local
workflow runs on a tiny built-in synthetic dataset, CPU-only, in seconds —
real training happens on a cluster:

1. **Edit a config**, then run it locally on the CPU-only smoke profile
   (overrides the dataset onto `synthetic` with a minimal budget, keeping
   your chosen method/scenario mechanism and code paths):
   ```
   clover run configs/my_run.yaml --profile smoke
   ```
2. **Submit the real config** to a SLURM cluster (auto-detects and uses a
   GPU if the node has one — `clover run` never has to be told):
   ```
   sbatch scripts/slurm_run.sh configs/my_run.yaml
   ```
   Fill in `scripts/slurm_run.sh`'s `#SBATCH` placeholders (account,
   partition, time budget) for your site first. A preempted or
   timed-out job resumes automatically on resubmission — no `--resume`
   flag needed; `Trainer` detects existing checkpoints in the run directory.
3. **Sweep a whole grid** instead of one config, with the same
   resume/retry/stale-detection machinery:
   ```
   sbatch scripts/slurm_run.sh configs/matrix_full.yaml
   ```
4. **Aggregate results** once runs land (locally, or synced from the
   cluster):
   ```
   clover report runs/*
   ```

`clover smoke` is the pre-submission gate — a non-finite loss or crash on
the synthetic dataset blocks submission before any cluster time is spent.

---

## Scenarios

Every scenario is a `@register_scenario`-decorated factory: `(dataset_info,
init_cls, increment, seed, **params) -> StreamSpec`. 7 ship built-in:

| Scenario | Mechanism |
|---|---|
| `disjoint_baseline` | No revisits — the standard disjoint class-incremental baseline |
| `exact_replay` | Task 0's classes reappear once, at the end, fresh label id, identical images |
| `long_range_revisit` | Like `exact_replay` but with disjoint (new) images for the reappearance |
| `mid_range_revisit` | An *interior* task's classes (not task 0's) reappear at the end, fresh label id |
| `partial_overlap` | The final task mixes fresh classes with echoes of task 0 (`overlap_fraction` configurable) |
| `cumulative_drift` | A small anchor set keeps its *original* label id and recurs in *every* later task |
| `distribution_shift` | Like `cumulative_drift` but fires *once*, at the end, on a different image subset (`split_ratio` configurable) |

See [`docs/CONCEPTS.md`](docs/CONCEPTS.md#3-streamspecrevisitspec-a-declarative-revisit-description)
for the full `RevisitSpec` axes (`classes`/`placement`/`label`/`images`/
`min_gap`/`times`) these scenarios compose, and its §8 for why
`symmetric_pair`/`near_miss`/`hierarchical` (present in v1) aren't
separately implemented in v2 — each has a precise technical reason, not a
silent omission.

---

## Datasets

| Dataset | Classes | Staging |
|---|---|---|
| `synthetic` | 20 | None — procedurally generated, powers `clover smoke` and local dev |
| `cifar100` | 100 | Auto-downloads via `torchvision.datasets.CIFAR100` |
| `cub200` | 200 | Manual — [download](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb) into `<data_root>/cub/{train,test}/<class>/` |
| `imagenet_r` | 200 | Manual — [download](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR) into `<data_root>/imagenet-r/{train,test}/<class>/` |
| `imagenet_a` | 200 | Manual — [download](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p) into `<data_root>/imagenet-a/{train,test}/<class>/` |
| `omnibenchmark` | 300 | Manual — [download](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u) into `<data_root>/omnibenchmark/{train,test}/<class>/` |
| `vtab` | 50 | Manual — [download](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_) into `<data_root>/vtab-cil/vtab/{train,test}/<class>/` |
| `image_folder` | user-declared | **No new code.** `dataset: {type: image_folder, root: ..., num_classes: ...}` — any `<root>/{train,test}/<class>/` directory |

A missing manual-staging directory raises `FileNotFoundError` naming the
exact expected path and the download URL — nothing silently falls back to
a different dataset. See [`docs/CONCEPTS.md`](docs/CONCEPTS.md#7-dataset-staging)
for the full staging section.

---

## Extending CLOVER

New methods, datasets, scenarios, and backbones are added by implementing
a small interface and registering it with a decorator — core never imports
a plugin by name. [`docs/extending.md`](docs/extending.md) has one
complete, runnable worked example of each (using `image_folder` and
`distribution_shift` as the dataset/scenario examples, since they need the
least ceremony to follow along with).

---

## Results

Published-accuracy comparisons against each method's original paper are
pending a real cluster run of [`configs/matrix_full.yaml`](configs/matrix_full.yaml)
(needs both a GPU cluster and P10's backbone fix landing first — see the
known gap above). [`docs/methods.md`](docs/methods.md) tracks per-method
implementation notes, hyperparameter provenance, and known deviations from
each published method in the meantime.

---

## Running tests

```
pytest tests/ -v
pytest tests/ --cov=clover --cov-report=term-missing
ruff check clover tests
mypy clover/core clover/config
```

All tests are CPU-only and require no dataset downloads or GPU (real
datasets are monkeypatched in tests — see `tests/conftest.py`).

---

## Citation

If you use CLOVER, please cite this repository and the foundational CIR
work its revisit framework builds on conceptually:

```bibtex
@software{clover2026,
  title   = {CLOVER: Continual Learning with OVERlap},
  author  = {Danushkumar Venkadesh},
  year    = {2026},
  url     = {https://github.com/danushkumar-v/clover-cl},
}

@inproceedings{hemati2023class,
  title     = {Class-Incremental Learning with Repetition},
  author    = {Hemati, Hamed and Cossu, Andrea and Carta, Antonio and
               Hurtado, Julio and Pellegrini, Lorenzo and Bacciu, Davide
               and Lomonaco, Vincenzo and Borth, Damian},
  booktitle = {Proceedings of The 2nd Conference on Lifelong Learning Agents},
  pages     = {437--455},
  year      = {2023},
  publisher = {PMLR},
  volume    = {232},
}

@article{carta2023avalanche,
  title   = {Avalanche: A {PyTorch} Library for Deep Continual Learning},
  author  = {Carta, Antonio and Pellegrini, Lorenzo and Cossu, Andrea
             and Hemati, Hamed and Lomonaco, Vincenzo},
  journal = {Journal of Machine Learning Research},
  volume  = {24},
  number  = {363},
  pages   = {1--6},
  year    = {2023},
}
```

---

## Acknowledgments and attribution

CLOVER v2 is released under the **MIT license** and contains **no code
copied from any of the projects below** — each is cited for the ideas it
contributed, not reused as a dependency:

- [CIR (Hemati et al., 2023)](https://github.com/HamedHemati/CIR) — the
  conceptual parent of CLOVER's revisit framework.
- [Avalanche](https://github.com/ContinualAI/avalanche) — the canonical
  Stream/Experience abstraction in CL; CLOVER's stream model mirrors its
  design but is self-contained, avoiding the full Avalanche dependency.
- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) — inspiration
  for CLOVER's 9 method implementations (SimpleCIL, L2P, DualPrompt,
  CODA-Prompt, APER-Adapter, RanPAC, EASE, MOS, TUNA) and its PTM-era
  benchmark dataset suite.
