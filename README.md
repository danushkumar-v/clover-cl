<p align="center">
  <h1 align="center">CLOVER — Continual Learning with OVERlap</h1>
</p>

<p align="center">
  <a href="#-what-is-clover">🎉 What is CLOVER?</a> •
  <a href="#-relation-to-prior-work">📚 Prior Work</a> •
  <a href="#-quickstart-modern-api">☄️ Quickstart</a> •
  <a href="#-two-apis">⚖️ Two APIs</a> •
  <a href="#-key-concepts">📖 Key Concepts</a> •
  <a href="#-overlap-scenarios">🌟 Scenarios</a> •
  <a href="#-datasets">🔎 Datasets</a> •
  <a href="#-reproducibility">🔁 Reproducibility</a> •
  <a href="#-citation">📜 Citation</a>
</p>

---

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/CLOVER-v0.2.0-darkcyan"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href=""><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange"></a>
  <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</p>

## 🎉 What is CLOVER?

CLOVER is a continual-learning benchmark library that adds configurable
class overlap to the pre-trained-model CL benchmark suite (CIFAR-100,
CUB-200, ImageNet-R, ImageNet-A, OmniBenchmark, VTAB). It is a drop-in
replacement for [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)'s
`DataManager`, so PILOT and TOSCA training loops work unchanged.

CLOVER builds on the Class-Incremental with Repetition (CIR) framework
introduced by [Hemati et al. (2023)](https://arxiv.org/abs/2301.11396),
extending it to the PTM-era benchmark suite and adding explicit
image-level overlap modes (disjoint / duplicate / partial_duplicate)
needed to study methods that distinguish "same class, same images"
from "same class, different images."

> New to the Stream/Experience model? Read
> [`clover/core/README.md`](clover/core/README.md) for a quick tour
> or [`docs/CONCEPTS.md`](docs/CONCEPTS.md) for the full explanation.

---

## 📚 Relation to prior work

CLOVER is **not** the first library to support class revisits or
stream-based CL benchmarks. Several established projects already
cover this ground:

- **CIR** ([Hemati et al., CoLLAs 2023](https://arxiv.org/abs/2301.11396),
  [code](https://github.com/HamedHemati/CIR)) introduced two stochastic
  stream generators (Gslot, Gsamp) that produce CIR scenarios from any
  classification dataset using interpretable control parameters
  (first-occurrence distribution, repetition probability). CIR is
  integrated with Avalanche.
- **Avalanche** ([Lomonaco et al., 2021](https://arxiv.org/abs/2104.00405);
  [Carta et al., JMLR 2023](https://www.jmlr.org/papers/v24/22-1280.html))
  provides the canonical Stream + Experience abstraction with per-experience
  metadata (`classes_in_this_experience`, `task_labels`, etc.).
- **i-Blurry / Si-Blurry** ([Koh et al., ICLR 2022](https://arxiv.org/abs/2110.10031);
  [Moon et al., ICCV 2023](https://arxiv.org/abs/2308.09303)) address
  class overlap via "blurry task boundary" scenarios.

What CLOVER adds on top is **infrastructure**, not paradigm:

- **Drop-in PILOT compatibility.** When `overlap_spec=None`, CLOVER
  produces byte-identical task splits to PILOT (seed-1993 verified),
  so PILOT/TOSCA pipelines run unchanged.
- **PTM benchmark coverage.** Prior CIR work targets CIFAR-100 and
  Tiny-ImageNet. CLOVER covers the full PTM suite: CUB-200, ImageNet-R,
  ImageNet-A, OmniBenchmark, VTAB.
- **Image-level overlap modes.** Three explicit strategies for shared
  classes — disjoint (no image in two tasks), duplicate (all images
  in both), partial_duplicate (configurable overlap fraction) — useful
  for methods that care about same-image vs. same-class repetition.
- **Reproducibility manifests.** Every run can be frozen to a JSON
  manifest recording exact image-to-task assignments.

If your work does not need PTM benchmark coverage or PILOT drop-in
compatibility, **use CIR or Avalanche directly** — those are the
canonical libraries for CL with class repetition.

---

## ☄️ Quickstart (modern API)

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

for exp in bench.train_stream:
    print(exp.task_label,
          exp.classes_in_this_experience,
          "revisits:", exp.revisiting_classes)
```

## ☄️ Quickstart (legacy / drop-in API)

```python
from clover import OverlapDataManager

# overlap_spec=None → identical class order to PILOT, seed 1993
dm = OverlapDataManager("cifar100", init_cls=10, increment=10)

train_set = dm.get_dataset(task_id=0, source="train", mode="train")
test_set  = dm.get_dataset(task_id=0, source="test",  mode="test")
print(dm.nb_tasks, dm.get_task_classes(0))
```

---

## ⚖️ Two APIs — when to use which

| | Modern (`build_benchmark`) | Legacy (`OverlapDataManager`) |
|---|---|---|
| **Recommended for** | New research, multi-seed experiments, overlap-aware metrics | Direct PILOT/TOSCA compatibility, low-level engine access |
| **Overlap config** | Declarative `StreamSpec` — describe structure, not exact pairs | Explicit `OverlapSpec` — declare exact task pairs |
| **Multi-seed support** | Yes — vary `stream_seed` | No — placement is fixed |
| **Revisit metadata** | Per-`Experience` (revisiting_classes, overlap_with_previous, …) | Not exposed |
| **PILOT-equivalent baseline** | `StreamSpec(revisits=[])` | `OverlapDataManager(overlap_spec=None)` |

---

## 📖 Key Concepts

**Stream** — An ordered sequence of Experiences representing a temporal CL training sequence. → deep dive: [docs/CONCEPTS.md](docs/CONCEPTS.md#2-random-access-task-lookup-vs-temporal-stream)

**Experience** — One step in a Stream. A Dataset *plus* metadata: which classes are revisiting, which are new, how this experience overlaps with previous ones. → implementation: [clover/core/README.md](clover/core/README.md#what-an-experience-carries)

**StreamSpec** — A declarative description of stream structure. Specifies what revisits should happen and how they should be placed; the framework resolves concrete task-pair assignments from the spec and `stream_seed`. → [docs/CONCEPTS.md](docs/CONCEPTS.md#3-task-pair-declaration-vs-declarative-stream-spec)

**RevisitSpec** — A single revisit pattern inside a `StreamSpec`: which classes, how many times, where to place them. → [docs/CONCEPTS.md](docs/CONCEPTS.md#4-the-experience-object)

---

## 🌟 Overlap Scenarios

Each scenario ships two helpers:
- `build_spec(...)` — returns an `OverlapSpec` (legacy / low-level API)
- `build_stream_spec(...)` — returns a `StreamSpec` (modern API)

| Scenario | Module | <div style="width: 350px">Description</div> | Stream-spec helper |
|---|---|---|---|
| `exact_replay` | `clover.scenarios.exact_replay` | Task N's classes are identical to Task 0's classes | `exact_replay.build_stream_spec` |
| `partial_overlap` | `clover.scenarios.partial_overlap` | A configurable fraction of Task N's classes appear in an earlier task | `partial_overlap.build_stream_spec` |
| `hierarchical` | `clover.scenarios.hierarchical` | Classes sharing a taxonomic parent are distributed across tasks | `hierarchical.build_stream_spec` |
| `distribution_shift` | `clover.scenarios.distribution_shift` | Same classes appear in multiple tasks from different feature distributions | `distribution_shift.build_stream_spec` |
| `near_miss` | `clover.scenarios.near_miss` | Tasks share no classes but contain visually adjacent classes (wolf/husky) | `near_miss.build_stream_spec` |
| `long_range_revisit` | `clover.scenarios.long_range_revisit` | Task 0 and Task T−1 share classes; intermediate tasks are unrelated | `long_range_revisit.build_stream_spec` |
| `cumulative_drift` | `clover.scenarios.cumulative_drift` | Anchor classes appear in every task alongside new classes | `cumulative_drift.build_stream_spec` |
| `symmetric_pair` | `clover.scenarios.symmetric_pair` | Tasks M and N share 50% classes; each retains 50% unique classes | `symmetric_pair.build_stream_spec` |

See [`clover/scenarios/README.md`](clover/scenarios/README.md) for guidance on when to use each.

---

## 🔎 Datasets

- **CIFAR-100**: Downloaded automatically by the code.
- **CUB-200-2011**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ImageNet-A**: Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
- **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
- **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)

> Dataset links mirror the pre-processed subsets distributed by LAMDA-PILOT. Download and unzip into your `data_root` directory (default `./data/`).

```python
dm = OverlapDataManager("cub200", init_cls=20, increment=20, data_root="/path/to/data")
```

---

## 🔁 Reproducibility

CLOVER has **two independent seeds**:

- `shuffle_seed` (default 1993) — controls PILOT-compatible class-order permutation. Setting `shuffle_seed=1993` reproduces PILOT's benchmark class ordering exactly.
- `stream_seed` — controls revisit placement and random class selection. Varying `stream_seed` produces different concrete streams with the same statistical structure.

**Manifest** — every run can be frozen:

```python
bench.save_manifest("run_seed42.json")
# or for the engine directly:
dm.save_manifest("run_baseline.json")
```

**Multi-seed experiment pattern:**

```python
results = []
for seed in [1, 2, 3, 4, 5]:
    spec = StreamSpec(
        dataset="cifar100", init_cls=10, increment=10,
        revisits=[RevisitSpec(classes=5, times=1, placement="random", min_gap=2)],
        stream_seed=seed,
    )
    bench = build_benchmark(spec)
    results.append(run_training(bench))  # your training loop
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=clover --cov-report=term   # coverage ≥ 87%
```

---

## 📜 Citation

If you use CLOVER, please cite both this repository and the foundational
CIR work it builds upon:

```bibtex
@software{clover2026,
  title   = {CLOVER: Continual Learning with OVERlap},
  author  = {Danush Kumar Venkatesh},
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

## 👨‍🏫 Acknowledgments

CLOVER stands on prior work in continual-learning benchmarks:

- [CIR (Hemati et al., 2023)](https://github.com/HamedHemati/CIR)
  — the conceptual parent of CLOVER's revisit framework.
- [Avalanche](https://github.com/ContinualAI/avalanche) — the canonical
  Stream/Experience abstraction in CL.
- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) — the
  DataManager API CLOVER preserves.
- [TOSCA](https://github.com/AAAI-25-TOSCA/TOSCA) — the immediate
  downstream use-case driving CLOVER's design.
- [PyCIL](https://github.com/G-U-N/PyCIL) — class-incremental learning toolbox.
