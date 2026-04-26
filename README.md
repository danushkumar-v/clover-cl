<p align="center">
  <h1 align="center">CLOVER — Continual Learning with OVERlap</h1>
</p>

<p align="center">
  <a href="#-introduction">🎉 Introduction</a> •
  <a href="#-overlap-scenarios">🌟 Overlap Scenarios</a> •
  <a href="#%EF%B8%8F-how-to-use">☄️ How to Use</a> •
  <a href="#-datasets">🔎 Datasets</a> •
  <a href="#-acknowledgments">👨‍🏫 Acknowledgments</a> •
  <a href="#-contact">🤗 Contact</a>
</p>

---

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/CLOVER-v0.1.0-darkcyan"></a>
  <a href=""><img src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href=""><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange"></a>
  <a href=""><img src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</p>

## 🎉 Introduction

Welcome to **CLOVER**, a continual learning benchmark library where classes are **allowed to revisit across tasks**. CLOVER is a drop-in replacement for [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)'s `DataManager` that adds configurable, reproducible **class overlap** on top of standard class-incremental learning benchmarks.

Standard CL benchmarks enforce **strict disjoint** task partitions — once a class appears in Task 0, it never appears again. Real-world streaming data does not work that way. CLOVER makes overlap **controlled and reproducible**:

```
# Standard CL (PILOT / TOSCA)
Task 0: classes {0, 1, 2, 3, 4}
Task 1: classes {5, 6, 7, 8, 9}   ← "cat" never revisited

# CLOVER with partial_overlap
Task 0: classes {0, 1, 2, 3, 4}
Task 5: classes {2, 3, 8, 9, 10}  ← classes 2 & 3 revisit
```

When `overlap_spec=None`, CLOVER produces **identical class orderings** to PILOT (seed-1993 verified), making it a zero-cost drop-in for existing pipelines.

## 🌟 Overlap Scenarios

CLOVER ships 8 built-in scenarios covering the major ways classes can revisit:

| Scenario | Module | Description |
|---|---|---|
| `exact_replay` | `clover.scenarios.exact_replay` | Task N's classes are identical to Task 0's classes — tests pure catastrophic forgetting |
| `partial_overlap` | `clover.scenarios.partial_overlap` | A configurable fraction of Task N's classes appear in an earlier task |
| `hierarchical` | `clover.scenarios.hierarchical` | Classes sharing a taxonomic parent are distributed across tasks |
| `distribution_shift` | `clover.scenarios.distribution_shift` | Same classes appear in multiple tasks from different feature distributions |
| `near_miss` | `clover.scenarios.near_miss` | Tasks share no classes but contain visually adjacent classes (wolf/husky) |
| `long_range_revisit` | `clover.scenarios.long_range_revisit` | Task 0 and Task T−1 share classes; intermediate tasks are unrelated |
| `cumulative_drift` | `clover.scenarios.cumulative_drift` | Class set grows cumulatively; anchor classes appear in every task |
| `symmetric_pair` | `clover.scenarios.symmetric_pair` | Tasks M and N share 50% classes; each has 50% unique classes |

## ☄️ How to Use

### 🕹️ Clone

```bash
git clone https://github.com/danushkumar-v/clover-cl
cd clover-cl
```

### 🗂️ Dependencies

1. [torch ≥ 2.0](https://github.com/pytorch/pytorch)
2. [torchvision ≥ 0.15](https://github.com/pytorch/vision)
3. [numpy ≥ 1.24](https://github.com/numpy/numpy)
4. [Pillow ≥ 10.0](https://github.com/python-pillow/Pillow)
5. [PyYAML ≥ 6.0](https://github.com/yaml/pyyaml)
6. [matplotlib](https://matplotlib.org/) *(optional — for visualisations)*

Install in editable mode (recommended for research):

```bash
pip install -e .[dev]
```

### 🔑 Quickstart

**Baseline — identical to PILOT (no overlap):**

```python
from clover import OverlapDataManager

dm = OverlapDataManager("cifar100", init_cls=10, increment=10)
# overlap_spec=None → same class order as PILOT with seed=1993

train_set = dm.get_dataset(task_id=0, source="train", mode="train")
test_set  = dm.get_dataset(task_id=0, source="test",  mode="test")
```

**With overlap — partial_overlap scenario:**

```python
from clover import OverlapDataManager
from clover.scenarios import partial_overlap

spec = partial_overlap.build_spec(
    total_classes=100, init_cls=10, increment=10,
    pair=(0, 5), overlap_fraction=0.5,
)
dm = OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)

# DataLoader — yields (sample_idx, image, remapped_label) like PILOT's DummyDataset
import torch
loader = torch.utils.data.DataLoader(
    dm.get_dataset(task_id=0, source="train", mode="train"),
    batch_size=32, shuffle=True,
)
for sample_idx, images, labels in loader:
    ...
```

**Visualise and inspect:**

```python
from clover.utils.visualizer import plot_overlap_matrix, plot_class_frequency

plot_overlap_matrix(dm, "overlap_matrix.png")
plot_class_frequency(dm, "class_freq.png")
print(dm.get_overlap_matrix())   # numpy array, shape (T, T)
```

**CLI:**

```bash
python -m clover.cli inspect configs/cifar100_partial.yaml
```

### 🔧 Key Parameters

When configuring CLOVER, the main parameters are:

- **`dataset_name`**: One of `cifar100`, `cub200`, `imagenet_r`, `imagenet_a`, `omnibenchmark`, `vtab`.
- **`init_cls`**: Number of classes in the initial (Task 0) stage.
- **`increment`**: Number of new classes added per subsequent task. Matches PILOT's convention.
- **`shuffle_seed`**: Random seed for class order permutation. Default `1993` — matches PILOT's iCaRL benchmark baseline.
- **`overlap_spec`**: An `OverlapSpec` object (or `None` for PILOT-identical disjoint splits). Built via scenario helpers or directly.
- **`data_root`**: Root directory where datasets are stored. Default `./data`.

### 📋 Reproducibility via Manifest

Every run can be frozen to a JSON manifest — a record of exactly which image indices go to which task and class:

```python
dm.save_manifest("run_seed1993.json")

# Reload in another script
from clover.utils.manifest import load_manifest
manifest = load_manifest("run_seed1993.json")
# manifest[task_id][class_id] → list of image indices
```

### 🧪 Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=clover --cov-report=term
```

All 109 tests pass; coverage is ≥ 86%.

## 🔎 Datasets

- **CIFAR-100**: Downloaded automatically by the code.
- **CUB-200-2011**: Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **ImageNet-A**: Google Drive: [link](https://drive.google.com/file/d/19l52ua_vvTtttgVRziCZJjal0TPE9f2p/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/ERYi36eg9b1KkfEplgFTW3gBg1otwWwkQPSml0igWBC46A?e=NiTUkL)
- **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
- **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)

> Dataset links mirror the pre-processed subsets distributed by LAMDA-PILOT. These subsets are sampled from the original datasets. Download and unzip into your `data_root` directory (default `./data/`).

After downloading, set the data path when constructing the manager:

```python
dm = OverlapDataManager("cub200", init_cls=10, increment=10, data_root="/path/to/data")
```

## 👨‍🏫 Acknowledgments

CLOVER builds on top of the following open-source projects:

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT) — Pre-trained model-based CL toolbox whose DataManager API CLOVER mirrors
- [TOSCA](https://github.com/AAAI-25-TOSCA/TOSCA) — CL toolbox built on PILOT
- [PyCIL](https://github.com/G-U-N/PyCIL) — Class-incremental learning toolbox

## 🤗 Contact

If you have questions or want to propose new scenarios, please open an issue on GitHub.

## 📜 Citation

If you use CLOVER in your research, please cite:

```bibtex
@software{clover2026,
  title   = {CLOVER: Continual Learning with OVERlap},
  author  = {Danush Kumar Venkatesh},
  year    = {2026},
  url     = {https://github.com/danushkumar-v/clover-cl},
}
```
