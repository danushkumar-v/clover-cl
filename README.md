# CLOVER — Continual Learning with OVERlap

CLOVER is a standalone Python library providing continual-learning dataset benchmarks where classes are **allowed to revisit** across tasks. It is a drop-in replacement for PILOT's `DataManager` that adds configurable, reproducible class overlap.

## Why CLOVER?

Standard CL benchmarks enforce **strict disjoint** task partitions:

```
# PILOT / standard CL
Task 0: classes {0, 1, 2, 3, 4}
Task 1: classes {5, 6, 7, 8, 9}   # "cat" never seen again
Task 2: classes {10, 11, ...}
```

Real-world streaming data does not work that way — the same concept can appear in multiple deployment windows. CLOVER makes overlap **controlled and configurable**:

```
# CLOVER with partial_overlap
Task 0: classes {0, 1, 2, 3, 4}
Task 5: classes {2, 3, 8, 9, 10}  # classes 2 & 3 revisit
```

## Install

```bash
git clone https://github.com/your-org/clover-cl
cd clover-cl
pip install -e .[dev]
```

## Quickstart

```python
from clover import OverlapDataManager
from clover.scenarios import partial_overlap

spec = partial_overlap.build_spec(
    total_classes=100, init_cls=10, increment=10,
    pair=(0, 5), overlap_fraction=0.5
)
dm = OverlapDataManager("cifar100", init_cls=10, increment=10, overlap_spec=spec)

# Iterate task 0 training data
dataset = dm.get_dataset(task_id=0, source="train", mode="train")
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
for sample_idx, images, labels in loader:
    ...

print(dm.get_overlap_matrix())
```

## The 8 Overlap Scenarios

| Scenario | Description |
|---|---|
| `exact_replay` | Task N's classes are identical to Task 0's classes |
| `partial_overlap` | A configurable fraction of Task N's classes appear in an earlier task |
| `hierarchical` | Classes sharing a taxonomic parent are distributed across tasks |
| `distribution_shift` | Same classes appear in multiple tasks but from different feature distributions |
| `near_miss` | Tasks share no classes but contain visually adjacent classes (wolf/husky) |
| `long_range_revisit` | Task 0 and Task T-1 share classes; intermediate tasks are unrelated |
| `cumulative_drift` | Class set grows cumulatively; anchor classes appear in every task |
| `symmetric_pair` | Tasks M and N share 50% classes; each has 50% unique classes |

## Reproducibility

Every `OverlapDataManager` can emit a **manifest** — a JSON record of which image indices go to which task and class:

```python
dm.save_manifest("run_42.json")
```

## Citation

```bibtex
@software{clover2024,
  title  = {CLOVER: Continual Learning with OVERlap},
  year   = {2024},
}
```
