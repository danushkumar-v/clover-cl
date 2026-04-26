"""One-shot script: capture PILOT-equivalent class orderings as a golden fixture.

Run once from the repo root:
    python scripts/capture_pilot_fixtures.py

Outputs: tests/fixtures/pilot_class_orders.json

The class ordering is purely a function of numpy's legacy RNG:
    np.random.seed(seed); np.random.permutation(n_classes)
This exactly mirrors what PILOT's DataManager._setup_data() does when
shuffle=True. The fixture freezes this computation so future tests can
assert byte equality without importing PILOT itself.

Combinations captured:
    cifar100 / init_cls=10, increment=10 / seed=1993
    cifar100 / init_cls=10, increment=10 / seed=1
    cifar100 / init_cls=20, increment=20 / seed=1993
    cub200   / init_cls=20, increment=20 / seed=1993  (200 classes)
    imagenet_r / init_cls=20, increment=20 / seed=1993  (200 classes)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clover.core.task_builder import compute_increments  # noqa: E402


DATASET_N_CLASSES = {
    "cifar100": 100,
    "cub200": 200,
    "imagenet_r": 200,
}

CONFIGS = [
    ("cifar100", 100, 10, 10, 1993),
    ("cifar100", 100, 10, 10, 1),
    ("cifar100", 100, 20, 20, 1993),
    ("cub200", 200, 20, 20, 1993),
    ("imagenet_r", 200, 20, 20, 1993),
]


def _pilot_class_order(n_classes: int, seed: int) -> list:
    """Reproduce PILOT's exact class-order call."""
    np.random.seed(seed)
    return np.random.permutation(n_classes).tolist()


def _task_class_lists(n_classes: int, init_cls: int, increment: int,
                      class_order: list) -> list:
    incs = compute_increments(n_classes, init_cls, increment)
    tasks = []
    cursor = 0
    for inc in incs:
        tasks.append(list(range(cursor, cursor + inc)))
        cursor += inc
    return tasks


def main():
    out = {}
    for dataset, n_cls, init_cls, increment, seed in CONFIGS:
        key = f"{dataset}__init{init_cls}__inc{increment}__seed{seed}"
        order = _pilot_class_order(n_cls, seed)
        tasks = _task_class_lists(n_cls, init_cls, increment, order)
        incs = compute_increments(n_cls, init_cls, increment)
        out[key] = {
            "dataset": dataset,
            "n_classes": n_cls,
            "init_cls": init_cls,
            "increment": increment,
            "seed": seed,
            "class_order": order,
            "task_class_lists": tasks,
            "increments": incs,
        }
        print(f"  captured {key}: {len(tasks)} tasks, first order entry = {order[0]}")

    dest = REPO_ROOT / "tests" / "fixtures" / "pilot_class_orders.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {len(out)} fixtures to {dest}")


if __name__ == "__main__":
    main()
