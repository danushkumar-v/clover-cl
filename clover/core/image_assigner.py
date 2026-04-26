"""Pure-function image assignment logic.

Decides which image indices go to which task for each class, respecting
the configured :class:`~clover.core.overlap_spec.ImageSplit` strategy.
No I/O; all randomness flows through the supplied ``rng``.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from clover.core.overlap_spec import ImageSplit, OverlapPair


def assign_images(
    class_to_image_indices: Dict[int, List[int]],
    task_class_lists: List[List[int]],
    overlap_pairs: List[OverlapPair],
    image_split: ImageSplit,
    rng: np.random.Generator,
) -> List[Dict[int, List[int]]]:
    """Assign image indices to tasks respecting the overlap configuration.

    For classes that appear in only one task, all images go to that task.
    For shared classes the :class:`~clover.core.overlap_spec.ImageSplit`
    strategy controls the split.

    Args:
        class_to_image_indices: Mapping from *remapped* class ID to the list of
            image indices (into the flat dataset array) that belong to that class.
        task_class_lists: Per-task list of remapped class IDs.
        overlap_pairs: Declared shared-class relationships between task pairs.
        image_split: Strategy for dividing images of shared classes.
        rng: Seeded :class:`numpy.random.Generator`; all randomness goes here.

    Returns:
        A list of dicts, one per task: ``{class_id: [image_indices]}``.
    """
    n_tasks = len(task_class_lists)

    # Build a map: class_id -> set of task indices that contain it
    class_to_tasks: Dict[int, List[int]] = {}
    for t_idx, cls_list in enumerate(task_class_lists):
        for cls in cls_list:
            class_to_tasks.setdefault(cls, []).append(t_idx)

    # Build a map: (task_a, task_b) -> shared_classes from declared pairs
    pair_shared: Dict[tuple, List[int]] = {}
    for pair in overlap_pairs:
        key = tuple(sorted(pair.tasks))
        pair_shared[key] = list(pair.shared_classes)

    # Initialise result: per task, empty dict
    result: List[Dict[int, List[int]]] = [{} for _ in range(n_tasks)]

    # Collect all shared classes across all pairs
    all_shared: set = set()
    for pair in overlap_pairs:
        all_shared.update(pair.shared_classes)

    # --- Assign non-shared classes first ---
    for t_idx, cls_list in enumerate(task_class_lists):
        for cls in cls_list:
            if cls not in all_shared:
                result[t_idx][cls] = list(class_to_image_indices.get(cls, []))

    # --- Assign shared classes: group ALL tasks for each class, then split once ---
    # Build cls -> sorted list of all task indices that share it (across all pairs)
    shared_cls_to_tasks: Dict[int, List[int]] = {}
    for pair in overlap_pairs:
        for cls in pair.shared_classes:
            for t in pair.tasks:
                if t < n_tasks and t not in shared_cls_to_tasks.get(cls, []):
                    shared_cls_to_tasks.setdefault(cls, []).append(t)

    for cls, involved in shared_cls_to_tasks.items():
        involved = sorted(set(involved))
        pool = list(class_to_image_indices.get(cls, []))
        if not pool:
            for t in involved:
                result[t][cls] = []
            continue
        k = len(involved)
        _apply_split(result, cls, pool, involved, k, image_split, rng)

    return result


def _apply_split(
    result: List[Dict[int, List[int]]],
    cls: int,
    pool: List[int],
    involved_tasks: List[int],
    k: int,
    image_split: ImageSplit,
    rng: np.random.Generator,
) -> None:
    """Write image assignments for *cls* into *result* for all *involved_tasks*."""
    shuffled = list(rng.permutation(pool))
    n = len(shuffled)

    if image_split.strategy == "duplicate":
        for t in involved_tasks:
            result[t][cls] = list(shuffled)

    elif image_split.strategy == "disjoint":
        if k == 2:
            t_a, t_b = involved_tasks[0], involved_tasks[1]
            cut = int(image_split.ratio * n)
            result[t_a][cls] = shuffled[:cut]
            result[t_b][cls] = shuffled[cut:]
        else:
            # k-way equal split
            chunk = n // k
            for i, t in enumerate(involved_tasks):
                start = i * chunk
                end = start + chunk if i < k - 1 else n
                result[t][cls] = shuffled[start:end]

    elif image_split.strategy == "partial_duplicate":
        if k == 2:
            t_a, t_b = involved_tasks[0], involved_tasks[1]
            n_shared_core = int(image_split.overlap_pct * n)
            n_unique_each = (n - n_shared_core) // 2

            unique_a = shuffled[:n_unique_each]
            unique_b = shuffled[n_unique_each: n_unique_each * 2]
            shared_core = shuffled[n_unique_each * 2: n_unique_each * 2 + n_shared_core]

            result[t_a][cls] = unique_a + shared_core
            result[t_b][cls] = unique_b + shared_core
        else:
            # k-way: each task gets a unique slice + shared core
            n_shared_core = int(image_split.overlap_pct * n)
            remaining = n - n_shared_core
            unique_per = remaining // k
            shared_core = shuffled[unique_per * k: unique_per * k + n_shared_core]

            for i, t in enumerate(involved_tasks):
                start = i * unique_per
                end = start + unique_per
                result[t][cls] = shuffled[start:end] + shared_core

    else:
        raise ValueError(f"Unknown image split strategy: {image_split.strategy!r}")
