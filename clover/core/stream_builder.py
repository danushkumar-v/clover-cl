"""Translates a StreamSpec into a fully realised Benchmark."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np

from clover.core.data_manager import OverlapDataManager
from clover.core.overlap_spec import ImageSplit, OverlapPair, OverlapSpec
from clover.core.stream_spec import RevisitSpec, StreamSpec
from clover.core.task_builder import compute_increments
from clover.utils.seeding import get_rng


def build_benchmark(spec: StreamSpec) -> "Benchmark":  # noqa: F821 (forward ref)
    """Convert a :class:`~clover.core.stream_spec.StreamSpec` into a
    :class:`~clover.core.benchmark.Benchmark`.

    Steps:

    1. Validate the spec.
    2. Compute the baseline task structure from ``init_cls`` / ``increment``.
    3. Resolve any ``classes: N`` (random count) into concrete class IDs.
    4. Place revisits across the experience sequence using the chosen strategy.
    5. Translate placements into an :class:`~clover.core.overlap_spec.OverlapSpec`.
    6. Instantiate :class:`~clover.core.data_manager.OverlapDataManager`.
    7. Build :class:`~clover.core.experience.Experience` objects with full metadata.
    8. Return a :class:`~clover.core.benchmark.Benchmark`.

    Args:
        spec: A :class:`~clover.core.stream_spec.StreamSpec`.

    Returns:
        A ready-to-use :class:`~clover.core.benchmark.Benchmark`.

    Raises:
        ValueError: If the spec is unsatisfiable (too many revisits, impossible
            min_gap, unknown class ID, etc.).
    """
    # Late import to avoid circular dependency
    from clover.core.benchmark import Benchmark
    from clover.core.stream import Stream

    spec.validate()

    rng = get_rng(spec.stream_seed)

    # Baseline task structure
    from clover.datasets.cifar100 import CIFAR100Dataset  # noqa: F401 (trigger registry)

    # Determine dataset's total classes without loading all data
    _n_classes_map = {
        "cifar100": 100,
        "cub200": 200,
        "cub": 200,
        "imagenet_r": 200,
        "imagenetr": 200,
        "imagenet_a": 200,
        "imageneta": 200,
        "omnibenchmark": 300,
        "omnibench": 300,
        "vtab": 50,
    }
    total_classes = _n_classes_map.get(spec.dataset.lower())
    if total_classes is None:
        raise ValueError(
            f"Unknown dataset {spec.dataset!r}. Known: {sorted(_n_classes_map)}"
        )

    increments = compute_increments(total_classes, spec.init_cls, spec.increment)
    n_exp = len(increments)

    # Baseline task class ranges (in remapped space, pre-shuffle — used for
    # validating revisit class IDs are in the dataset)
    task_starts = [sum(increments[:t]) for t in range(n_exp)]

    # Resolve "pick N" classes and validate explicit class IDs
    resolved_revisits: List[Tuple[List[int], RevisitSpec]] = []
    for rv in spec.revisits:
        if isinstance(rv.classes, int):
            n_pick = rv.classes
            if n_pick < 1 or n_pick > total_classes:
                raise ValueError(
                    f"classes={n_pick} is out of range [1, {total_classes}]."
                )
            # Only pick from classes that have enough feasible revisit positions
            feasible_pool = [
                cls for cls in range(total_classes)
                if len([
                    t for t in range(n_exp)
                    if t != _find_first_experience(cls, task_starts, increments)
                    and t > _find_first_experience(cls, task_starts, increments) + rv.min_gap - 1
                ]) >= rv.times
            ]
            if len(feasible_pool) < n_pick:
                raise ValueError(
                    f"Only {len(feasible_pool)} classes can support {rv.times} revisit(s) "
                    f"with min_gap={rv.min_gap} in a {n_exp}-experience stream, "
                    f"but {n_pick} were requested."
                )
            chosen = sorted(
                rng.choice(feasible_pool, size=n_pick, replace=False).tolist()
            )
        else:
            chosen = list(rv.classes)
            bad = [c for c in chosen if not (0 <= c < total_classes)]
            if bad:
                raise ValueError(
                    f"Revisit class IDs {bad} are out of range for dataset "
                    f"{spec.dataset!r} (0..{total_classes - 1})."
                )
        resolved_revisits.append((chosen, rv))

    # For each resolved revisit, place the class into target experiences
    # placements[exp_idx] = set of class IDs to inject into that experience
    placements: Dict[int, Set[int]] = {t: set() for t in range(n_exp)}

    for classes, rv in resolved_revisits:
        for cls in classes:
            # Find the "first" experience that naturally contains this class
            first_exp = _find_first_experience(cls, task_starts, increments)

            # How many valid positions exist for this class?
            n_feasible = len(
                [t for t in range(n_exp) if t != first_exp and t > first_exp + rv.min_gap - 1]
            )
            if n_feasible < rv.times:
                raise ValueError(
                    f"Cannot place {rv.times} revisit(s) of class {cls} "
                    f"(first at experience {first_exp}) with min_gap={rv.min_gap} "
                    f"in a {n_exp}-experience stream. "
                    f"Maximum achievable: {n_feasible}."
                )

            target_exps = _place_revisits(
                first_exp=first_exp,
                n_exp=n_exp,
                times=rv.times,
                placement=rv.placement,
                min_gap=rv.min_gap,
                rng=rng,
            )
            for t in target_exps:
                placements[t].add(cls)

    # Build OverlapSpec from placements
    pairs: List[OverlapPair] = []
    for target_exp, cls_set in placements.items():
        if not cls_set:
            continue
        for cls in cls_set:
            first_exp = _find_first_experience(cls, task_starts, increments)
            if first_exp == target_exp:
                continue
            pairs.append(
                OverlapPair(
                    tasks=(first_exp, target_exp),
                    shared_classes=[cls],
                )
            )

    # Merge pairs that share the same (task_a, task_b) to keep OverlapSpec tidy
    pairs = _merge_pairs(pairs)

    if pairs:
        image_split = ImageSplit(
            strategy=spec.image_strategy,
            ratio=spec.image_split_ratio,
            overlap_pct=spec.image_overlap_pct,
        )
        overlap_spec = OverlapSpec(
            mode="partial",
            pairs=pairs,
            image_split=image_split,
            seed=spec.stream_seed,
        )
        overlap_spec.validate()
    else:
        overlap_spec = None

    # Instantiate the engine
    manager = OverlapDataManager(
        dataset_name=spec.dataset,
        init_cls=spec.init_cls,
        increment=spec.increment,
        overlap_spec=overlap_spec,
        shuffle_seed=spec.shuffle_seed,
        data_root=spec.data_root,
        preserve_task_size=spec.preserve_task_size,
    )

    benchmark_name = (
        f"{spec.dataset}_stream_seed{spec.stream_seed}_shuffle{spec.shuffle_seed}"
    )

    # Build Experience objects
    train_experiences = _build_experiences(manager, "train", benchmark_name)
    test_experiences = _build_experiences(manager, "test", benchmark_name)

    train_stream = Stream(train_experiences, split="train")
    test_stream = Stream(test_experiences, split="test")

    return Benchmark(
        train_stream=train_stream,
        test_stream=test_stream,
        stream_spec=spec,
        underlying_manager=manager,
    )


# ---------------------------------------------------------------------------
# Placement algorithms
# ---------------------------------------------------------------------------

def _find_first_experience(
    cls: int, task_starts: List[int], increments: List[int]
) -> int:
    """Return the index of the experience that naturally contains class *cls*."""
    for t, (start, inc) in enumerate(zip(task_starts, increments)):
        if start <= cls < start + inc:
            return t
    # cls >= total_classes — should have been caught upstream
    return len(increments) - 1


def _place_revisits(
    first_exp: int,
    n_exp: int,
    times: int,
    placement: str,
    min_gap: int,
    rng: np.random.Generator,
) -> List[int]:
    """Return a list of experience indices where the class should be injected.

    Only the *revisit* experiences are returned (not the original first_exp).
    """
    available = [t for t in range(n_exp) if t != first_exp and t > first_exp + min_gap - 1]

    if len(available) < times:
        raise ValueError(
            f"Cannot place {times} revisit(s) with min_gap={min_gap} for a class "
            f"first appearing at experience {first_exp} in a {n_exp}-experience "
            f"stream. Maximum achievable: {len(available)}."
        )

    if placement == "random":
        return _place_random(available, times, min_gap, first_exp, rng)
    elif placement == "spaced":
        return _place_spaced(available, times)
    elif placement == "end_of_stream":
        return _place_end(available, times)
    elif placement == "clustered":
        return _place_clustered(available, times, rng)
    else:
        raise ValueError(f"Unknown placement {placement!r}.")


def _place_random(
    available: List[int], times: int, min_gap: int, first_exp: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample ``times`` positions respecting min_gap between consecutive revisits."""
    # Greedy: sample one at a time, enforcing gap from previously chosen
    chosen = []
    forbidden: Set[int] = set()
    pool = list(available)
    for _ in range(times):
        valid = [t for t in pool if t not in forbidden]
        if not valid:
            raise ValueError(
                "Cannot satisfy min_gap constraint with random placement."
            )
        idx = int(rng.integers(0, len(valid)))
        pick = valid[idx]
        chosen.append(pick)
        for gap in range(min_gap):
            forbidden.add(pick - gap)
            forbidden.add(pick + gap)
        pool = [t for t in pool if t not in forbidden and t != pick]
    return sorted(chosen)


def _place_spaced(available: List[int], times: int) -> List[int]:
    """Distribute *times* revisit positions as evenly as possible."""
    if times == 1:
        return [available[len(available) // 2]]
    step = max(1, (len(available) - 1) // (times - 1))
    indices = [min(i * step, len(available) - 1) for i in range(times)]
    return sorted(set(available[i] for i in indices))


def _place_end(available: List[int], times: int) -> List[int]:
    """Place all *times* revisits in the last experiences."""
    return sorted(available[-times:])


def _place_clustered(
    available: List[int], times: int, rng: np.random.Generator
) -> List[int]:
    """Place revisits in a tight consecutive window."""
    if len(available) < times:
        return list(available)
    start = int(rng.integers(0, len(available) - times + 1))
    return available[start: start + times]


# ---------------------------------------------------------------------------
# OverlapPair merging
# ---------------------------------------------------------------------------

def _merge_pairs(pairs: List[OverlapPair]) -> List[OverlapPair]:
    """Merge pairs that share the same ``(task_a, task_b)`` tuple."""
    merged: Dict[Tuple[int, int], List[int]] = {}
    for p in pairs:
        key = p.tasks
        merged.setdefault(key, [])
        for cls in p.shared_classes:
            if cls not in merged[key]:
                merged[key].append(cls)
    return [
        OverlapPair(tasks=tasks, shared_classes=sorted(classes))
        for tasks, classes in merged.items()
    ]


# ---------------------------------------------------------------------------
# Experience builder
# ---------------------------------------------------------------------------

def _build_experiences(
    manager: OverlapDataManager,
    split: str,
    benchmark_name: str,
) -> List["Experience"]:  # noqa: F821
    from clover.core.experience import Experience

    n_exp = manager.nb_tasks
    total_classes = manager.nb_classes

    # Pre-compute which classes appear in which experiences
    exp_classes: List[Set[int]] = [
        set(manager.get_task_classes(t)) for t in range(n_exp)
    ]

    # All classes across the whole stream
    all_stream_classes: Set[int] = set()
    for s in exp_classes:
        all_stream_classes.update(s)

    experiences = []
    seen_so_far: Set[int] = set()

    for t in range(n_exp):
        here = sorted(exp_classes[t])
        revisiting = sorted(c for c in here if c in seen_so_far)
        first_app = sorted(c for c in here if c not in seen_so_far)

        future = sorted(
            c for c in all_stream_classes
            if c not in seen_so_far | set(here)
            or any(c in exp_classes[future_t] for future_t in range(t + 1, n_exp))
        )
        # Simpler accurate definition: classes appearing in any t' > t
        future = sorted(
            c for c in all_stream_classes
            if any(c in exp_classes[future_t] for future_t in range(t + 1, n_exp))
        )

        # overlap_with_previous: for each earlier experience sharing any class
        overlap_with_prev: Dict[int, int] = {}
        for prev_t in range(t):
            shared_count = len(set(here) & exp_classes[prev_t])
            if shared_count > 0:
                overlap_with_prev[prev_t] = shared_count

        # Retrieve dataset
        dataset = manager.get_dataset(t, source=split, mode=split)

        # Image indices from assignment
        assignment_map = (
            manager._train_assignment[t]
            if split == "train"
            else manager._test_assignment[t]
        )
        image_indices = {c: list(idxs) for c, idxs in assignment_map.items()}

        exp = Experience(
            task_label=t,
            benchmark_name=benchmark_name,
            dataset=dataset,
            classes_in_this_experience=here,
            classes_seen_so_far=sorted(seen_so_far | set(here)),
            classes_in_future=future,
            total_classes_in_stream=total_classes,
            revisiting_classes=revisiting,
            first_appearance_of=first_app,
            overlap_with_previous=overlap_with_prev,
            image_indices=image_indices,
            n_samples=len(dataset),
        )
        experiences.append(exp)
        seen_so_far.update(here)

    return experiences
