"""Stream, Benchmark (SPEC §3-§4)."""

from __future__ import annotations

from typing import Dict, Iterator, List, Set

import numpy as np

from clover.core.assignment import assign_images
from clover.core.experience import Experience
from clover.core.plan import StreamPlan
from clover.core.planner import resolve
from clover.core.spec import DatasetInfo, StreamSpec


class Stream:
    """An ordered sequence of :class:`~clover.core.experience.Experience` objects.

    Args:
        experiences: Ordered list of Experience objects.
        split: ``"train"`` or ``"test"``.
    """

    def __init__(self, experiences: List[Experience], split: str) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
        self._experiences = list(experiences)
        self.split = split

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._experiences)

    def __len__(self) -> int:
        return len(self._experiences)

    def __getitem__(self, k: int) -> Experience:
        return self._experiences[k]

    def revisit_density(self) -> float:
        """Fraction of experiences that contain at least one revisiting class."""
        if not self._experiences:
            return 0.0
        return sum(1 for e in self._experiences if e.is_revisit_experience()) / len(
            self._experiences
        )

    def class_appearance_count(self) -> Dict[int, int]:
        """Return ``{class_id: number_of_experiences_it_appears_in}``."""
        counts: Dict[int, int] = {}
        for exp in self._experiences:
            for c in exp.classes_in_this_experience:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def average_overlap(self) -> float:
        """Mean number of shared classes between consecutive experiences."""
        if len(self._experiences) < 2:
            return 0.0
        total = 0
        for i in range(len(self._experiences) - 1):
            a = set(self._experiences[i].classes_in_this_experience)
            b = set(self._experiences[i + 1].classes_in_this_experience)
            total += len(a & b)
        return total / (len(self._experiences) - 1)


class Benchmark:
    """A resolved stream, ready to drive training/evaluation.

    Unlike v1, there is no ``underlying_manager`` — ``total_classes``,
    ``nb_experiences`` and ``overlap_matrix()`` are computed directly from
    the plan/experiences.
    """

    def __init__(
        self,
        train_stream: Stream,
        test_stream: Stream,
        spec: StreamSpec,
        plan: StreamPlan,
    ) -> None:
        self.train_stream = train_stream
        self.test_stream = test_stream
        self.spec = spec
        self.plan = plan

    @property
    def name(self) -> str:
        return f"{self.spec.dataset}_stream_seed{self.spec.stream_seed}_shuffle{self.spec.shuffle_seed}"

    @property
    def nb_experiences(self) -> int:
        return len(self.train_stream)

    @property
    def total_classes(self) -> int:
        return self.plan.head_size_schedule[-1] if self.plan.head_size_schedule else 0

    def overlap_matrix(self) -> np.ndarray:
        """Return an ``[n_exp, n_exp]`` matrix of shared-class counts."""
        n = self.nb_experiences
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            a = set(self.train_stream[i].classes_in_this_experience)
            for j in range(n):
                b = set(self.train_stream[j].classes_in_this_experience)
                matrix[i, j] = len(a & b)
        return matrix

    def save_manifest(self, path: str) -> None:
        self.plan.save(path)


def build_experiences(
    plan: StreamPlan,
    image_assignment: List[Dict[int, List[int]]],
    benchmark_name: str,
    dataset_name: str,
) -> List[Experience]:
    """Build the ordered list of :class:`Experience` objects for one split."""
    n_exp = len(plan.task_class_lists)
    exp_classes: List[Set[int]] = [set(cls_list) for cls_list in plan.task_class_lists]
    all_stream_classes: Set[int] = set().union(*exp_classes) if exp_classes else set()
    total_classes = len(all_stream_classes)

    experiences: List[Experience] = []
    seen_so_far: Set[int] = set()

    for t in range(n_exp):
        here = sorted(exp_classes[t])
        revisiting = sorted(c for c in here if c in seen_so_far)
        first_app = sorted(c for c in here if c not in seen_so_far)
        future = sorted(
            c for c in all_stream_classes if any(c in exp_classes[ft] for ft in range(t + 1, n_exp))
        )
        overlap_with_prev: Dict[int, int] = {}
        for prev_t in range(t):
            shared = len(set(here) & exp_classes[prev_t])
            if shared > 0:
                overlap_with_prev[prev_t] = shared

        echo_map = {c: plan.echo_table[c] for c in here if c in plan.echo_table}
        image_indices = dict(image_assignment[t])
        n_samples = sum(len(idxs) for idxs in image_indices.values())

        experiences.append(
            Experience(
                task_label=t,
                benchmark_name=benchmark_name,
                dataset=dataset_name,
                classes_in_this_experience=here,
                classes_seen_so_far=sorted(seen_so_far | set(here)),
                classes_in_future=future,
                total_classes_in_stream=total_classes,
                revisiting_classes=revisiting,
                first_appearance_of=first_app,
                overlap_with_previous=overlap_with_prev,
                echo_map=echo_map,
                image_indices=image_indices,
                n_samples=n_samples,
            )
        )
        seen_so_far.update(here)

    return experiences


def build_benchmark(
    spec: StreamSpec,
    dataset_info: DatasetInfo,
    train_class_to_indices: Dict[int, List[int]],
    test_class_to_indices: Dict[int, List[int]],
) -> Benchmark:
    """Resolve *spec* and assemble a ready-to-use :class:`Benchmark`.

    Args:
        spec: The stream spec to resolve.
        dataset_info: The dataset's name/class-count.
        train_class_to_indices: Original-class-id -> image indices (train split).
        test_class_to_indices: Original-class-id -> image indices (test split).
    """
    plan = resolve(spec, dataset_info)
    rng = np.random.default_rng(spec.stream_seed)

    train_assignment = assign_images(train_class_to_indices, plan, rng)
    test_assignment = assign_images(test_class_to_indices, plan, rng)

    benchmark_name = f"{spec.dataset}_stream_seed{spec.stream_seed}_shuffle{spec.shuffle_seed}"
    train_experiences = build_experiences(plan, train_assignment, benchmark_name, spec.dataset)
    test_experiences = build_experiences(plan, test_assignment, benchmark_name, spec.dataset)

    return Benchmark(
        train_stream=Stream(train_experiences, split="train"),
        test_stream=Stream(test_experiences, split="test"),
        spec=spec,
        plan=plan,
    )
