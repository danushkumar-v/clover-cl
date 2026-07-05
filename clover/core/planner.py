"""spec -> plan: placement, echo-id allocation, image split (SPEC §3.3).

``resolve(spec, dataset_info)`` is the one place a v2 stream gets turned from
a statistical description into concrete class lists. The core trick, carried
over from v1's fixed-size scenarios and generalized here: echo ids and
same-id revisits are accounted for out of each experience's class *budget*
(no eviction), and label ids are handed out from one monotonically
increasing cursor shared between "draw the next real class" and "allocate
the next echo id" — so id space stays perfectly contiguous even when a
task_size="fixed" budget shortfall means some tail real classes are never
drawn at all (they simply aren't needed by this particular stream).
"""

from __future__ import annotations

import numpy as np

from clover.core.plan import EchoEntry, StreamPlan
from clover.core.spec import DatasetInfo, ImageRelation, RevisitSpec, StreamSpec


def _pilot_class_order(n_classes: int, seed: int) -> list[int]:
    """Reproduce PILOT's class-order shuffle exactly (byte-compat).

    The only sanctioned use of implicit global RNG state in this codebase:
    the seed-1993 golden fixture was captured with literal
    ``np.random.seed`` + ``np.random.permutation``, so this function must
    keep using it verbatim. Every other stochastic choice in the planner
    goes through an explicit ``np.random.default_rng(stream_seed)``.
    """
    np.random.seed(seed)
    return np.random.permutation(n_classes).tolist()


def compute_increments(num_classes: int, init_cls: int, increment: int) -> list[int]:
    """Return the per-experience class-count schedule (PILOT-style slicing)."""
    increments = [init_cls]
    while sum(increments) + increment < num_classes:
        increments.append(increment)
    offset = num_classes - sum(increments)
    if offset > 0:
        increments.append(offset)
    return increments


def task_class_range(num_classes: int, init_cls: int, increment: int, task_index: int) -> list[int]:
    """Return the nominal (pre-revisit) backbone class-position range for one task.

    Used by scenario factories (e.g. ``mid_range_revisit``) that need to name
    an existing task's classes without duplicating the slicing math.
    """
    increments = compute_increments(num_classes, init_cls, increment)
    if task_index >= len(increments):
        raise ValueError(
            f"task_index={task_index} out of range: dataset yields only "
            f"{len(increments)} experiences."
        )
    start = sum(increments[:task_index])
    return list(range(start, start + increments[task_index]))


def _find_first_experience(cls: int, task_starts: list[int], increments: list[int]) -> int:
    for t, (start, inc) in enumerate(zip(task_starts, increments)):
        if start <= cls < start + inc:
            return t
    return len(increments) - 1


def _resolve_revisit_classes(
    rv: RevisitSpec, backbone: list[list[int]], num_classes: int, rng: np.random.Generator
) -> list[int]:
    if isinstance(rv.classes, str):
        return list(backbone[0])  # "task0" (the only string form; validated earlier)
    if isinstance(rv.classes, dict):
        n_pick = rv.classes["random"]
        if n_pick > num_classes:
            raise ValueError(f"classes.random={n_pick} exceeds dataset size {num_classes}.")
        chosen = rng.choice(num_classes, size=n_pick, replace=False)
        return sorted(int(c) for c in chosen)
    explicit_classes: list[int] = rv.classes
    bad = [c for c in explicit_classes if not (0 <= c < num_classes)]
    if bad:
        raise ValueError(
            f"Revisit class ids {bad} are out of range for a {num_classes}-class "
            f"dataset (0..{num_classes - 1})."
        )
    return list(explicit_classes)


def _place_random(
    available: list[int], times: int, min_gap: int, rng: np.random.Generator
) -> list[int]:
    chosen: list[int] = []
    forbidden: set[int] = set()
    pool = list(available)
    for _ in range(times):
        valid = [t for t in pool if t not in forbidden]
        if not valid:
            raise ValueError("Cannot satisfy min_gap constraint with random placement.")
        pick = valid[int(rng.integers(0, len(valid)))]
        chosen.append(pick)
        for gap in range(min_gap):
            forbidden.add(pick - gap)
            forbidden.add(pick + gap)
        pool = [t for t in pool if t not in forbidden and t != pick]
    return sorted(chosen)


def _place_spaced(available: list[int], times: int) -> list[int]:
    if times == 1:
        return [available[len(available) // 2]]
    step = max(1, (len(available) - 1) // (times - 1))
    indices = [min(i * step, len(available) - 1) for i in range(times)]
    return sorted(set(available[i] for i in indices))


def _place_end(available: list[int], times: int) -> list[int]:
    return sorted(available[-times:])


def _place_clustered(available: list[int], times: int, rng: np.random.Generator) -> list[int]:
    if len(available) < times:
        return list(available)
    start = int(rng.integers(0, len(available) - times + 1))
    return available[start : start + times]


def _place(
    available: list[int],
    times: int,
    placement: str,
    min_gap: int,
    rng: np.random.Generator,
    cls: int,
    first_exp: int,
    n_exp: int,
) -> list[int]:
    if placement == "every_task":
        return sorted(available)
    if len(available) < times:
        raise ValueError(
            f"Cannot place {times} revisit(s) of class {cls} (first appearing at "
            f"experience {first_exp}) with min_gap={min_gap} in a {n_exp}-experience "
            f"stream. Maximum achievable: {len(available)}."
        )
    if placement == "random":
        return _place_random(available, times, min_gap, rng)
    if placement == "spaced":
        return _place_spaced(available, times)
    if placement == "end_of_stream":
        return _place_end(available, times)
    if placement == "clustered":
        return _place_clustered(available, times, rng)
    raise AssertionError(f"unreachable: placement {placement!r} passed RevisitSpec.validate()")


def _validate_contiguous_first_appearance(task_class_lists: list[list[int]]) -> None:
    seen_first: dict[int, int] = {}
    for t, cls_list in enumerate(task_class_lists):
        for c in cls_list:
            seen_first.setdefault(c, t)
    first_order = sorted(seen_first, key=lambda c: (seen_first[c], c))
    for expected, c in enumerate(first_order):
        if c != expected:
            raise ValueError(
                "task_class_lists must introduce class ids in contiguous "
                f"first-appearance order 0..N-1; expected id {expected} but got {c}. "
                "(Gaps break CIL head sizing.)"
            )


def resolve(spec: StreamSpec, dataset_info: DatasetInfo) -> StreamPlan:
    """Resolve a StreamSpec into a concrete, serializable StreamPlan.

    Deterministic given ``(spec, dataset_info)``: ``shuffle_seed`` drives the
    PILOT-compatible class order, ``stream_seed`` drives every other
    stochastic choice.
    """
    spec.validate()
    num_classes = dataset_info.num_classes
    if spec.init_cls > num_classes:
        raise ValueError(
            f"init_cls={spec.init_cls} exceeds dataset {dataset_info.name!r}'s "
            f"{num_classes} classes."
        )

    class_order = _pilot_class_order(num_classes, spec.shuffle_seed)
    rng = np.random.default_rng(spec.stream_seed)

    increments = compute_increments(num_classes, spec.init_cls, spec.increment)
    n_exp = len(increments)
    task_starts = [sum(increments[:t]) for t in range(n_exp)]
    backbone = [
        list(range(start, start + inc)) for start, inc in zip(task_starts, increments)
    ]

    # occurrences[t] = [(source_position, label, images), ...] landing on task t
    occurrences: list[list[tuple[int, str, str]]] = [[] for _ in range(n_exp)]

    for rv in spec.revisits:
        classes = _resolve_revisit_classes(rv, backbone, num_classes, rng)
        for cls in classes:
            first_exp = _find_first_experience(cls, task_starts, increments)
            available = [
                t for t in range(n_exp) if t != first_exp and t > first_exp + rv.min_gap - 1
            ]
            targets = _place(available, rv.times, rv.placement, rv.min_gap, rng, cls, first_exp, n_exp)
            for t in targets:
                occurrences[t].append((cls, rv.label, rv.images))

    task_class_lists: list[list[int]] = []
    echo_table: dict[int, EchoEntry] = {}
    revisit_ids: set[int] = set()
    next_id = 0

    for t in range(n_exp):
        budget = increments[t]
        occ = occurrences[t]
        consumed = len(occ)
        if spec.task_size == "fixed":
            if consumed > budget:
                raise ValueError(
                    f"Experience {t}: {consumed} revisit occurrence(s) exceed its "
                    f"class budget of {budget} under task_size='fixed'. Reduce "
                    f"revisits landing here, increase min_gap, or use task_size='grow'."
                )
            n_new = budget - consumed
        else:  # grow
            n_new = budget

        task_ids = list(range(next_id, next_id + n_new))
        next_id += n_new

        for source_id, label, images in occ:
            relation = ImageRelation.parse(images)
            if label == "new":
                echo_id = next_id
                next_id += 1
                echo_table[echo_id] = EchoEntry(source_id, relation.kind, relation.overlap_pct)
                task_ids.append(echo_id)
            else:  # same
                task_ids.append(source_id)
                revisit_ids.add(source_id)

        task_class_lists.append(sorted(task_ids))

    _validate_contiguous_first_appearance(task_class_lists)

    head_size_schedule: list[int] = []
    seen: set[int] = set()
    for cls_list in task_class_lists:
        seen.update(cls_list)
        head_size_schedule.append(len(seen))

    return StreamPlan(
        spec=spec,
        num_classes=num_classes,
        class_order=class_order,
        task_class_lists=task_class_lists,
        echo_table=echo_table,
        revisit_ids=frozenset(revisit_ids),
        head_size_schedule=head_size_schedule,
    )
