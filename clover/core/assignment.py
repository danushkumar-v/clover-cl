"""Pure image -> experience assignment, the v1 image_assigner successor (SPEC §4).

Unifies v1's two separate mechanisms (a global ``ImageSplit`` for same-id
overlap pairs, plus a bolted-on ``_assign_echo_images`` post-step for
echoes) into one pass: every appearance of a class beyond its first (echo or
same-id revisit) carries its own ``image_relation`` (``same``/``new``/
``partial:<pct>``), and all appearances that draw from one source class's
image pool are split together in a single call.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from clover.core.plan import StreamPlan

_Occurrence = Tuple[int, int, str, "float | None"]  # (task_idx, class_id, relation, overlap_pct)


def assign_images(
    class_to_image_indices: Dict[int, List[int]],
    plan: StreamPlan,
    rng: np.random.Generator,
) -> List[Dict[int, List[int]]]:
    """Assign image indices to experiences for every class in *plan*.

    Args:
        class_to_image_indices: Mapping from *original* dataset class id
            (``CLDataset.get_class_to_indices()``'s contract) to image
            indices. Training positions are translated via
            ``plan.class_order``.
        plan: A resolved :class:`~clover.core.plan.StreamPlan`.
        rng: Seeded generator; all randomness goes through it.

    Returns:
        One dict per experience: ``{class_id: [image_indices]}``.
    """
    occurrences_by_source: Dict[int, List[_Occurrence]] = {}

    for t, cls_list in enumerate(plan.task_class_lists):
        for cls in cls_list:
            if cls in plan.echo_table:
                entry = plan.echo_table[cls]
                occurrences_by_source.setdefault(entry.source_id, []).append(
                    (t, cls, entry.image_relation, entry.overlap_pct)
                )
            else:
                # Own natural appearance, or a same-id revisit reusing this
                # position's id: each appearance defaults to a disjoint
                # ("new") share of the source's images unless grouped only
                # with "same" siblings (see _split_pool).
                occurrences_by_source.setdefault(cls, []).append((t, cls, "new", None))

    n_exp = len(plan.task_class_lists)
    result: List[Dict[int, List[int]]] = [{} for _ in range(n_exp)]

    for source_position, occs in occurrences_by_source.items():
        pool = (
            list(class_to_image_indices.get(plan.class_order[source_position], []))
            if source_position < len(plan.class_order)
            else []
        )
        _split_pool(result, pool, occs, rng)

    return result


def _split_pool(
    result: List[Dict[int, List[int]]],
    pool: List[int],
    occs: List[_Occurrence],
    rng: np.random.Generator,
) -> None:
    if len(occs) == 1:
        t, cls, _relation, _pct = occs[0]
        result[t][cls] = list(pool)
        return

    shuffled = list(rng.permutation(pool)) if pool else []
    n = len(shuffled)

    same_occs = [o for o in occs if o[2] == "same"]
    split_occs = [o for o in occs if o[2] != "same"]

    for t, cls, _relation, _pct in same_occs:
        result[t][cls] = list(shuffled)

    if not split_occs:
        return

    partial_pcts = [o[3] for o in split_occs if o[2] == "partial" and o[3] is not None]
    n_shared = int(max(partial_pcts) * n) if partial_pcts else 0
    shared_core = shuffled[n - n_shared :] if n_shared else []
    remaining = shuffled[: n - n_shared] if n_shared else shuffled

    k = len(split_occs)
    chunk = len(remaining) // k if k else 0
    for i, (t, cls, _relation, _pct) in enumerate(split_occs):
        start = i * chunk
        end = start + chunk if i < k - 1 else len(remaining)
        # shared_core (if any partial:<pct> occurrence requested one) is a
        # relationship between all co-occurrences of this source, not a
        # unilateral grant to whichever occurrence asked for it.
        result[t][cls] = remaining[start:end] + shared_core
