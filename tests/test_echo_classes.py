"""Tests for the echo-class (clone) feature and the fixed-size scenarios.

Echo classes re-present an earlier category under a fresh label id, with images
that are either the SAME as the source's first appearance or NEW (disjoint).
These tests exercise the layout, the image assignment, label remapping, and the
PILOT-compatibility invariants the harness relies on. They run against the
synthetic CIFAR-100 fixture, so no data download is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from clover.core.data_manager import OverlapDataManager
from clover.scenarios import (
    cumulative_drift,
    exact_replay,
    long_range_revisit,
    mid_range_revisit,
    partial_overlap,
)

INIT_CLS = 10
INCREMENT = 10
TOTAL = 100


def _dm(spec):
    return OverlapDataManager(
        "cifar100", init_cls=INIT_CLS, increment=INCREMENT, overlap_spec=spec
    )


# ---------------------------------------------------------------------------
# Layout invariants (all scenarios)
# ---------------------------------------------------------------------------

ALL_SPECS = {
    "exact_replay": lambda: exact_replay.build_spec(TOTAL, INIT_CLS, INCREMENT),
    "long_range": lambda: long_range_revisit.build_spec(TOTAL, INIT_CLS, INCREMENT),
    "mid_range": lambda: mid_range_revisit.build_spec(TOTAL, INIT_CLS, INCREMENT),
    "partial": lambda: partial_overlap.build_spec(TOTAL, INIT_CLS, INCREMENT),
    "cumulative": lambda: cumulative_drift.build_spec(TOTAL, INIT_CLS, INCREMENT),
}


@pytest.mark.parametrize("name", list(ALL_SPECS))
def test_every_task_is_fixed_size(patched_cifar100, name):
    dm = _dm(ALL_SPECS[name]())
    sizes = {len(dm.get_task_classes(t)) for t in range(dm.nb_tasks)}
    assert sizes == {INIT_CLS}, f"{name}: task sizes vary: {sizes}"


@pytest.mark.parametrize("name", list(ALL_SPECS))
def test_first_appearance_ids_are_contiguous(patched_cifar100, name):
    """PILOT sizes its head by cumulative new classes; ids must be gap-free."""
    dm = _dm(ALL_SPECS[name]())
    first = {}
    for t in range(dm.nb_tasks):
        for c in dm.get_task_classes(t):
            first.setdefault(c, t)
    order = sorted(first, key=lambda c: (first[c], c))
    assert order == list(range(len(order))), f"{name}: ids not contiguous"


@pytest.mark.parametrize("name", list(ALL_SPECS))
def test_no_empty_new_class_tasks(patched_cifar100, name):
    """Each task must introduce at least one new id (PILOT can't train empty)."""
    dm = _dm(ALL_SPECS[name]())
    seen = set()
    for t in range(dm.nb_tasks):
        new = [c for c in dm.get_task_classes(t) if c not in seen]
        assert new, f"{name}: task {t} introduces no new classes"
        seen.update(new)


def test_head_sizes(patched_cifar100):
    assert _dm(ALL_SPECS["exact_replay"]()).nb_classes == 110
    assert _dm(ALL_SPECS["long_range"]()).nb_classes == 110
    assert _dm(ALL_SPECS["mid_range"]()).nb_classes == 110
    assert _dm(ALL_SPECS["partial"]()).nb_classes == 100
    assert _dm(ALL_SPECS["cumulative"]()).nb_classes == 73


# ---------------------------------------------------------------------------
# Echo image relation: SAME vs NEW
# ---------------------------------------------------------------------------

def test_exact_replay_echo_uses_same_images(patched_cifar100):
    dm = _dm(ALL_SPECS["exact_replay"]())
    echo_task = dm.nb_tasks - 1
    # echo id 100 clones source id 0
    src_imgs = set(dm._train_assignment[0][0])
    echo_imgs = set(dm._train_assignment[echo_task][100])
    assert echo_imgs == src_imgs and len(echo_imgs) > 0


def test_long_range_echo_uses_disjoint_images(patched_cifar100):
    dm = _dm(ALL_SPECS["long_range"]())
    echo_task = dm.nb_tasks - 1
    src_imgs = set(dm._train_assignment[0][0])
    echo_imgs = set(dm._train_assignment[echo_task][100])
    assert src_imgs and echo_imgs
    assert src_imgs.isdisjoint(echo_imgs), "new-image echo must not reuse source images"


def test_mid_range_echo_source_is_mid_stream(patched_cifar100):
    spec = ALL_SPECS["mid_range"]()
    echoes = {e.new_id: e.source_id for e in spec.echoes}
    # anchor_task=4 -> source ids are task 4's classes (40..49)
    assert sorted(echoes.values()) == list(range(40, 50))


# ---------------------------------------------------------------------------
# Label remapping: echo rows carry the ECHO id, not the source id
# ---------------------------------------------------------------------------

def test_echo_task_targets_are_echo_ids(patched_cifar100):
    dm = _dm(ALL_SPECS["long_range"]())
    echo_task = dm.nb_tasks - 1
    _, targets, _ = dm.get_dataset(echo_task, source="train", mode="test", ret_data=True)
    uniq = set(int(t) for t in targets)
    assert uniq == set(range(100, 110)), f"echo targets wrong: {sorted(uniq)}"


def test_full_range_test_loader_includes_echo_classes(patched_cifar100):
    dm = _dm(ALL_SPECS["long_range"]())
    _, targets, _ = dm.get_dataset(
        list(range(dm.nb_classes)), source="test", mode="test", ret_data=True
    )
    uniq = set(int(t) for t in targets)
    for echo_id in range(100, 110):
        assert echo_id in uniq, f"echo class {echo_id} missing from test set"


def test_getlen_echo_class(patched_cifar100):
    dm = _dm(ALL_SPECS["long_range"]())
    echo_task = dm.nb_tasks - 1
    assert dm.getlen(100) == len(dm._train_assignment[echo_task][100]) > 0


# ---------------------------------------------------------------------------
# partial_overlap mixed task
# ---------------------------------------------------------------------------

def test_partial_mixed_task_has_fresh_and_returning(patched_cifar100):
    spec = ALL_SPECS["partial"]()
    dm = _dm(spec)
    echo_ids = {e.new_id for e in spec.echoes}
    mixed = dm.get_task_classes(dm.nb_tasks - 1)
    returning = [c for c in mixed if c in echo_ids]
    fresh = [c for c in mixed if c not in echo_ids]
    assert len(returning) == 5 and len(fresh) == 5


# ---------------------------------------------------------------------------
# cumulative_drift anchors
# ---------------------------------------------------------------------------

def test_cumulative_anchors_in_every_task(patched_cifar100):
    dm = _dm(ALL_SPECS["cumulative"]())
    for t in range(dm.nb_tasks):
        cls = set(dm.get_task_classes(t))
        assert {0, 1, 2}.issubset(cls), f"anchors missing from task {t}"


def test_cumulative_anchor_images_disjoint_across_tasks(patched_cifar100):
    dm = _dm(ALL_SPECS["cumulative"]())
    seen: set = set()
    for t in range(dm.nb_tasks):
        imgs = set(dm._train_assignment[t].get(0, []))
        assert imgs.isdisjoint(seen), f"anchor images reused at task {t}"
        seen.update(imgs)


# ---------------------------------------------------------------------------
# Echo map accessor
# ---------------------------------------------------------------------------

def test_get_echo_map(patched_cifar100):
    dm = _dm(ALL_SPECS["exact_replay"]())
    em = dm.get_echo_map()
    assert em[100] == (0, "same")
    assert len(em) == 10
    # baseline has no echoes
    base = OverlapDataManager("cifar100", init_cls=10, increment=10)
    assert base.get_echo_map() == {}
