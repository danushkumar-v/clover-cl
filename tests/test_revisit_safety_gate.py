"""The revisit-safety gate (SPEC §5.3): loss/params finite, accuracy above chance.

Runs a tiny stub training loop (backbone + linear head, no CLMethod/Trainer
yet -- those are P3) across three synthetic stream shapes: disjoint, a
same-id revisit mid-stream, and an echo revisit at the end. A "fixed" stub
uses the real loss policies (new_class_ce/seen_class_ce); a "broken" stub
reproduces the v1 bug verbatim (`logits[:, :known_classes] = -inf` applied
to the whole batch, no sample filtering) to prove the gate has teeth.

Neither stub is registered in `clover.methods` -- that registry stays empty
until P3's SimpleCIL (see test_package_scaffold.py). P3 will parametrize
this same gate machinery over `list_methods()` once real CLMethods exist.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from clover.backbones.tiny_mlp import TinyMLP
from clover.core.experience import Experience
from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.core.stream import Benchmark, build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods.losses import new_class_ce, seen_class_ce

NUM_CLASSES = 20
INIT_CLS = 4
INCREMENT = 4
NUM_EPOCHS = 40
LR = 0.05


def _disjoint_spec() -> StreamSpec:
    return StreamSpec(dataset="synthetic", init_cls=INIT_CLS, increment=INCREMENT)


def _same_id_mid_stream_spec() -> StreamSpec:
    return StreamSpec(
        dataset="synthetic",
        init_cls=INIT_CLS,
        increment=INCREMENT,
        revisits=[
            RevisitSpec(classes=[0, 1], placement="spaced", label="same", images="new", min_gap=1)
        ],
    )


def _echo_at_end_spec() -> StreamSpec:
    return StreamSpec(
        dataset="synthetic",
        init_cls=INIT_CLS,
        increment=INCREMENT,
        revisits=[
            RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="new")
        ],
    )


def _fixed_step(backbone, head, images, targets, view, optimizer) -> torch.Tensor:
    optimizer.zero_grad()
    logits = head(backbone(images))
    loss = new_class_ce(logits, targets, view) + seen_class_ce(logits, targets, view)
    loss.backward()
    optimizer.step()
    return loss


def _broken_step(backbone, head, images, targets, view, optimizer) -> torch.Tensor:
    """Reproduces the v1 bug verbatim: blind range-arithmetic column masking
    applied to the whole batch, no per-sample filtering."""
    optimizer.zero_grad()
    logits = head(backbone(images))
    known_classes = view.head_size - len(view.new_classes)
    masked = logits.clone()
    masked[:, :known_classes] = float("-inf")
    loss = F.cross_entropy(masked, targets)
    loss.backward()
    optimizer.step()
    return loss


def _batch_for(exp: Experience, dataset: SyntheticDataset) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [(i, c) for c in exp.classes_in_this_experience for i in exp.image_indices[c]]
    images = torch.stack([dataset[i][0] for i, _ in pairs])
    targets = torch.tensor([c for _, c in pairs])
    return images, targets


def _accuracy_on(backbone, head, exp: Experience, dataset: SyntheticDataset, classes) -> float | None:
    pairs = [(i, c) for c in classes for i in exp.image_indices.get(c, [])]
    if not pairs:
        return None
    images = torch.stack([dataset[i][0] for i, _ in pairs])
    targets = torch.tensor([c for _, c in pairs])
    with torch.no_grad():
        logits = head(backbone(images))
        logits = logits.masked_fill(~exp.label_space.logit_mask(logits.shape[-1]), float("-inf"))
        preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def run_gate(train_step, spec: StreamSpec) -> tuple[Benchmark, SyntheticDataset, torch.nn.Module, torch.nn.Module]:
    info = DatasetInfo("synthetic", NUM_CLASSES)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(
        spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices()
    )

    backbone = TinyMLP(input_size=train_ds.input_size, feature_dim=16)
    head = torch.nn.Linear(16, benchmark.total_classes)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=LR
    )

    for exp in benchmark.train_stream:
        images, targets = _batch_for(exp, train_ds)
        loss = torch.tensor(float("nan"))
        for _ in range(NUM_EPOCHS):
            loss = train_step(backbone, head, images, targets, exp.label_space, optimizer)

        assert math.isfinite(loss.item()), f"non-finite loss at experience {exp.task_label}"
        for p in list(backbone.parameters()) + list(head.parameters()):
            assert torch.isfinite(p).all(), f"non-finite parameter at experience {exp.task_label}"

    return benchmark, test_ds, backbone, head


# ---------------------------------------------------------------------------
# Fixed stub: green across all three shapes, with above-chance revisit accuracy
# ---------------------------------------------------------------------------


def test_fixed_stub_disjoint_learns_above_chance():
    benchmark, test_ds, backbone, head = run_gate(_fixed_step, _disjoint_spec())
    last_exp = benchmark.test_stream[-1]
    acc = _accuracy_on(backbone, head, last_exp, test_ds, last_exp.classes_in_this_experience)
    assert acc is not None and acc > 1 / NUM_CLASSES


def test_fixed_stub_same_id_revisit_is_finite_and_above_chance():
    benchmark, test_ds, backbone, head = run_gate(_fixed_step, _same_id_mid_stream_spec())
    revisit_exp = next(e for e in benchmark.train_stream if e.is_revisit_experience())
    test_exp = benchmark.test_stream[revisit_exp.task_label]
    acc = _accuracy_on(backbone, head, test_exp, test_ds, test_exp.revisiting_classes)
    assert acc is not None and acc > 1 / NUM_CLASSES


def test_fixed_stub_echo_revisit_is_finite_and_above_chance():
    benchmark, test_ds, backbone, head = run_gate(_fixed_step, _echo_at_end_spec())
    last_exp = benchmark.train_stream[-1]
    assert last_exp.echo_map  # confirms this shape actually exercises echoes
    test_exp = benchmark.test_stream[-1]
    acc = _accuracy_on(backbone, head, test_exp, test_ds, test_exp.echo_map.keys())
    assert acc is not None and acc > 1 / NUM_CLASSES


# ---------------------------------------------------------------------------
# Broken stub: proves the gate has teeth. The v1 bug only manifests when an
# *old*-labelled sample shares a batch with a blind range-arithmetic column
# mask -- true for the same-id shape, but not disjoint (no old samples ever)
# or echo (echo ids are always fresh, never numerically "old").
# ---------------------------------------------------------------------------


def test_broken_stub_disjoint_stays_finite():
    run_gate(_broken_step, _disjoint_spec())  # no assertion error == finite throughout


def test_broken_stub_echo_revisit_stays_finite():
    run_gate(_broken_step, _echo_at_end_spec())  # echoes never collide with the bug


def test_broken_stub_same_id_revisit_produces_non_finite_loss():
    with pytest.raises(AssertionError, match="non-finite loss"):
        run_gate(_broken_step, _same_id_mid_stream_spec())
