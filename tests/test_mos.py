"""MOS correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.mos import MOS


class _FakeLabelSpace:
    def __init__(self, head_size, new_classes):
        self.head_size = head_size
        self._new_classes = frozenset(new_classes)
        self._seen_classes = frozenset(range(head_size)) - self._new_classes

    def new_mask(self, targets):
        return torch.isin(targets, torch.tensor(sorted(self._new_classes)))

    def old_mask(self, targets):
        if not self._seen_classes:
            return torch.zeros_like(targets, dtype=torch.bool)
        return torch.isin(targets, torch.tensor(sorted(self._seen_classes)))

    @property
    def new_classes(self):
        return self._new_classes

    @property
    def seen_classes(self):
        return self._seen_classes


class _FakeExperience:
    def __init__(self, label_space, task_label):
        self.label_space = label_space
        self.task_label = task_label


def _make_method() -> MOS:
    method = get_method("mos")(crct_epochs=3, ca_lr=5e-3, samples_per_class=8)
    info = StreamInfo(dataset="synthetic", nb_experiences=3, total_classes=6, input_size=8)
    method.build(info, {"depth": 2, "bottleneck_dim": 4})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_mos():
    assert get_method("mos") is MOS


def test_first_experience_takes_no_snapshot():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    assert len(method.backbone.adapter_list) == 0

    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )
    # CA only runs for task_label > 0 -- nothing to align against yet.
    assert method._class_stats  # prototypes/stats still recorded though


def test_second_experience_snapshots_and_ca_preserves_finite_head():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    assert len(method.backbone.adapter_list) == 1

    method.train_experience(
        exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx
    )
    assert torch.isfinite(method.head.weight).all()
    assert set(method._class_stats.keys()) == {0, 1, 2, 3}


def test_cur_adapter_trains_but_base_stays_frozen():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)

    base_before = copy.deepcopy(method.backbone.base.state_dict())
    adapter_before = copy.deepcopy(method.backbone.cur_adapter.state_dict())
    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )
    base_after = method.backbone.base.state_dict()
    adapter_after = method.backbone.cur_adapter.state_dict()

    for key in base_before:
        assert torch.equal(base_before[key], base_after[key]), f"frozen base param {key} changed"
    assert any(not torch.equal(adapter_before[k], adapter_after[k]) for k in adapter_before)


def test_classifier_output_shape_matches_head_size():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=1,
    )
    exp = _FakeExperience(_FakeLabelSpace(head_size=3, new_classes=[0, 1, 2]), task_label=0)
    method.before_experience(exp, ctx)
    method.train_experience(
        exp, _loader(torch.randn(9, 8, 8), torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])), ctx
    )

    classifier = method.classifier()
    logits = classifier(torch.randn(5, 8, 8))
    assert logits.shape == (5, 3)
    assert torch.isfinite(logits).all()


def test_state_dict_round_trip():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=2,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )
    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    method.train_experience(
        exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx
    )

    state = method.state_dict()

    other = _make_method()
    other.before_experience(exp0, ctx)
    other.before_experience(exp1, ctx)
    other.load_state_dict(state)

    assert torch.equal(other.head.weight, method.head.weight)
    assert set(other._class_stats.keys()) == set(method._class_stats.keys())
    other_backbone_state = other.backbone.state_dict()
    method_backbone_state = method.backbone.state_dict()
    for key in method_backbone_state:
        assert torch.equal(method_backbone_state[key], other_backbone_state[key])
