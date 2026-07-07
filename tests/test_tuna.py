"""TUNA correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.tuna import TUNA


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


def _make_method() -> TUNA:
    method = get_method("tuna")(crct_epochs=3, ca_lr=5e-3, samples_per_class=8)
    info = StreamInfo(dataset="synthetic", nb_experiences=3, total_classes=6, input_size=8, channels=1)
    method.build(info, {"depth": 2, "bottleneck_dim": 4})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_tuna():
    assert get_method("tuna") is TUNA


def test_grow_happens_in_before_experience():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    assert len(method.backbone.adapter_list) == 1

    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    assert len(method.backbone.adapter_list) == 2


def test_merge_recomputed_after_training_each_experience():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    merged_before = copy.deepcopy(method.backbone.merged_adapter.state_dict())
    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )
    merged_after = method.backbone.merged_adapter.state_dict()
    assert any(not torch.equal(merged_before[k], merged_after[k]) for k in merged_before)


def test_only_current_adapter_trains_others_frozen():
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
    first_adapter_state = {k: v.clone() for k, v in method.backbone.adapter_list[0].state_dict().items()}

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    assert all(not p.requires_grad for p in method.backbone.adapter_list[0].parameters())
    method.train_experience(
        exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx
    )

    after_state = method.backbone.adapter_list[0].state_dict()
    for key in first_adapter_state:
        assert torch.equal(first_adapter_state[key], after_state[key]), f"frozen adapter param {key} changed"


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
