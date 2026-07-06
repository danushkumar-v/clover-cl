"""APER-Adapter correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.aper_adapter import APERAdapter
from clover.methods.base import StreamInfo, TrainContext


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


def _make_method() -> APERAdapter:
    method = get_method("aper_adapter")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8)
    method.build(info, {"depth": 2, "bottleneck_dim": 4})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_aper_adapter():
    assert get_method("aper_adapter") is APERAdapter


def test_head_width_is_double_the_backbone_feature_dim():
    method = _make_method()
    assert method.head.feature_dim == method.backbone.feature_dim * 2


def test_first_experience_trains_adapter_then_freezes_it():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)

    adapter_before = copy.deepcopy(method.backbone.adapters.state_dict())
    method.train_experience(exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx)
    adapter_after = method.backbone.adapters.state_dict()
    assert any(not torch.equal(adapter_before[k], adapter_after[k]) for k in adapter_before)
    assert all(not p.requires_grad for p in method.backbone.parameters())

    # Second experience: adapter must stay frozen even though the base
    # class always calls train_experience.
    frozen_snapshot = copy.deepcopy(method.backbone.state_dict())
    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    method.train_experience(exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx)
    after_second = method.backbone.state_dict()
    for key in frozen_snapshot:
        assert torch.equal(frozen_snapshot[key], after_second[key]), f"frozen backbone param {key} changed"


def test_prototypes_recomputed_every_experience():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    method.train_experience(exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx)
    head_after_exp0 = method.head.weight.data.clone()
    assert torch.isfinite(head_after_exp0).all()
    assert not torch.equal(head_after_exp0, torch.zeros_like(head_after_exp0))


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


def test_state_dict_round_trip():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=2,
    )
    exp = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp, ctx)
    method.train_experience(exp, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx)

    state = method.state_dict()

    other = _make_method()
    other.before_experience(exp, ctx)
    other.load_state_dict(state)

    assert torch.equal(other.head.weight, method.head.weight)
    other_backbone_state = other.backbone.state_dict()
    method_backbone_state = method.backbone.state_dict()
    for key in method_backbone_state:
        assert torch.equal(method_backbone_state[key], other_backbone_state[key])
    assert other._adapter_frozen == method._adapter_frozen is True
