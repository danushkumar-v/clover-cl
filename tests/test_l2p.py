"""L2P correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.l2p import L2P


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

    def logit_mask(self, width):
        mask = torch.zeros(width, dtype=torch.bool)
        for c in self._new_classes | self._seen_classes:
            if c < width:
                mask[c] = True
        return mask

    @property
    def new_classes(self):
        return self._new_classes

    @property
    def seen_classes(self):
        return self._seen_classes


class _FakeExperience:
    def __init__(self, label_space):
        self.label_space = label_space


def _make_method() -> L2P:
    method = get_method("l2p")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8)
    method.build(info, {"pool_size": 6, "prompt_length": 3, "top_k": 2})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_l2p():
    assert get_method("l2p") is L2P


def test_base_vit_is_frozen_pool_and_head_are_not():
    method = _make_method()
    assert all(not p.requires_grad for p in method.backbone.base.parameters())
    assert all(p.requires_grad for p in method.backbone.pool.parameters())
    assert all(p.requires_grad for p in method.head.parameters())


def test_gradient_step_changes_prompt_key_and_head_not_base():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]))
    method.before_experience(exp, ctx)

    base_before = copy.deepcopy(method.backbone.base.state_dict())
    pool_before = copy.deepcopy(method.backbone.pool.state_dict())
    head_before = copy.deepcopy(method.head.state_dict())

    images = torch.randn(8, 8, 8)
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    method.train_experience(exp, _loader(images, targets), ctx)

    base_after = method.backbone.base.state_dict()
    pool_after = method.backbone.pool.state_dict()
    head_after = method.head.state_dict()

    for key in base_before:
        assert torch.equal(base_before[key], base_after[key]), f"frozen base param {key} changed"
    assert any(not torch.equal(pool_before[k], pool_after[k]) for k in pool_before)
    assert any(not torch.equal(head_before[k], head_after[k]) for k in head_before)


def test_classifier_output_shape_matches_head_size():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))
    exp = _FakeExperience(_FakeLabelSpace(head_size=3, new_classes=[0, 1, 2]))
    method.before_experience(exp, ctx)

    classifier = method.classifier()
    logits = classifier(torch.randn(5, 8, 8))
    assert logits.shape == (5, 3)


def test_state_dict_round_trip():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))
    exp = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]))
    method.before_experience(exp, ctx)
    with torch.no_grad():
        method.backbone.pool.prompt.fill_(0.5)

    state = method.state_dict()

    other = _make_method()
    other.before_experience(exp, ctx)
    other.load_state_dict(state)

    assert torch.equal(other.backbone.pool.prompt, method.backbone.pool.prompt)
    assert torch.equal(other.head.weight, method.head.weight)
