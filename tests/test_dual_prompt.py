"""DualPrompt correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import copy

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.dual_prompt import DualPrompt


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


def _make_method() -> DualPrompt:
    method = get_method("dualprompt")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8, channels=1)
    method.build(info, {"depth": 3, "pool_size": 6, "top_k": 2})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_dualprompt():
    assert get_method("dualprompt") is DualPrompt


def test_base_vit_is_frozen_prompts_and_head_are_not():
    method = _make_method()
    assert all(not p.requires_grad for p in method.backbone.base.parameters())
    assert method.backbone.g_prompt.requires_grad
    assert method.backbone.e_pool.requires_grad
    assert all(p.requires_grad for p in method.head.parameters())


def test_gradient_step_changes_prompts_and_head_not_base():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]))
    method.before_experience(exp, ctx)

    base_before = copy.deepcopy(method.backbone.base.state_dict())
    g_prompt_before = method.backbone.g_prompt.clone()
    e_pool_before = method.backbone.e_pool.clone()
    head_before = copy.deepcopy(method.head.state_dict())

    images = torch.randn(8, 8, 8)
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    method.train_experience(exp, _loader(images, targets), ctx)

    base_after = method.backbone.base.state_dict()
    for key in base_before:
        assert torch.equal(base_before[key], base_after[key]), f"frozen base param {key} changed"
    assert not torch.equal(g_prompt_before, method.backbone.g_prompt)
    assert not torch.equal(e_pool_before, method.backbone.e_pool)
    head_after = method.head.state_dict()
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
        method.backbone.g_prompt.fill_(0.5)

    state = method.state_dict()

    other = _make_method()
    other.before_experience(exp, ctx)
    other.load_state_dict(state)

    assert torch.equal(other.backbone.g_prompt, method.backbone.g_prompt)
    assert torch.equal(other.head.weight, method.head.weight)
