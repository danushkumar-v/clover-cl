"""EASE correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.ease import EASE
from clover.methods.heads import EaseHead


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


def _make_method() -> EASE:
    method = get_method("ease")()
    info = StreamInfo(dataset="synthetic", nb_experiences=3, total_classes=6, input_size=8, channels=1)
    method.build(info, {"depth": 2, "bottleneck_dim": 4})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_ease():
    assert get_method("ease") is EASE


# --- EaseHead unit tests -----------------------------------------------


def test_ease_head_add_block_and_expand_classes():
    head = EaseHead(block_dim=4)
    head.add_block()
    head.expand_classes(2, home_block=0)
    assert head.num_blocks == 1
    assert head.num_classes == 2
    assert head.home_block == [0, 0]

    head.add_block()
    head.expand_classes(4, home_block=1)
    assert head.num_blocks == 2
    assert head.num_classes == 4
    assert head.home_block == [0, 0, 1, 1]
    assert head.weight.shape == (4, 2, 4)


def test_ease_head_forward_shape_and_finite():
    head = EaseHead(block_dim=4)
    head.add_block()
    head.expand_classes(2, home_block=0)
    head.set_block_row(0, 0, torch.randn(4))
    head.set_block_row(1, 0, torch.randn(4))

    logits = head(torch.randn(3, 1, 4))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


def test_ease_head_home_block_gets_full_weight_others_get_alpha():
    head = EaseHead(block_dim=4, alpha=0.0)  # alpha=0 -> non-home blocks contribute nothing
    head.add_block()
    head.expand_classes(1, home_block=0)
    head.add_block()  # class 0's home block is 0; block 1 is non-home for it
    home_vec = torch.randn(4)
    other_vec = torch.randn(4)
    head.set_block_row(0, 0, home_vec)
    head.set_block_row(0, 1, other_vec)

    features = torch.zeros(1, 2, 4)
    features[0, 0] = home_vec  # matches home block exactly
    features[0, 1] = -other_vec  # would anti-match the non-home block, but alpha=0 ignores it
    logits = head(features)
    # With alpha=0, only the home-block term contributes; a perfect match there
    # should give a large positive logit despite the anti-matching non-home block.
    assert logits[0, 0].item() > 0


# --- EASE method tests ---------------------------------------------------


def test_backbone_and_head_grow_every_experience():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    assert method.backbone.num_blocks == 1
    assert method.head.num_blocks == 1
    assert method.head.num_classes == 2

    method.train_experience(
        exp0, _loader(torch.randn(8, 8, 8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx
    )

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    assert method.backbone.num_blocks == 2
    assert method.head.num_blocks == 2
    assert method.head.num_classes == 4


def test_only_most_recent_adapter_set_trains():
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
    first_adapter_state = {k: v.clone() for k, v in method.backbone.adapter_sets[0].state_dict().items()}

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    method.train_experience(
        exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx
    )

    after_state = method.backbone.adapter_sets[0].state_dict()
    for key in first_adapter_state:
        assert torch.equal(first_adapter_state[key], after_state[key]), f"frozen adapter param {key} changed"
    assert all(not p.requires_grad for p in method.backbone.adapter_sets[0].parameters())
    assert all(p.requires_grad for p in method.backbone.adapter_sets[1].parameters())


def test_old_classes_survive_a_new_block_with_finite_logits():
    method = _make_method()
    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-2),
        epochs=3,
    )
    exp0 = _FakeExperience(_FakeLabelSpace(head_size=2, new_classes=[0, 1]), task_label=0)
    method.before_experience(exp0, ctx)
    images0 = torch.randn(8, 8, 8)
    method.train_experience(exp0, _loader(images0, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])), ctx)

    exp1 = _FakeExperience(_FakeLabelSpace(head_size=4, new_classes=[2, 3]), task_label=1)
    method.before_experience(exp1, ctx)
    method.train_experience(
        exp1, _loader(torch.randn(8, 8, 8), torch.tensor([2, 2, 2, 2, 3, 3, 3, 3])), ctx
    )

    classifier = method.classifier()
    with torch.no_grad():
        logits_old = classifier(images0)
    assert logits_old.shape == (8, 4)
    assert torch.isfinite(logits_old).all()


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
    assert other.head.home_block == method.head.home_block
    other_backbone_state = other.backbone.state_dict()
    method_backbone_state = method.backbone.state_dict()
    for key in method_backbone_state:
        assert torch.equal(method_backbone_state[key], other_backbone_state[key])
