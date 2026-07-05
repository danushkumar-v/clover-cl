"""SimpleCIL correctness in isolation, no Trainer (SPEC §6.5)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.simple_cil import SimpleCIL


def _make_method() -> SimpleCIL:
    method = get_method("simplecil")()
    info = StreamInfo(dataset="synthetic", nb_experiences=2, total_classes=4, input_size=8)
    method.build(info, {})
    return method


def _loader(images: torch.Tensor, targets: torch.Tensor) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=8)


def test_registered_as_simplecil():
    assert get_method("simplecil") is SimpleCIL


def test_backbone_is_frozen():
    method = _make_method()
    assert all(not p.requires_grad for p in method.backbone.parameters())


def test_prototype_matches_hand_computed_mean():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))

    class Exp:
        label_space = type("LS", (), {"head_size": 2})()

    method.before_experience(Exp(), ctx)

    images = torch.randn(6, 8, 8)
    targets = torch.tensor([0, 0, 0, 1, 1, 1])
    method.train_experience(Exp(), _loader(images, targets), ctx)

    with torch.no_grad():
        expected_0 = method.backbone(images[:3]).mean(dim=0)
        expected_1 = method.backbone(images[3:]).mean(dim=0)

    assert torch.allclose(method.head.weight[0], expected_0, atol=1e-6)
    assert torch.allclose(method.head.weight[1], expected_1, atol=1e-6)


def test_revisited_class_prototype_is_recomputed_not_accumulated():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))

    class Exp:
        label_space = type("LS", (), {"head_size": 1})()

    method.before_experience(Exp(), ctx)

    first_images = torch.randn(4, 8, 8)
    method.train_experience(Exp(), _loader(first_images, torch.zeros(4, dtype=torch.long)), ctx)
    first_prototype = method.head.weight[0].clone()

    second_images = torch.randn(4, 8, 8)
    method.train_experience(Exp(), _loader(second_images, torch.zeros(4, dtype=torch.long)), ctx)
    second_prototype = method.head.weight[0].clone()

    with torch.no_grad():
        expected_second = method.backbone(second_images).mean(dim=0)

    assert torch.allclose(second_prototype, expected_second, atol=1e-6)
    assert not torch.allclose(second_prototype, first_prototype)


def test_classifier_produces_logits_matching_head_width():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))

    class Exp:
        label_space = type("LS", (), {"head_size": 3})()

    method.before_experience(Exp(), ctx)
    classifier = method.classifier()
    logits = classifier(torch.randn(5, 8, 8))
    assert logits.shape == (5, 3)


def test_state_dict_round_trip():
    method = _make_method()
    ctx = TrainContext(device=torch.device("cpu"))

    class Exp:
        label_space = type("LS", (), {"head_size": 2})()

    method.before_experience(Exp(), ctx)
    method.head.set_prototype(0, torch.ones(method.head.feature_dim))
    state = method.state_dict()

    other = _make_method()
    other.before_experience(Exp(), ctx)
    other.load_state_dict(state)
    assert torch.equal(other.head.weight, method.head.weight)
