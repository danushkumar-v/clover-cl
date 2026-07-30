"""Full real-backbone lifecycle for the adapter family (P11-B2 Task 4):
``build -> before_experience -> train_experience -> classifier()`` driven
on a real (untrained) timm ViT -- ``vit_tiny_patch16_224``,
``pretrained=False`` (architecture only, no download, CPU) -- rather than
``TinyViT``.

EASE/MOS legally require ``before_experience`` (which calls
``grow()``/``snapshot()``) before ``forward()`` is legal -- calling
``forward()`` straight after ``build()`` is a test-contract error, not a
method bug (this is exactly the mistake P11_PLAN.md's "prior art" stash
made). Every test here drives the real lifecycle via ``make_experience``
(a real ``Experience``/``LabelSpaceView``, not a fake label-space stub --
``tests/conftest.py``), matching how the registry-driven safety gate in
``tests/test_method_registry_safety_gate.py`` already exercises every
registered method (including these five) on this same real backbone --
this file additionally asserts a finite *loss* value per method, which
that shared gate does not compute.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from clover.methods import get_method
from clover.methods.base import StreamInfo, TrainContext

REAL_BASE = "vit_tiny_patch16_224"
IMG_SIZE = 224
CHANNELS = 3
BATCH_SIZE = 2

_ADAPTER_METHODS = ["aper_adapter", "ranpac", "ease", "mos", "tuna"]


def _real_loader(classes: list[int]) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Two samples per class, batch_size=2 -- matches the shape
    ``test_registered_method_is_finite_on_a_real_timm_base`` already
    validates for every registered method."""
    targets = torch.tensor([c for c in classes for _ in range(2)])
    images = torch.randn(len(targets), CHANNELS, IMG_SIZE, IMG_SIZE)
    return DataLoader(TensorDataset(images, targets), batch_size=BATCH_SIZE), images, targets


@pytest.mark.parametrize("method_name", _ADAPTER_METHODS)
def test_full_lifecycle_on_real_timm_base_is_finite(method_name, make_experience):
    method = get_method(method_name)()
    stream_info = StreamInfo(
        dataset="synthetic",
        nb_experiences=1,
        total_classes=2,
        input_size=IMG_SIZE,
        channels=CHANNELS,
    )
    method.build(stream_info, {"base_model": REAL_BASE, "pretrained": False})

    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
        epochs=1,
    )
    exp = make_experience(0, [0, 1], head_size=2)
    loader, images, targets = _real_loader([0, 1])

    method.before_experience(exp, ctx)
    method.train_experience(exp, loader, ctx)

    classifier = method.classifier()
    for p in classifier.parameters():
        assert torch.isfinite(p).all(), f"{method_name}: non-finite classifier param"

    with torch.no_grad():
        logits = classifier(images)
    assert logits.shape == (len(images), 2)
    assert torch.isfinite(logits).all(), f"{method_name}: non-finite logits"

    loss = F.cross_entropy(logits, targets)
    assert torch.isfinite(loss).all(), f"{method_name}: non-finite loss"
