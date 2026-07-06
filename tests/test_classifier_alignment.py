"""Gaussian-resample classifier alignment correctness (SPEC §6.5, MOS/TUNA)."""

from __future__ import annotations

import torch

from clover.methods.classifier_alignment import gaussian_resample_finetune, update_class_stats
from clover.methods.heads import IncrementalHead


def test_update_class_stats_stores_mean_and_damped_covariance():
    stats = {}
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    update_class_stats(stats, class_id=0, features=features)

    mean, cov = stats[0]
    assert torch.allclose(mean, features.mean(dim=0))
    assert cov.shape == (2, 2)
    # Damped -- diagonal is at least the damping term even for zero-variance dims.
    assert (torch.diagonal(cov) >= 1e-4).all()


def test_update_class_stats_single_sample_uses_damping_only():
    stats = {}
    update_class_stats(stats, class_id=0, features=torch.tensor([[1.0, 2.0]]))
    mean, cov = stats[0]
    assert torch.equal(mean, torch.tensor([1.0, 2.0]))
    assert torch.allclose(cov, 1e-4 * torch.eye(2))


def test_gaussian_resample_finetune_is_a_no_op_on_empty_stats():
    head = IncrementalHead(feature_dim=4, cosine=True, num_classes=2)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    generator = torch.Generator()
    generator.manual_seed(0)
    gaussian_resample_finetune(head, {}, lr=1e-2, epochs=5, samples_per_class=8, generator=generator)
    after = head.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_gaussian_resample_finetune_moves_head_toward_separating_classes():
    head = IncrementalHead(feature_dim=2, cosine=False, num_classes=2)
    stats = {
        0: (torch.tensor([-3.0, 0.0]), 0.1 * torch.eye(2)),
        1: (torch.tensor([3.0, 0.0]), 0.1 * torch.eye(2)),
    }
    generator = torch.Generator()
    generator.manual_seed(0)
    gaussian_resample_finetune(head, stats, lr=0.5, epochs=50, samples_per_class=16, generator=generator)

    with torch.no_grad():
        test_features = torch.tensor([[-3.0, 0.0], [3.0, 0.0]])
        logits = head(test_features)
    preds = logits.argmax(dim=-1)
    assert preds.tolist() == [0, 1]


def test_gaussian_resample_finetune_is_deterministic_given_a_seeded_generator():
    head_a = IncrementalHead(feature_dim=3, cosine=True, num_classes=2)
    head_b = IncrementalHead(feature_dim=3, cosine=True, num_classes=2)
    head_b.load_state_dict(head_a.state_dict())
    stats = {
        0: (torch.zeros(3), torch.eye(3)),
        1: (torch.ones(3), torch.eye(3)),
    }

    gen_a = torch.Generator()
    gen_a.manual_seed(42)
    gaussian_resample_finetune(head_a, stats, lr=0.1, epochs=3, samples_per_class=4, generator=gen_a)

    gen_b = torch.Generator()
    gen_b.manual_seed(42)
    gaussian_resample_finetune(head_b, stats, lr=0.1, epochs=3, samples_per_class=4, generator=gen_b)

    for key in head_a.state_dict():
        assert torch.equal(head_a.state_dict()[key], head_b.state_dict()[key])
