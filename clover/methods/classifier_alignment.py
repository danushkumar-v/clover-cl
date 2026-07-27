"""Classifier alignment (CA) via Gaussian resampling (SPEC §6.5, MOS/TUNA):
confirmed via read-only research on ``models/mos.py``/``models/tuna.py``
(``classifer_align``). As a shared adapter continues to change underneath
a shared classifier head, old classes' decision boundaries can drift out
of alignment even though nothing about those classes themselves changed.
CA counteracts this without replay: each class's ``(mean, covariance)`` is
stored once (from real data, no raw features retained -- damped by a small
diagonal term for numerical stability, matching PILOT's own
``+ 1e-4 * I``), then the head is fine-tuned via cross-entropy on
*synthetic* features resampled from ``N(mean, cov)`` for every class seen
so far, old and new alike.

Sampling goes through an explicitly passed ``torch.Generator`` rather than
the global RNG (Cholesky decomposition + ``randn`` instead of
``torch.distributions.MultivariateNormal.sample()``, which doesn't accept
one) -- the same reasoning as ``Trainer._build_loader``'s per-experience
seeding: this runs inside ``train_experience``, and a resumed run skips
replaying ``train_experience`` for already-completed experiences, so
drawing from the global RNG here would desynchronize a resumed run's RNG
position from an uninterrupted run's at the same point.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ClassStats = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


def update_class_stats(stats: ClassStats, class_id: int, features: torch.Tensor) -> None:
    """Recompute ``(mean, covariance)`` for one class from this experience's
    features, overwriting any previous entry for it. Classes not passed
    here (already seen in an earlier experience, whose images are gone)
    keep whatever stats they already have -- a structurally faithful
    simplification of PILOT's own approach, which is under the same
    no-exemplar-memory constraint.
    """
    mean = features.mean(dim=0)
    dim = features.shape[-1]
    if features.shape[0] > 1:
        cov = torch.cov(features.t()) + 1e-4 * torch.eye(dim, device=features.device)
    else:
        cov = 1e-4 * torch.eye(dim, device=features.device)
    stats[class_id] = (mean.detach(), cov.detach())


def _sample_gaussian(
    mean: torch.Tensor, cov: torch.Tensor, n: int, generator: torch.Generator
) -> torch.Tensor:
    chol = torch.linalg.cholesky(cov)
    # Draw on CPU (torch.Generator is a CPU generator -- keeps the draw
    # deterministic and device-independent), then move to the stats' device.
    z = torch.randn(n, mean.shape[0], generator=generator).to(mean.device)
    return mean.unsqueeze(0) + z @ chol.t()


def gaussian_resample_finetune(
    head: nn.Module,
    stats: ClassStats,
    lr: float,
    epochs: int,
    samples_per_class: int,
    generator: torch.Generator,
) -> None:
    """Fine-tune ``head`` via cross-entropy on synthetic features resampled
    from every stored class's ``(mean, covariance)`` -- no real images
    touched. No-op if ``stats`` is empty (nothing to align against yet).
    """
    if not stats:
        return
    class_ids = sorted(stats.keys())
    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    for _epoch in range(epochs):
        features = []
        targets = []
        for class_id in class_ids:
            mean, cov = stats[class_id]
            features.append(_sample_gaussian(mean, cov, samples_per_class, generator))
            targets.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
        features_t = torch.cat(features, dim=0)
        targets_t = torch.cat(targets, dim=0).to(features_t.device)

        optimizer.zero_grad()
        logits = head(features_t)
        loss = F.cross_entropy(logits, targets_t)
        loss.backward()
        optimizer.step()
