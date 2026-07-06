"""Incremental classifier heads (SPEC §6.2): implemented once, shared by all methods."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalHead(nn.Module):
    """A linear classifier head that grows its class dimension over time.

    Args:
        feature_dim: Backbone feature width.
        cosine: If ``True``, L2-normalizes both features and weight rows and
            scales the resulting cosine similarities by a learned scalar
            (SimpleCIL's mode: each row *is* a class prototype direction).
            If ``False``, a plain linear layer.
        num_classes: Initial width (usually 0; grown via ``expand_to``).
        scale: Initial value of the learned cosine scale (ignored if
            ``cosine=False``).
    """

    def __init__(
        self, feature_dim: int, cosine: bool = False, num_classes: int = 0, scale: float = 10.0
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.cosine = cosine
        self.weight = nn.Parameter(torch.zeros(num_classes, feature_dim))
        self.scale = nn.Parameter(torch.tensor(float(scale))) if cosine else None

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    def expand_to(self, new_num_classes: int) -> None:
        """Grow the head to *new_num_classes*, preserving existing rows.

        No-op if the head is already at least that wide (idempotent --
        class counts never shrink in continual learning).
        """
        if new_num_classes <= self.num_classes:
            return
        old_weight = self.weight.data
        new_weight = torch.zeros(new_num_classes, self.feature_dim, dtype=old_weight.dtype)
        new_weight[: old_weight.shape[0]] = old_weight
        nn.init.normal_(new_weight[old_weight.shape[0] :], std=0.01)
        self.weight = nn.Parameter(new_weight)

    def set_prototype(self, class_id: int, vector: torch.Tensor) -> None:
        """Directly assign class *class_id*'s weight row (closed-form, no grad)."""
        if class_id >= self.num_classes:
            raise ValueError(
                f"class_id {class_id} exceeds current head width {self.num_classes}; "
                "call expand_to first."
            )
        with torch.no_grad():
            self.weight.data[class_id] = vector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.cosine:
            features = F.normalize(features, dim=-1)
            weight = F.normalize(self.weight, dim=-1)
            assert self.scale is not None
            return self.scale * F.linear(features, weight)
        return F.linear(features, self.weight)


class RandomProjectionRidgeHead(nn.Module):
    """RanPAC's closed-form head (SPEC §6.5): a frozen random Gaussian
    projection expands frozen backbone features into a wider, nonlinear
    random feature space (``relu(x @ W_rand)``), and a ridge-regression
    linear head is solved in closed form from *accumulated sufficient
    statistics* -- never gradient-trained.

    ``G`` (``[M, M]``, Gram matrix of projected features) and ``Q``
    (``[M, C]``, feature/one-hot-label correlation) are updated once per
    experience via ``accumulate``; ``solve`` re-derives ``weight`` from
    scratch each time (``torch.linalg.solve(G + ridge*I, Q)``) -- this is
    the accumulate-sufficient-statistics trick PILOT's RanPAC uses, not a
    literal Sherman-Morrison-Woodbury inverse update. ``M`` (projection
    width) is scaled down from PILOT's 10000 (a 768-dim pretrained ViT) to
    fit this project's tiny synthetic-dataset backbones.
    """

    #: Explicit types for mypy -- ``register_buffer`` alone doesn't give it
    #: enough to infer these aren't plain ``nn.Module`` attributes.
    w_rand: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor

    def __init__(self, feature_dim: int, projection_dim: int = 256, num_classes: int = 0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.register_buffer("w_rand", torch.randn(feature_dim, projection_dim))
        self.register_buffer("g", torch.zeros(projection_dim, projection_dim))
        self.register_buffer("q", torch.zeros(projection_dim, num_classes))
        self.weight = nn.Parameter(torch.zeros(num_classes, projection_dim))

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    def project(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features @ self.w_rand)

    def expand_to(self, new_num_classes: int) -> None:
        """Grow ``q``/``weight`` to *new_num_classes* columns/rows, preserving
        existing ones. No-op if already at least that wide."""
        if new_num_classes <= self.num_classes:
            return
        old_q = self.q
        new_q = torch.zeros(
            self.projection_dim, new_num_classes, dtype=old_q.dtype, device=old_q.device
        )
        new_q[:, : old_q.shape[1]] = old_q
        self.q = new_q

        old_weight = self.weight.data
        new_weight = torch.zeros(
            new_num_classes, self.projection_dim, dtype=old_weight.dtype, device=old_weight.device
        )
        new_weight[: old_weight.shape[0]] = old_weight
        self.weight = nn.Parameter(new_weight)

    def accumulate(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Fold one batch/experience's features into the running ``G``/``Q``
        sufficient statistics (no solve yet -- call ``solve`` after)."""
        phi = self.project(features)
        y_onehot = F.one_hot(targets, num_classes=self.num_classes).to(phi.dtype)
        self.g = self.g + phi.t() @ phi
        self.q = self.q + phi.t() @ y_onehot

    def solve(self, ridge: float) -> None:
        """Recompute ``weight`` in closed form from the current ``G``/``Q``."""
        eye = torch.eye(self.projection_dim, dtype=self.g.dtype, device=self.g.device)
        solved = torch.linalg.solve(self.g + ridge * eye, self.q)  # [M, C]
        self.weight = nn.Parameter(solved.t().to(self.weight.dtype))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        phi = self.project(features)
        return F.linear(phi, self.weight)
