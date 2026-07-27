"""Incremental classifier heads (SPEC §6.2): implemented once, shared by all methods."""

from __future__ import annotations

from typing import List

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
        new_weight = torch.zeros(
            new_num_classes, self.feature_dim, dtype=old_weight.dtype, device=old_weight.device
        )
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


class EaseHead(nn.Module):
    """EASE's growing-dim head (SPEC §6.5): ``weight`` is ``[num_classes,
    num_blocks, block_dim]`` -- one row per class, spanning every adapter
    block (one per experience) introduced so far. Each class has a "home
    block" (the block index it was first introduced in); at inference,
    cosine similarity is computed per block and summed, with non-home
    blocks down-weighted by ``alpha`` -- PILOT's own reweighting scheme
    (``EaseCosineLinear.forward_reweight``).

    A class's *non-home*-block rows can't be recomputed from real data once
    that class's images are gone (no exemplar memory in this framework), so
    a newly added block's rows for already-existing classes are filled by a
    cosine-similarity-weighted combination of the newly-introduced classes'
    own rows in that same block -- a structurally faithful simplification
    of PILOT's ``solve_similarity``/``solve_sim_reset``. The caller (
    ``clover/methods/ease.py``) is responsible for this interpolation;
    ``EaseHead`` itself only owns storage/growth/the reweighted forward
    pass.
    """

    def __init__(self, block_dim: int, alpha: float = 0.1, scale: float = 10.0) -> None:
        super().__init__()
        self.block_dim = block_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.zeros(0, 0, block_dim))
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.home_block: List[int] = []

    @property
    def num_classes(self) -> int:
        return self.weight.shape[0]

    @property
    def num_blocks(self) -> int:
        return self.weight.shape[1]

    def add_block(self) -> None:
        """Grow the block dimension by one, zero-initialized (filled in by
        the caller via ``set_block_row``). No-op guard isn't needed here --
        the caller (``EASE.before_experience``) calls this exactly once per
        experience, matching the replay-safe growth pattern every other
        per-experience-growing head/pool in this codebase uses."""
        old = self.weight.data
        new = torch.zeros(
            old.shape[0], old.shape[1] + 1, self.block_dim, dtype=old.dtype, device=old.device
        )
        new[:, : old.shape[1], :] = old
        self.weight = nn.Parameter(new)

    def expand_classes(self, new_num_classes: int, home_block: int) -> None:
        """Grow the class dimension to *new_num_classes*, recording
        *home_block* for each newly added class row. No-op if already at
        least that wide."""
        if new_num_classes <= self.num_classes:
            return
        old = self.weight.data
        new = torch.zeros(
            new_num_classes, old.shape[1], self.block_dim, dtype=old.dtype, device=old.device
        )
        new[: old.shape[0]] = old
        self.weight = nn.Parameter(new)
        self.home_block.extend([home_block] * (new_num_classes - len(self.home_block)))

    def set_block_row(self, class_id: int, block_id: int, vector: torch.Tensor) -> None:
        with torch.no_grad():
            self.weight.data[class_id, block_id] = vector

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``[batch, num_blocks, block_dim]`` (one slice per
        stored adapter, e.g. ``EaseAdapterViT.forward``'s output)."""
        features_n = F.normalize(features, dim=-1)
        weight_n = F.normalize(self.weight, dim=-1)  # [C, N, D]
        similarity = torch.einsum("bnd,cnd->bcn", features_n, weight_n)  # [batch, C, N]

        home = torch.tensor(self.home_block, dtype=torch.long, device=features.device)  # [C]
        block_idx = torch.arange(self.num_blocks, device=features.device)  # [N]
        is_home = home.unsqueeze(1) == block_idx.unsqueeze(0)  # [C, N]
        one = torch.tensor(1.0, device=features.device)
        alpha = torch.tensor(self.alpha, device=features.device)
        weights = torch.where(is_home, one, alpha)  # [C, N]

        logits = (similarity * weights.unsqueeze(0)).sum(dim=-1)  # [batch, C]
        return self.scale * logits
