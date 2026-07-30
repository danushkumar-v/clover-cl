"""RanPAC (SPEC §6.5): one-shot adapter tuning (like APER-Adapter), then a
permanently-frozen backbone plus a random-projection ridge-regression head
-- no gradient training on the head, ever. Clean-room reimplementation (no
code copied from LAMDA-PILOT) of the recipe confirmed via read-only
research on ``models/ranpac.py``/``utils/inc_net.py`` (``SimpleVitNet``): a
random Gaussian projection expands frozen features into a wider nonlinear
space (``relu(x @ W_rand)``), and ridge regression is solved in closed form
from accumulated sufficient statistics rather than a literal
Sherman-Morrison-Woodbury inverse update. The ridge parameter is chosen
per-experience by a small grid search against a held-out split of that
experience's own data, mirroring PILOT's ``optimise_ridge_parameter``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.adapter_common import AdapterMethodBase
from clover.methods.base import TrainContext
from clover.methods.heads import RandomProjectionRidgeHead

#: Scaled down from PILOT's 17-value 1e-8..1e8 grid (tuned for 768-dim
#: pretrained-ViT features) -- this project's tiny synthetic-dataset
#: features don't need that range.
_RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

#: PILOT's published RanPAC projection width (``M``, see
#: ``bench/configs/methods/ranpac.yaml``) is a fixed 10000 at a 768-dim
#: ViT-B/16 feature. ``_default_projection_dim`` scales that proportionally
#: to any base's own feature_dim, floored at 256 -- the literal this
#: project's TinyViT (16-dim) safety gate was already tuned against -- so
#: recovering the exact published value at ViT-B/16 scale never shrinks a
#: synthetic backbone's projection width below what's known to work.
_PUBLISHED_M = 10000
_PUBLISHED_FEATURE_DIM = 768
_MIN_PROJECTION_DIM = 256


def _default_projection_dim(feature_dim: int) -> int:
    """Scale RanPAC's published projection width ``M`` to *feature_dim*.

    Args:
        feature_dim: The adapter-wrapped backbone's own feature width.

    Returns:
        ``round(feature_dim * _PUBLISHED_M / _PUBLISHED_FEATURE_DIM)``,
        floored at :data:`_MIN_PROJECTION_DIM`. At ``feature_dim=768``
        (ViT-B/16) this is exactly the published 10000; at TinyViT's 16-dim
        it floors to 256, unchanged from before this rule existed.
    """
    scaled = round(feature_dim * _PUBLISHED_M / _PUBLISHED_FEATURE_DIM)
    return max(_MIN_PROJECTION_DIM, scaled)


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: RandomProjectionRidgeHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register_method("ranpac")
class RanPAC(AdapterMethodBase):
    name = "ranpac"
    default_backbone = "vit_adapter"

    def __init__(self, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        self._projection_dim_cfg = projection_dim
        #: Resolved to a concrete int by `_build_head` (called from
        #: `build()`, before any other use); `None` only ever describes the
        #: "derive it" constructor argument, never this attribute post-build.
        self.projection_dim: int = 0
        self.head: RandomProjectionRidgeHead | None = None

    def _build_head(self, adapter_feature_dim: int) -> None:
        self.projection_dim = (
            self._projection_dim_cfg
            if self._projection_dim_cfg is not None
            else _default_projection_dim(adapter_feature_dim)
        )
        self.head = RandomProjectionRidgeHead(adapter_feature_dim, self.projection_dim)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        super().before_experience(exp, ctx)
        assert self.head is not None
        self.head.expand_to(exp.label_space.head_size)
        self.head = self.head.to(ctx.device)

    def _update_head(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        all_features: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                all_features.append(self.backbone(images))
                all_targets.append(targets.to(ctx.device))
        features = torch.cat(all_features, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # A local generator (seeded from the experience index, not the
        # global RNG) for the train/val split used only to pick a ridge
        # value -- never draw an unseeded/global-RNG-consuming random call
        # in library code (CLAUDE.md); this doesn't have access to the
        # run's actual seed (TrainContext doesn't carry one to methods), so
        # it's deterministic per-experience rather than per-run, which only
        # affects which ridge candidate is chosen, not the correctness of
        # the closed-form solve itself.
        generator = torch.Generator()
        generator.manual_seed(exp.task_label)
        n = features.shape[0]
        # torch.Generator() is CPU-only, so randperm's output is a CPU
        # tensor regardless of ctx.device; moved to features.device before
        # use as an index -- indexing a CUDA tensor with a CPU LongTensor
        # raises at runtime (device-correctness re-audit, P11-B2; the same
        # fix pattern classifier_alignment.py's _sample_gaussian already
        # uses for its own CPU-generator draw).
        perm = torch.randperm(n, generator=generator).to(features.device)
        split = max(1, int(n * 0.8))
        train_idx, val_idx = perm[:split], perm[split:]
        if val_idx.numel() == 0:
            val_idx = train_idx

        with torch.no_grad():
            phi_train = self.head.project(features[train_idx])
            y_train = F.one_hot(targets[train_idx], num_classes=self.head.num_classes).to(
                phi_train.dtype
            )
            trial_g = self.head.g + phi_train.t() @ phi_train
            trial_q = self.head.q + phi_train.t() @ y_train

            phi_val = self.head.project(features[val_idx])
            y_val = F.one_hot(targets[val_idx], num_classes=self.head.num_classes).to(phi_val.dtype)
            eye = torch.eye(self.head.projection_dim, dtype=trial_g.dtype, device=trial_g.device)

            best_ridge = _RIDGE_GRID[0]
            best_mse = float("inf")
            for ridge in _RIDGE_GRID:
                weight = torch.linalg.solve(trial_g + ridge * eye, trial_q)
                pred = phi_val @ weight
                mse = ((pred - y_val) ** 2).mean().item()
                if mse < best_mse:
                    best_mse = mse
                    best_ridge = ridge

            self.head.accumulate(features, targets)
            self.head.solve(best_ridge)

    def classifier(self) -> nn.Module:
        assert self.backbone is not None and self.head is not None
        return _Classifier(self.backbone, self.head)

    def state_dict(self) -> Dict[str, Any]:
        assert self.head is not None
        state = self._base_state()
        state["head"] = self.head.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        assert self.head is not None
        self._load_base_state(state)
        self.head.load_state_dict(state["head"])
