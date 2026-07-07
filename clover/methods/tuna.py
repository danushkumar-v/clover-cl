"""TUNA (SPEC §6.5): a growing per-experience adapter list (like EASE),
combined via EMR-merge (``clover/backbones/adapter_tuna.py:TunaAdapterViT``)
into a single consensus adapter used directly for evaluation -- PILOT's
per-sample entropy-based adapter routing at test time is simplified out in
favor of using the merged adapter directly (more faithful to "TUNA" than
skipping the merge itself would be, and simpler than reimplementing
per-sample routing).

Unlike EASE (growing feature width) or MOS (fixed-width head gradient
-trained then prototype-overwritten), TUNA's head is a plain, global-width
``IncrementalHead(cosine=True)`` trained via a CosFace-style angular
-margin loss (``angular_margin_ce``) with no closed-form prototype
overwrite step -- PILOT's own ``TunaLinear`` (per-task heads, concatenated,
never frozen) is an implementation detail for isolating gradient to the
newest task's columns, a property ``angular_margin_ce``'s masking already
gives for free without needing separate per-task ``nn.Linear`` objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from clover.backbones.loader import resolve_base_model
from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.methods.classifier_alignment import (
    ClassStats,
    gaussian_resample_finetune,
    update_class_stats,
)
from clover.methods.heads import IncrementalHead
from clover.methods.losses import angular_margin_ce


class _Classifier(nn.Module):
    """``images -> logits`` through the EMR-merged consensus adapter."""

    def __init__(self, backbone: nn.Module, head: IncrementalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register_method("tuna")
class TUNA(CLMethod):
    name = "tuna"
    default_backbone = "vit_adapter_tuna"
    cacheable_features = False

    def __init__(
        self,
        margin: float = 0.0,
        crct_epochs: int = 10,
        ca_lr: float = 5e-3,
        samples_per_class: int = 32,
    ) -> None:
        self.backbone: Any = None
        self.head: Optional[IncrementalHead] = None
        self._class_stats: ClassStats = {}
        self.margin = margin
        self.crct_epochs = crct_epochs
        self.ca_lr = ca_lr
        self.samples_per_class = samples_per_class

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        backbone_kwargs = {k: v for k, v in cfg.items() if k != "backbone"}
        backbone_kwargs.setdefault("input_size", stream_info.input_size)
        backbone_kwargs.setdefault("in_chans", stream_info.channels)
        self.backbone = resolve_base_model(cfg.get("backbone", self.default_backbone), **backbone_kwargs)
        feature_dim: int = self.backbone.feature_dim
        self.head = IncrementalHead(feature_dim, cosine=True)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        self.backbone = self.backbone.to(ctx.device)
        self.backbone.grow()
        self.head = self.head.to(ctx.device)
        self.head.expand_to(exp.label_space.head_size)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        assert ctx.optimizer_factory is not None

        trainable = list(self.backbone.cur_adapter.parameters()) + list(self.head.parameters())
        optimizer = ctx.optimizer_factory(trainable)
        for _epoch in range(ctx.epochs):
            for images, targets in loader:
                images = images.to(ctx.device)
                targets = targets.to(ctx.device)
                optimizer.zero_grad()
                logits = self.head(self.backbone.forward_current(images))
                loss = angular_margin_ce(logits, targets, exp.label_space, margin=self.margin)
                loss.backward()
                optimizer.step()

        # Value-only update (merged_adapter's shape never changes) -- safe
        # here, unlike the backbone's structural growth in before_experience.
        self.backbone.recompute_merge()
        self._update_class_stats(exp, loader, ctx)

        if exp.task_label > 0:
            # Locally-seeded, not the global RNG (see classifier_alignment.py):
            # this runs inside train_experience, which resume-replay skips
            # for already-completed experiences.
            generator = torch.Generator()
            generator.manual_seed(exp.task_label)
            gaussian_resample_finetune(
                self.head,
                self._class_stats,
                lr=self.ca_lr,
                epochs=self.crct_epochs,
                samples_per_class=self.samples_per_class,
                generator=generator,
            )

    def _update_class_stats(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None
        embeddings_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                # The merged pass, matching what evaluation will actually
                # see -- not forward_current, which is this round's
                # not-yet-merged adapter alone.
                features = self.backbone(images)
                for feature, label in zip(features, targets.tolist()):
                    embeddings_by_class.setdefault(label, []).append(feature)

        for class_id, feats in embeddings_by_class.items():
            update_class_stats(self._class_stats, class_id, torch.stack(feats))

    def classifier(self) -> nn.Module:
        assert self.backbone is not None and self.head is not None
        return _Classifier(self.backbone, self.head)

    def state_dict(self) -> Dict[str, Any]:
        assert self.backbone is not None and self.head is not None
        return {
            "backbone": self.backbone.state_dict(),
            "head": self.head.state_dict(),
            "class_stats": dict(self._class_stats),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        assert self.backbone is not None and self.head is not None
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
        self._class_stats = dict(state["class_stats"])
