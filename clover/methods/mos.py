"""MOS (SPEC §6.5): a single continuously-trained adapter
(``clover/backbones/adapter_mos.py:MOSAdapterViT``), EMA-regularized
toward the mean of its own history every optimizer step, plus classifier
alignment (CA) via Gaussian resampling to counteract drift in the shared
head as the adapter keeps changing underneath it.
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
from clover.methods.losses import new_class_ce


class _Classifier(nn.Module):
    """``images -> logits`` averaged over every stored adapter's
    prediction through the same shared head -- a plain ensemble in place
    of PILOT's entropy-based self-refinement search (SPEC §6.5 scope
    simplification, agreed before implementing)."""

    def __init__(self, backbone: nn.Module, head: IncrementalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # [num_adapters, batch, feature_dim]
        logits = self.head(features)  # [num_adapters, batch, num_classes]
        return logits.mean(dim=0)


@register_method("mos")
class MOS(CLMethod):
    name = "mos"
    default_backbone = "vit_adapter_mos"
    cacheable_features = False

    def __init__(self, crct_epochs: int = 10, ca_lr: float = 5e-3, samples_per_class: int = 32) -> None:
        self.backbone: Any = None
        self.head: Optional[IncrementalHead] = None
        self._class_stats: ClassStats = {}
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
        if exp.task_label >= 1:
            self.backbone.snapshot()
        self.head = self.head.to(ctx.device)
        self.head.expand_to(exp.label_space.head_size)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        assert ctx.optimizer_factory is not None

        trainable = list(self.backbone.cur_adapter.parameters()) + list(self.head.parameters())
        optimizer = ctx.optimizer_factory(trainable)
        scheduler = ctx.make_scheduler(optimizer)
        for _epoch in range(ctx.epochs):
            for images, targets in loader:
                images = images.to(ctx.device)
                targets = targets.to(ctx.device)
                optimizer.zero_grad()
                logits = self.head(self.backbone.forward_current(images))
                loss = new_class_ce(logits, targets, exp.label_space)
                loss.backward()
                optimizer.step()
                self.backbone.merge_step()
            if scheduler is not None:
                scheduler.step()

        self._update_prototypes_and_stats(exp, loader, ctx)

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

    def _update_prototypes_and_stats(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        embeddings_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                features = self.backbone.forward_current(images)
                for feature, label in zip(features, targets.tolist()):
                    embeddings_by_class.setdefault(label, []).append(feature)

        for class_id, feats in embeddings_by_class.items():
            stacked = torch.stack(feats)
            self.head.set_prototype(class_id, stacked.mean(dim=0))
            update_class_stats(self._class_stats, class_id, stacked)

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
