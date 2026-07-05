"""Shared scaffolding for prompt-family methods (SPEC §6.5): L2P, DualPrompt,
CODA-Prompt all wrap a frozen base ViT with a small set of trainable
prompt/key/head params and one gradient loop using the core loss policies.
Only ``default_backbone`` (and each backbone's own hyperparameters) differ
between them -- everything else is identical, so it lives here once.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from clover.backbones import get_backbone
from clover.core.experience import Experience
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.methods.heads import IncrementalHead
from clover.methods.losses import new_class_ce, seen_class_ce


class _Classifier(nn.Module):
    """``images -> logits``, for the shared evaluator (SPEC §6.1)."""

    def __init__(self, backbone: nn.Module, head: IncrementalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class PromptMethodBase(CLMethod):
    """Frozen backbone (a prompt-wrapped ViT) + an incremental head, trained
    end-to-end each experience with ``new_class_ce + seen_class_ce`` -- no
    hand-rolled logit masking. Subclasses set ``name``/``default_backbone``.
    """

    #: The prompt/pool mechanism is a function of what's currently learned
    #: (and, for L2P/DualPrompt, of the *input* via query-based selection)
    #: -- not safe to cache across experiences.
    cacheable_features = False

    def __init__(self) -> None:
        self.backbone: Optional[nn.Module] = None
        self.head: Optional[IncrementalHead] = None

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        backbone_cls = get_backbone(cfg.get("backbone", self.default_backbone))
        backbone_kwargs = {k: v for k, v in cfg.items() if k != "backbone"}
        backbone_kwargs.setdefault("input_size", stream_info.input_size)
        backbone: nn.Module = backbone_cls(**backbone_kwargs)
        self.backbone = backbone
        feature_dim: int = backbone.feature_dim  # type: ignore[assignment]
        self.head = IncrementalHead(feature_dim, cosine=False)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        assert self.head is not None and self.backbone is not None
        self.head.expand_to(exp.label_space.head_size)
        self.backbone = self.backbone.to(ctx.device)
        self.head = self.head.to(ctx.device)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        assert ctx.optimizer_factory is not None

        trainable_params: List[nn.Parameter] = [
            p for p in self.backbone.parameters() if p.requires_grad
        ] + list(self.head.parameters())
        optimizer = ctx.optimizer_factory(trainable_params)

        for _epoch in range(ctx.epochs):
            for images, targets in loader:
                images = images.to(ctx.device)
                targets = targets.to(ctx.device)

                optimizer.zero_grad()
                logits = self.head(self.backbone(images))
                loss = new_class_ce(logits, targets, exp.label_space) + seen_class_ce(
                    logits, targets, exp.label_space
                )
                loss.backward()
                optimizer.step()

    def classifier(self) -> nn.Module:
        assert self.backbone is not None and self.head is not None
        return _Classifier(self.backbone, self.head)

    def state_dict(self) -> Dict[str, Any]:
        assert self.backbone is not None and self.head is not None
        return {"backbone": self.backbone.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        assert self.backbone is not None and self.head is not None
        self.backbone.load_state_dict(state["backbone"])
        self.head.load_state_dict(state["head"])
