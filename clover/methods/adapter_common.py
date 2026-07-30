"""Shared "train adapter once, freeze forever" lifecycle (SPEC §6.5) for
APER-Adapter and RanPAC: both train an AdaptFormer bottleneck adapter only
during the *first* experience this method ever sees (``exp.task_label ==
0`` -- CLOVER's equivalent of PILOT's "cur_task == 0" gate, confirmed via
read-only research on ``models/aper_adapter.py``/``models/ranpac.py``: both
skip their gradient-training step entirely once past the first task), then
freeze the backbone permanently and rely entirely on a closed-form head
(no further gradient steps, ever) to absorb every later experience's new
classes. Subclasses provide the closed-form head and its per-experience
update; EASE/MOS/TUNA don't fit this base (they keep training a growing
per-task adapter list every experience) and get their own method modules.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn

from clover.backbones.loader import resolve_base_model
from clover.core.experience import Experience
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.methods.heads import IncrementalHead
from clover.methods.losses import new_class_ce


class AdapterMethodBase(CLMethod):
    #: The backbone still trains during the first experience -- only safe
    #: to treat as cacheable (declared, not guessed) once frozen, and this
    #: class attribute can't vary over a run's lifetime.
    cacheable_features = False

    def __init__(self) -> None:
        self.backbone: Optional[nn.Module] = None
        self._train_head: Optional[IncrementalHead] = None
        self._adapter_frozen = False

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        backbone_kwargs = {k: v for k, v in cfg.items() if k != "backbone"}
        backbone_kwargs.setdefault("input_size", stream_info.input_size)
        backbone_kwargs.setdefault("in_chans", stream_info.channels)
        self.backbone = resolve_base_model(cfg.get("backbone", self.default_backbone), **backbone_kwargs)
        feature_dim: int = self.backbone.feature_dim  # type: ignore[assignment]
        # Throwaway: only used to backprop through the adapter during the
        # first experience. Its trained values are never read afterward --
        # each subclass's real (closed-form) head takes over from
        # `_update_head` onward -- so it's deliberately excluded from
        # state_dict()/load_state_dict(); replaying before_experience(0) on
        # resume reconstructs it identically from stream_info alone.
        self._train_head = IncrementalHead(feature_dim, cosine=False)
        self._build_head(feature_dim)

    @abstractmethod
    def _build_head(self, adapter_feature_dim: int) -> None:
        """Construct the method's real (closed-form) head."""

    @abstractmethod
    def _update_head(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        """Recompute the closed-form head using this experience's data."""

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        assert self.backbone is not None and self._train_head is not None
        self.backbone = self.backbone.to(ctx.device)
        if exp.task_label == 0:
            self._train_head.expand_to(exp.label_space.head_size)
            self._train_head = self._train_head.to(ctx.device)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None
        if exp.task_label == 0:
            self._train_adapter(exp, loader, ctx)
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            self._adapter_frozen = True
        self._update_head(exp, loader, ctx)

    def _train_adapter(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self._train_head is not None
        assert ctx.optimizer_factory is not None
        trainable = [p for p in self.backbone.parameters() if p.requires_grad] + list(
            self._train_head.parameters()
        )
        optimizer = ctx.optimizer_factory(trainable)
        scheduler = ctx.make_scheduler(optimizer)
        for _epoch in range(ctx.epochs):
            for images, targets in loader:
                images = images.to(ctx.device)
                targets = targets.to(ctx.device)
                optimizer.zero_grad()
                logits = self._train_head(self.backbone(images))
                # Experience 0 is always all-new-class by construction (no
                # revisit can predate the first experience), but this still
                # goes through the shared loss policy rather than a plain
                # CE call -- no method hand-rolls its own masking.
                loss = new_class_ce(logits, targets, exp.label_space)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def _base_state(self) -> Dict[str, Any]:
        assert self.backbone is not None
        return {"backbone": self.backbone.state_dict(), "adapter_frozen": self._adapter_frozen}

    def _load_base_state(self, state: Dict[str, Any]) -> None:
        assert self.backbone is not None
        self.backbone.load_state_dict(state["backbone"])
        self._adapter_frozen = state["adapter_frozen"]
        if self._adapter_frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
