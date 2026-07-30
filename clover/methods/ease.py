"""EASE (SPEC §6.5): a growing per-experience list of frozen adapters
(``clover/backbones/adapter_ease.py:EaseAdapterViT``), each contributing a
concatenated feature block to a growing-dim cosine head
(``clover/methods/heads.py:EaseHead``). Per-experience training uses a
temporary "proxy" head over just the current (single) adapter's features
and the shared ``new_class_ce`` loss policy -- P2's loss policy already
makes EASE's real ``aux_targets``/``ignore_index=-1`` patch unnecessary,
and matches what that patch's *effect* actually is: only new-class samples
contribute to the loss, revisited-class samples in the same batch are
skipped (not trained on), exactly like the pristine method's own
disjoint-task assumption.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from clover.backbones.loader import resolve_base_model
from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.methods.heads import EaseHead, IncrementalHead
from clover.methods.losses import new_class_ce


class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: EaseHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register_method("ease")
class EASE(CLMethod):
    name = "ease"
    default_backbone = "vit_adapter_ease"
    cacheable_features = False

    def __init__(self, alpha: float = 0.1) -> None:
        self.backbone: Any = None
        self.head: EaseHead | None = None
        self._proxy_head: IncrementalHead | None = None
        self.alpha = alpha

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        backbone_kwargs = {k: v for k, v in cfg.items() if k != "backbone"}
        backbone_kwargs.setdefault("input_size", stream_info.input_size)
        backbone_kwargs.setdefault("in_chans", stream_info.channels)
        self.backbone = resolve_base_model(cfg.get("backbone", self.default_backbone), **backbone_kwargs)
        feature_dim: int = self.backbone.feature_dim
        self.head = EaseHead(feature_dim, alpha=self.alpha)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        assert self.backbone is not None and self.head is not None
        self.backbone = self.backbone.to(ctx.device)
        self.backbone.grow()
        self.head = self.head.to(ctx.device)
        self.head.add_block()
        self.head.expand_classes(exp.label_space.head_size, home_block=self.head.num_blocks - 1)

        # Throwaway, like AdapterMethodBase's _train_head: sized to the
        # full global head so it works directly with new_class_ce; never
        # touched again after this experience, so it's excluded from
        # state_dict()/load_state_dict() -- replaying before_experience on
        # resume reconstructs an equivalent one from stream_info alone.
        self._proxy_head = IncrementalHead(self.backbone.feature_dim, cosine=False).to(ctx.device)
        self._proxy_head.expand_to(exp.label_space.head_size)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        assert self.backbone is not None and self._proxy_head is not None
        assert ctx.optimizer_factory is not None

        trainable = list(self.backbone.cur_adapter.parameters()) + list(self._proxy_head.parameters())
        optimizer = ctx.optimizer_factory(trainable)
        scheduler = ctx.make_scheduler(optimizer)
        for _epoch in range(ctx.epochs):
            for images, targets in loader:
                images = images.to(ctx.device)
                targets = targets.to(ctx.device)
                optimizer.zero_grad()
                logits = self._proxy_head(self.backbone.forward_current(images))
                loss = new_class_ce(logits, targets, exp.label_space)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        self._update_head(exp, loader, ctx)

    def _update_head(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        """Prototype = mean concatenated-feature embedding for every class
        seen in this experience's images, written into *every* block
        (real data + a still-available frozen old adapter for each old
        block, plus the just-trained new block). Classes not present this
        experience keep their existing rows in every already-existing
        block untouched; their row in the *new* block is filled by
        similarity interpolation (see ``_interpolate_old_classes``), since
        their images are no longer available to recompute a real one.
        """
        assert self.backbone is not None and self.head is not None
        embeddings_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                features = self.backbone(images)  # [batch, num_blocks, dim]
                for feature, label in zip(features, targets.tolist()):
                    embeddings_by_class.setdefault(label, []).append(feature)

        new_block = self.head.num_blocks - 1
        classes_this_round = list(embeddings_by_class.keys())
        for class_id, feats in embeddings_by_class.items():
            prototype = torch.stack(feats).mean(dim=0)  # [num_blocks, dim]
            for block in range(self.head.num_blocks):
                self.head.set_block_row(class_id, block, prototype[block])

        if new_block > 0:
            self._interpolate_old_classes(new_block, classes_this_round)

    def _interpolate_old_classes(self, new_block: int, classes_this_round: List[int]) -> None:
        """Fill in ``new_block``'s row for every class *not* seen this
        round via a cosine-similarity-weighted combination of this round's
        classes' own ``new_block`` rows -- a structurally faithful
        simplification of PILOT's ``solve_similarity``/``solve_sim_reset``
        (no old images available to recompute a real value)."""
        assert self.head is not None
        if not classes_this_round:
            return
        new_rows = torch.stack(
            [self.head.weight.data[c, new_block] for c in classes_this_round]
        )  # [n_new, dim]
        new_home_rows = torch.stack(
            [self.head.weight.data[c, self.head.home_block[c]] for c in classes_this_round]
        )  # [n_new, dim]

        for class_id in range(self.head.num_classes):
            if class_id in classes_this_round:
                continue
            home = self.head.home_block[class_id]
            old_home_row = self.head.weight.data[class_id, home]
            similarity = F.cosine_similarity(old_home_row.unsqueeze(0), new_home_rows, dim=-1)
            weights = F.softmax(similarity, dim=-1)
            interpolated = (weights.unsqueeze(-1) * new_rows).sum(dim=0)
            self.head.set_block_row(class_id, new_block, interpolated)

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
