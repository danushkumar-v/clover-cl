"""SimpleCIL: frozen backbone + cosine prototype head (SPEC §6.5, smallest method).

Clean-room reimplementation of the "frozen backbone, closed-form
nearest-prototype classifier" design (no code copied from LAMDA-PILOT):
the backbone is never trained; each class's prototype is the mean embedding
of that class's images in the experience it's computed from, assigned
directly into the cosine head's weight row. Validates the Trainer, the
per-class evaluator, and incremental heads end-to-end -- the smallest
possible real method.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from clover.backbones import get_backbone
from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.methods.heads import IncrementalHead


class _Classifier(nn.Module):
    """``images -> logits``, for the shared evaluator (SPEC §6.1)."""

    def __init__(self, backbone: nn.Module, head: IncrementalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register_method("simplecil")
class SimpleCIL(CLMethod):
    name = "simplecil"
    default_backbone = "tiny_mlp"
    #: Backbone is frozen -- its embeddings never change, so they're safe
    #: to cache (declared, not guessed -- bench lesson).
    cacheable_features = True

    def __init__(self) -> None:
        self.backbone: nn.Module | None = None
        self.head: IncrementalHead | None = None

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        backbone_cls = get_backbone(cfg.get("backbone", self.default_backbone))
        self.backbone = backbone_cls(input_size=stream_info.input_size)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.head = IncrementalHead(self.backbone.feature_dim, cosine=True)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        assert self.head is not None
        self.head.expand_to(exp.label_space.head_size)
        self.backbone = self.backbone.to(ctx.device) if self.backbone is not None else None
        self.head = self.head.to(ctx.device)

    def train_experience(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        """No gradient step: one pass over the loader, prototype = mean embedding.

        Every class present this experience (both first appearances and
        revisits) gets its prototype recomputed from *this* experience's
        images -- SimpleCIL never fine-tunes a prototype across experiences,
        so there is nothing to accumulate for a revisit either.
        """
        assert self.backbone is not None and self.head is not None
        embeddings_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                features = self.backbone(images)
                for feature, label in zip(features, targets.tolist()):
                    embeddings_by_class.setdefault(label, []).append(feature)

        for class_id, features in embeddings_by_class.items():
            prototype = torch.stack(features).mean(dim=0)
            self.head.set_prototype(class_id, prototype)

    def classifier(self) -> nn.Module:
        assert self.backbone is not None and self.head is not None
        return _Classifier(self.backbone, self.head)

    def state_dict(self) -> Dict[str, Any]:
        assert self.head is not None
        return {"head": self.head.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        assert self.head is not None
        self.head.load_state_dict(state["head"])
