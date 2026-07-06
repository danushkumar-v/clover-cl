"""APER-Adapter (SPEC §6.5): a dual-branch NCM classifier over a frozen
plain ViT pass concatenated with a frozen adapter-tuned pass through the
*same* base weights, adapter trained only during the first experience then
frozen forever. Clean-room reimplementation (no code copied from
LAMDA-PILOT) of the "PEFT-then-freeze, prototype head" recipe confirmed via
read-only research on ``models/aper_adapter.py``/``utils/inc_net.py``
(``MultiBranchCosineIncrementalNet``): PILOT loads two separately-checkpointed
backbones (plain + adapter-tuned); here, since ``AdapterViT.base`` freezes
its weights before any adapter training happens, a plain pass through
``backbone.base`` and an adapter pass through ``backbone`` read the exact
same frozen weights without needing a second model instance.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.adapter_common import AdapterMethodBase
from clover.methods.base import TrainContext
from clover.methods.heads import IncrementalHead


class _Classifier(nn.Module):
    """``images -> logits`` over the concatenation of a plain pass through
    the frozen base and an adapter-tuned pass through the same base."""

    def __init__(self, backbone: nn.Module, head: IncrementalHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        plain = self.backbone.base(x)  # type: ignore[operator]
        adapted = self.backbone(x)
        return torch.cat([plain, adapted], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._features(x))


@register_method("aper_adapter")
class APERAdapter(AdapterMethodBase):
    name = "aper_adapter"
    default_backbone = "vit_adapter"

    def __init__(self) -> None:
        super().__init__()
        self.head: IncrementalHead | None = None

    def _build_head(self, adapter_feature_dim: int) -> None:
        # Dual-branch concat doubles the feature width fed to the head.
        self.head = IncrementalHead(adapter_feature_dim * 2, cosine=True)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        super().before_experience(exp, ctx)
        assert self.head is not None
        self.head.expand_to(exp.label_space.head_size)
        self.head = self.head.to(ctx.device)

    def _update_head(self, exp: Experience, loader: Any, ctx: TrainContext) -> None:
        """Prototype = mean concatenated-feature embedding, recomputed from
        this experience's images alone (SimpleCIL's pattern: no
        cross-experience accumulation, since the head is never fine-tuned
        after being set either way)."""
        assert self.backbone is not None and self.head is not None
        embeddings_by_class: Dict[int, List[torch.Tensor]] = {}
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(ctx.device)
                plain = self.backbone.base(images)  # type: ignore[operator]
                adapted = self.backbone(images)
                features = torch.cat([plain, adapted], dim=-1)
                for feature, label in zip(features, targets.tolist()):
                    embeddings_by_class.setdefault(label, []).append(feature)

        for class_id, feats in embeddings_by_class.items():
            prototype = torch.stack(feats).mean(dim=0)
            self.head.set_prototype(class_id, prototype)

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
