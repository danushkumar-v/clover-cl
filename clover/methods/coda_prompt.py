"""CODA-Prompt (SPEC §6.5, method order item 2): a frozen ViT + a
soft-attention-combined prompt pool spliced as prefix K/V, with
Gram-Schmidt orthogonalization of each task's newly unlocked pool slots.
Trained with the core loss policies -- no hand-rolled logit masking.
"""

from __future__ import annotations

from typing import Any, Dict

from clover.core.experience import Experience
from clover.methods import register_method
from clover.methods.base import StreamInfo, TrainContext
from clover.methods.prompt_common import PromptMethodBase


@register_method("coda_prompt")
class CODAPrompt(PromptMethodBase):
    name = "coda_prompt"
    default_backbone = "vit_coda_prompt"

    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        cfg = dict(cfg)
        cfg.setdefault("nb_experiences", stream_info.nb_experiences)
        super().build(stream_info, cfg)

    def before_experience(self, exp: Experience, ctx: TrainContext) -> None:
        super().before_experience(exp, ctx)
        assert self.backbone is not None
        self.backbone.pool.start_new_task(exp.task_label)  # type: ignore[union-attr,operator]
