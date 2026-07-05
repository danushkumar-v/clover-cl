"""L2P (SPEC §6.5, method order item 2): a frozen ViT + a learnable prompt
pool, trained with the core loss policies -- no hand-rolled logit masking.
"""

from __future__ import annotations

from clover.methods import register_method
from clover.methods.prompt_common import PromptMethodBase


@register_method("l2p")
class L2P(PromptMethodBase):
    name = "l2p"
    default_backbone = "vit_prompt_pool"
