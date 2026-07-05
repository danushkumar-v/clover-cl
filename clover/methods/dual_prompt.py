"""DualPrompt (SPEC §6.5, method order item 2): a frozen ViT + general and
expert prefix prompts spliced into specific attention blocks, trained with
the core loss policies -- no hand-rolled logit masking.
"""

from __future__ import annotations

from clover.methods import register_method
from clover.methods.prompt_common import PromptMethodBase


@register_method("dualprompt")
class DualPrompt(PromptMethodBase):
    name = "dualprompt"
    default_backbone = "vit_dual_prompt"
