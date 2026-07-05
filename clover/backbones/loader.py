"""Base-model resolution (SPEC §6.4): timm/HF name, a user class, or a
built-in test backbone. The one place base-model resolution happens --
wrapper mechanisms (``prompt_pool.py`` etc.) never import timm directly.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch.nn as nn

from clover.backbones import get_backbone, list_backbones


def resolve_base_model(name: str, source: str = "auto", **kwargs: Any) -> nn.Module:
    """Resolve a base model by name.

    Args:
        name: A registered backbone key (e.g. ``"tiny_vit"``), a timm model
            name, or (when ``source="class"``) ``"module.path:ClassName"``.
        source: ``"auto"`` (default) checks the backbone registry first,
            falling back to timm; ``"timm"``/``"class"`` force that source.
        **kwargs: Forwarded to the model constructor.
    """
    if source == "class":
        module_name, _, class_name = name.partition(":")
        if not class_name:
            raise ValueError(
                f"source='class' name must be 'module.path:ClassName', got {name!r}."
            )
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(**kwargs)

    if source not in ("auto", "timm"):
        raise ValueError(f"source must be 'auto', 'timm', or 'class', got {source!r}.")

    if source == "auto" and name in list_backbones():
        return get_backbone(name)(**kwargs)

    import timm

    return timm.create_model(name, pretrained=False, num_classes=0, **kwargs)
