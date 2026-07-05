"""Revisit-safe loss policies (SPEC §5.2): the only place loss masking lives.

``new_class_ce``, ``seen_class_ce``, ``masked_logits`` are implemented in P2,
built on ``LabelSpaceView`` (``clover/core/experience.py``) and set-membership
masks — never range arithmetic (``targets >= known_classes``,
``logits[:, :k] = -inf``). No method module may hand-roll its own masking.
"""

from __future__ import annotations
