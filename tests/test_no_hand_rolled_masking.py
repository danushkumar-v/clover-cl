"""Lint: no method hand-rolls -inf logit masking (SPEC §5.2).

The core loss policies (``clover/methods/losses.py``) are the only
sanctioned place ``-inf`` masking happens; a method writing its own is a
bug even if its tests pass -- this is exactly the v1 bug SPEC §5 exists to
prevent (SPEC §5.2: "Methods never write logits[:, :k] = -inf by hand --
enforced by convention, code review, and a grep-based lint test in CI").
"""

from __future__ import annotations

from pathlib import Path

METHODS_DIR = Path(__file__).parent.parent / "clover" / "methods"
_EXEMPT = {"losses.py"}


def test_no_method_module_hand_rolls_inf_masking():
    offenders = []
    for path in sorted(METHODS_DIR.glob("*.py")):
        if path.name in _EXEMPT:
            continue
        text = path.read_text()
        if "-inf" in text or "-math.inf" in text:
            offenders.append(path.name)
    assert not offenders, (
        f"hand-rolled -inf masking found outside methods/losses.py: {offenders}. "
        "Use new_class_ce/seen_class_ce/masked_logits instead (SPEC §5.2)."
    )


def test_losses_module_is_the_only_place_using_inf():
    # sanity: confirm the exemption is meaningful (losses.py does use -inf).
    losses_text = (METHODS_DIR / "losses.py").read_text()
    assert "-inf" in losses_text
