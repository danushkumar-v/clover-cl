"""P10: backbone real-image support (PLAN.md, added 2026-07-07).

Real (3-channel) datasets no longer crash on the first training batch --
channel-count threading now flows ``CLDataset.channels`` ->
``StreamInfo.channels`` -> every method's ``build()`` ->
``resolve_base_model`` -> ``TinyViT``/``TinyMLP``/a real timm model.

This work also surfaced a real, previously invisible bug: ``Trainer.run()``
hardcoded ``self.method.build(stream_info, {})`` -- so a config's
``method:`` extra keys (e.g. ``backbone:``) never reached a method through
a real ``clover run``, only through hand-constructed unit tests that called
``method.build(info, cfg)`` directly. The end-to-end test below is the
first thing that actually exercises the real config -> Trainer -> method
path with a non-default backbone.

P10 left one gap open, closed since by P11-A: the prompt/adapter wrapper
mechanisms (``vit_prompt_pool``, ``vit_dual_prompt``, ``vit_coda_prompt``,
``vit_adapter*``) require their ``base_model`` to implement CLOVER-specific
hooks (``query_features``, ``forward_tokens``/``forward(x, adapter=...)``)
that a raw timm ViT doesn't have, so ``resolve_base_model`` would happily
*construct* a real timm model as a wrapper's base and then fail
(``AttributeError``) on the wrapper's own forward pass. ``resolve_base_model``
now wraps a timm ViT in ``clover/backbones/timm_vit.py:TimmViTHooks``, which
supplies those hooks over timm's own submodules -- see
``tests/test_timm_vit_hooks.py``.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml
from PIL import Image

from clover.backbones.loader import resolve_base_model
from clover.cli import main


def _make_fixture(root, classes_and_counts):
    for split in ("train", "test"):
        for class_name, count in classes_and_counts.items():
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(count):
                img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
                img.save(class_dir / f"{i}.png")


def _write_config(tmp_path, fixture_root, method_cfg):
    config = {
        "run": {"output_dir": str(tmp_path / "runs")},
        "stream": {
            "dataset": {"type": "image_folder", "root": str(fixture_root), "num_classes": 2},
            "init_cls": 1,
            "increment": 1,
        },
        "method": method_cfg,
        "training": {"batch_size": 2, "epochs": 1},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_resolve_base_model_timm_fallback_constructs_and_runs_forward():
    model = resolve_base_model("vit_tiny_patch16_224", source="auto", in_chans=3)
    # `.feature_dim` is the project-wide name (TinyViT/TinyMLP/every wrapper
    # mechanism expose it); timm calls the same number `.num_features`. Since
    # P11-A a ViT comes back inside TimmViTHooks, so the timm-side name lives
    # on the wrapped base -- the two must still agree.
    assert model.feature_dim == model.base.num_features
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, model.feature_dim)
    assert torch.isfinite(out).all()


def test_real_dataset_trains_one_experience_with_a_real_timm_backbone(tmp_path, capsys):
    """The literal P10 DONE gate: a real (non-synthetic) dataset trains one
    experience without a shape/channel error, on CPU, with a real timm
    backbone selected purely via config -- proving channel threading,
    resolve_base_model's timm fallback, and the method_cfg fix all work
    together through the actual clover run path, not a hand-wired test."""
    fixture_root = tmp_path / "fixture"
    _make_fixture(fixture_root, {"class_a": 3, "class_b": 3})
    config_path = _write_config(
        tmp_path, fixture_root, {"name": "simplecil", "backbone": "vit_tiny_patch16_224"}
    )

    exit_code = main(["run", config_path])
    assert exit_code == 0, capsys.readouterr().err
    assert "run complete" in capsys.readouterr().out


def test_real_dataset_trains_one_experience_with_l2ps_default_backbone(tmp_path, capsys):
    """Same gate for a gradient-trained, wrapper-mechanism method using its
    own default (tiny_vit-based) backbone -- proves channel threading works
    for the prompt/adapter family too, without claiming real-timm
    compatibility for them (see module docstring)."""
    fixture_root = tmp_path / "fixture"
    _make_fixture(fixture_root, {"class_a": 3, "class_b": 3})
    config_path = _write_config(tmp_path, fixture_root, {"name": "l2p"})

    exit_code = main(["run", config_path])
    assert exit_code == 0, capsys.readouterr().err
    assert "run complete" in capsys.readouterr().out
