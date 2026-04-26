"""Tests for clover/cli/__main__.py."""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture
def cifar_partial_yaml(tmp_path):
    """Write a minimal cifar100 partial-overlap YAML to a temp file."""
    content = """dataset_name: cifar100
init_cls: 10
increment: 10
shuffle_seed: 1993
data_root: ./data

overlap_spec:
  mode: partial
  seed: 42
  image_split:
    strategy: duplicate
    ratio: 0.5
    overlap_pct: 0.0
  pairs:
    - tasks: [0, 5]
      shared_classes: [0, 1, 2, 3, 4]
"""
    p = tmp_path / "test_partial.yaml"
    p.write_text(content)
    return str(p)


def test_inspect_command(patched_cifar100, cifar_partial_yaml, tmp_path, monkeypatch, capsys):
    """inspect command should print matrix and save plots."""
    import clover.cli.__main__ as cli_mod

    # Run in a temp dir so inspect_output is isolated
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["clover", "inspect", cifar_partial_yaml])
    cli_mod.main()

    captured = capsys.readouterr()
    assert "cifar100" in captured.out.lower()
    assert "T00" in captured.out or "T0" in captured.out

    out_dir = tmp_path / "inspect_output"
    pngs = list(out_dir.glob("*.png"))
    assert len(pngs) >= 2, f"Expected at least 2 PNG files, got: {[p.name for p in pngs]}"


def test_no_args_prints_usage(capsys, monkeypatch):
    import clover.cli.__main__ as cli_mod

    monkeypatch.setattr(sys, "argv", ["clover"])
    with pytest.raises(SystemExit):
        cli_mod.main()
    captured = capsys.readouterr()
    assert "Usage" in captured.out


def test_unknown_command_exits(capsys, monkeypatch):
    import clover.cli.__main__ as cli_mod

    monkeypatch.setattr(sys, "argv", ["clover", "frobnicate"])
    with pytest.raises(SystemExit):
        cli_mod.main()


def test_inspect_missing_config_exits(capsys, monkeypatch):
    import clover.cli.__main__ as cli_mod

    monkeypatch.setattr(sys, "argv", ["clover", "inspect"])
    with pytest.raises(SystemExit):
        cli_mod.main()
