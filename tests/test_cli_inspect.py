"""``clover inspect`` -- prints a plan summary, no training (SPEC §9)."""

from __future__ import annotations

import os

import yaml

from clover.cli import main


def _write_config(tmp_path, **stream_overrides):
    stream = {"dataset": "synthetic", "init_cls": 5, "increment": 5, "scenario": "exact_replay"}
    stream.update(stream_overrides)
    config = {
        "run": {"output_dir": str(tmp_path / "runs")},
        "stream": stream,
        "method": {"name": "simplecil"},
        "training": {},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_inspect_prints_plan_summary(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    exit_code = main(["inspect", config_path])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "dataset: synthetic (20 classes)" in out
    assert "experiences: 4" in out
    assert "head size schedule:" in out


def test_inspect_creates_no_run_directory_or_artifacts(tmp_path):
    config_path = _write_config(tmp_path)
    main(["inspect", config_path])
    assert not (tmp_path / "runs").exists()
    # inspect must not touch the filesystem beyond reading the config itself
    assert os.listdir(tmp_path) == ["config.yaml"]
