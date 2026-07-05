"""``clover run`` end-to-end (SPEC §9, §11)."""

from __future__ import annotations

import os

import yaml

from clover.cli import main


def _write_config(tmp_path, output_dir, **stream_overrides):
    stream = {"dataset": "synthetic", "init_cls": 5, "increment": 5}
    stream.update(stream_overrides)
    config = {
        "run": {"name": "test_run", "seed": 42, "output_dir": str(output_dir)},
        "stream": stream,
        "method": {"name": "simplecil"},
        "training": {"epochs": 1, "batch_size": 16},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_run_writes_config_resolved_and_trainer_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, output_dir)

    exit_code = main(["run", config_path])
    assert exit_code == 0

    run_dir = output_dir / "test_run"
    files = os.listdir(run_dir)
    assert "config_resolved.yaml" in files
    assert "status.json" in files
    assert "R_matrix.npy" in files
    assert any(f.startswith("ckpt_task") for f in files)

    out = capsys.readouterr().out
    assert "run complete" in out


def test_run_smoke_profile_overrides_dataset_that_does_not_exist(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, output_dir, dataset="cifar100_not_built_yet")

    exit_code = main(["run", config_path, "--profile", "smoke"])
    assert exit_code == 0

    run_dir = output_dir / "test_run"
    with open(run_dir / "config_resolved.yaml") as fh:
        resolved = yaml.safe_load(fh)
    assert resolved["stream"]["dataset"] == "synthetic"


def test_run_with_typoed_key_fails_cleanly_before_any_run_dir(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, output_dir)
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    raw["stream"]["incrament"] = 5
    with open(config_path, "w") as fh:
        yaml.safe_dump(raw, fh)

    exit_code = main(["run", config_path])
    assert exit_code == 1
    assert not output_dir.exists()

    err = capsys.readouterr().err
    assert "incrament" in err
    assert "increment" in err
