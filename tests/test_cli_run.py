"""``clover run`` end-to-end (SPEC §9, §11)."""

from __future__ import annotations

import json
import logging
import os
from unittest.mock import patch

import yaml

from clover.cli import _build_run_config, main
from clover.config import resolve_config


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
    assert "manifest.json" in files
    assert "log.txt" in files
    assert any(f.startswith("ckpt_task") for f in files)

    out = capsys.readouterr().out
    assert "run complete" in out


def test_run_writes_a_valid_manifest(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, output_dir)
    main(["run", config_path])

    with open(output_dir / "test_run" / "manifest.json") as fh:
        manifest = json.load(fh)
    assert manifest["_header"]["dataset"] == "synthetic"
    assert manifest["_header"]["init_cls"] == 5
    assert manifest["num_classes"] == 20
    assert len(manifest["task_class_lists"]) == 4


def test_run_writes_log_txt_with_lifecycle_messages(tmp_path):
    output_dir = tmp_path / "runs"
    config_path = _write_config(tmp_path, output_dir)
    main(["run", config_path])

    with open(output_dir / "test_run" / "log.txt") as fh:
        contents = fh.read()
    assert "run starting" in contents
    assert "experience 0 done" in contents
    assert "run complete" in contents


def _write_named_config(tmp_path, name, output_dir):
    config = {
        "run": {"name": name, "seed": 42, "output_dir": str(output_dir)},
        "stream": {"dataset": "synthetic", "init_cls": 5, "increment": 5},
        "method": {"name": "simplecil"},
        "training": {"epochs": 1, "batch_size": 16},
    }
    path = tmp_path / f"{name}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh)
    return str(path)


def test_run_log_handler_does_not_leak_across_separate_runs(tmp_path):
    """Two runs in the same process must each get their own log.txt, not
    accumulate handlers that write into every prior run's file too."""
    output_dir = tmp_path / "runs"
    config_a = _write_named_config(tmp_path, "run_a", output_dir)
    config_b = _write_named_config(tmp_path, "run_b", output_dir)

    main(["run", config_a])
    first_log = output_dir / "run_a" / "log.txt"
    with open(first_log) as fh:
        first_contents_after_first_run = fh.read()

    main(["run", config_b])
    with open(first_log) as fh:
        first_contents_after_second_run = fh.read()

    # the first run's log.txt must be unchanged by the second run
    assert first_contents_after_first_run == first_contents_after_second_run
    assert "run complete" in first_contents_after_first_run
    logger = logging.getLogger("clover.training")
    assert len(logger.handlers) == 1


def test_build_run_config_auto_detects_cuda_when_available(tmp_path):
    raw = {
        "stream": {"dataset": "synthetic", "init_cls": 5, "increment": 5},
        "method": {"name": "simplecil"},
    }
    resolved = resolve_config(raw)
    with patch("torch.cuda.is_available", return_value=True):
        run_config = _build_run_config(resolved, str(tmp_path), smoke=False)
    assert run_config.device == "cuda"


def test_build_run_config_stays_cpu_without_cuda(tmp_path):
    raw = {
        "stream": {"dataset": "synthetic", "init_cls": 5, "increment": 5},
        "method": {"name": "simplecil"},
    }
    resolved = resolve_config(raw)
    with patch("torch.cuda.is_available", return_value=False):
        run_config = _build_run_config(resolved, str(tmp_path), smoke=False)
    assert run_config.device == "cpu"


def test_build_run_config_forces_cpu_for_smoke_even_with_cuda_available(tmp_path):
    raw = {
        "stream": {"dataset": "synthetic", "init_cls": 5, "increment": 5},
        "method": {"name": "simplecil"},
    }
    resolved = resolve_config(raw)
    with patch("torch.cuda.is_available", return_value=True):
        run_config = _build_run_config(resolved, str(tmp_path), smoke=True)
    assert run_config.device == "cpu"


def test_build_run_config_threads_cudnn_benchmark_from_training_section(tmp_path):
    raw = {
        "stream": {"dataset": "synthetic", "init_cls": 5, "increment": 5},
        "method": {"name": "simplecil"},
        "training": {"cudnn_benchmark": True},
    }
    resolved = resolve_config(raw)
    run_config = _build_run_config(resolved, str(tmp_path), smoke=False)
    assert run_config.cudnn_benchmark is True


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
