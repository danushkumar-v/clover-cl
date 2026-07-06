"""``clover run-matrix`` end-to-end via the CLI (SPEC §11.1): real
subprocess dispatch, skip-on-done, and resume-after-interruption -- one
level up from ``tests/test_trainer_resume.py``'s single-run resume test.
"""

from __future__ import annotations

import json
import os

import yaml

from clover.cli import main


def _write_matrix(tmp_path, output_dir, seeds=(1, 2)):
    matrix = {
        "methods": ["simplecil"],
        "scenarios": ["disjoint_baseline"],
        "datasets": ["synthetic"],
        "seeds": list(seeds),
        "stream": {"init_cls": 5, "increment": 5},
        "training": {"epochs": 1, "batch_size": 16},
        "output_dir": str(output_dir),
    }
    path = tmp_path / "matrix.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(matrix, fh)
    return str(path)


def test_run_matrix_dispatches_every_cell(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    matrix_path = _write_matrix(tmp_path, output_dir)

    exit_code = main(["run-matrix", matrix_path, "--confirm"])
    assert exit_code == 0

    for seed in (1, 2):
        run_dir = output_dir / f"synthetic__simplecil__disjoint_baseline__seed{seed}"
        assert (run_dir / "status.json").exists()
        with open(run_dir / "status.json") as fh:
            assert json.load(fh)["state"] == "done"
        assert any(f.startswith("ckpt_task") for f in os.listdir(run_dir))

    out = capsys.readouterr().out
    assert "2 done, 0 failed, 0 skipped" in out


def test_run_matrix_skips_already_done_cells_on_resubmit(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    matrix_path = _write_matrix(tmp_path, output_dir, seeds=(1,))

    assert main(["run-matrix", matrix_path, "--confirm"]) == 0
    capsys.readouterr()

    assert main(["run-matrix", matrix_path, "--confirm"]) == 0
    out = capsys.readouterr().out
    assert "0 done, 0 failed, 1 skipped" in out


def test_run_matrix_resumes_an_interrupted_cell_rather_than_restarting(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    matrix_path = _write_matrix(tmp_path, output_dir, seeds=(1,))
    assert main(["run-matrix", matrix_path, "--confirm"]) == 0
    capsys.readouterr()

    run_dir = output_dir / "synthetic__simplecil__disjoint_baseline__seed1"
    checkpoints = sorted(f for f in os.listdir(run_dir) if f.startswith("ckpt_task"))
    os.remove(run_dir / checkpoints[-1])
    with open(run_dir / "status.json", "w") as fh:
        json.dump(
            {"state": "running", "last_completed_task": 2, "heartbeat": "2020-01-01T00:00:00+00:00"},
            fh,
        )

    exit_code = main(["run-matrix", matrix_path, "--confirm"])
    assert exit_code == 0

    with open(run_dir / "status.json") as fh:
        status = json.load(fh)
    assert status["state"] == "done"
    assert status["last_completed_task"] == 3
    assert (run_dir / checkpoints[-1]).exists()

    out = capsys.readouterr().out
    assert "1 done, 0 failed, 0 skipped" in out
