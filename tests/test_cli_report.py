"""``clover report`` end-to-end via the CLI (SPEC §10-§11)."""

from __future__ import annotations

import csv

import yaml

from clover.cli import main


def _write_config(tmp_path, name, output_dir):
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


def test_report_aggregates_two_real_runs(tmp_path, capsys):
    output_dir = tmp_path / "runs"
    run_a = "synthetic__simplecil__disjoint_baseline__seed42"
    run_b = "synthetic__simplecil__disjoint_baseline__seed43"

    config_a = _write_config(tmp_path, run_a, output_dir)
    assert main(["run", config_a]) == 0

    config_b_path = tmp_path / f"{run_b}.yaml"
    with open(config_a) as fh:
        raw = yaml.safe_load(fh)
    raw["run"]["name"] = run_b
    raw["run"]["seed"] = 43
    with open(config_b_path, "w") as fh:
        yaml.safe_dump(raw, fh)
    assert main(["run", str(config_b_path)]) == 0

    out_dir = tmp_path / "results"
    exit_code = main(
        ["report", str(output_dir / run_a), str(output_dir / run_b), "--out", str(out_dir)]
    )
    assert exit_code == 0

    with open(out_dir / "all_runs_long.csv", newline="") as fh:
        long_rows = list(csv.DictReader(fh))
    run_ids = {row["run_id"] for row in long_rows}
    assert run_ids == {run_a, run_b}
    assert all(row["method"] == "simplecil" for row in long_rows)

    with open(out_dir / "summary.csv", newline="") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 1  # both runs share (method, scenario, dataset)
    assert summary_rows[0]["method"] == "simplecil"

    out = capsys.readouterr().out
    assert "2/2 run(s) done" in out


def test_report_excludes_a_run_that_never_finished(tmp_path):
    output_dir = tmp_path / "runs"
    run_a = "synthetic__simplecil__disjoint_baseline__seed42"
    config_a = _write_config(tmp_path, run_a, output_dir)
    assert main(["run", config_a]) == 0

    # A second run dir that only ever got as far as status.json="running".
    incomplete_dir = output_dir / "incomplete_run"
    incomplete_dir.mkdir()
    with open(incomplete_dir / "status.json", "w") as fh:
        import json

        json.dump({"state": "running", "last_completed_task": 0}, fh)

    out_dir = tmp_path / "results"
    main(["report", str(output_dir / run_a), str(incomplete_dir), "--out", str(out_dir)])

    with open(out_dir / "all_runs_long.csv", newline="") as fh:
        long_rows = list(csv.DictReader(fh))
    assert all(row["run_id"] == run_a for row in long_rows)
