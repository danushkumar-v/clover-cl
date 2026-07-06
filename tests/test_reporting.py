"""``clover report`` aggregation correctness (SPEC §10-§11): rebuild-from
-scratch long CSV + summary pivot, hand-crafted per-run fixtures."""

from __future__ import annotations

import csv
import json
import math
import os

from clover.reporting import (
    build_summary,
    parse_run_id,
    rebuild_long_csv,
    write_long_csv,
    write_summary_csv,
)


def test_parse_run_id_valid():
    assert parse_run_id("cifar100__l2p__exact_replay__seed1993") == (
        "cifar100",
        "l2p",
        "exact_replay",
        "1993",
    )


def test_parse_run_id_rejects_wrong_shape():
    assert parse_run_id("my_custom_run_name") is None
    assert parse_run_id("a__b__c__notseed1") is None


def _write_run(tmp_path, name, state, rows):
    run_dir = tmp_path / name
    run_dir.mkdir()
    with open(run_dir / "status.json", "w") as fh:
        json.dump({"state": state, "last_completed_task": 0}, fh)
    if rows is not None:
        with open(run_dir / "per_task.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["task_idx", "metric", "value"])
            writer.writerows(rows)
    return str(run_dir)


def test_rebuild_long_csv_skips_non_done_runs(tmp_path):
    done_dir = _write_run(tmp_path, "synthetic__simplecil__disjoint_baseline__seed42", "done", [(0, "A_t", 0.9)])
    failed_dir = _write_run(tmp_path, "synthetic__l2p__disjoint_baseline__seed42", "failed", [(0, "A_t", 0.5)])

    rows = rebuild_long_csv([done_dir, failed_dir])
    assert len(rows) == 1
    assert rows[0]["method"] == "simplecil"
    assert rows[0]["value"] == "0.9"


def test_rebuild_long_csv_tags_unparseable_dir_as_unknown_but_keeps_rows(tmp_path):
    run_dir = _write_run(tmp_path, "my_custom_run", "done", [(0, "A_t", 0.9)])
    rows = rebuild_long_csv([run_dir])
    assert len(rows) == 1
    assert rows[0]["dataset"] == "unknown"
    assert rows[0]["method"] == "unknown"


def test_write_long_csv_round_trip(tmp_path):
    row = {
        "run_id": "r",
        "dataset": "synthetic",
        "method": "simplecil",
        "scenario": "disjoint_baseline",
        "seed": "42",
        "task_idx": "0",
        "metric": "A_t",
        "value": "0.9",
    }
    out_path = str(tmp_path / "out" / "all_runs_long.csv")
    write_long_csv([row], out_path)
    assert os.path.exists(out_path)
    with open(out_path, newline="") as fh:
        read_back = list(csv.DictReader(fh))
    assert read_back == [row]


def test_build_summary_averages_over_seeds_using_last_task_value():
    rows = [
        {"run_id": "r1", "dataset": "d", "method": "m", "scenario": "s", "seed": "1", "task_idx": "0", "metric": "AIA", "value": "0.8"},
        {"run_id": "r1", "dataset": "d", "method": "m", "scenario": "s", "seed": "1", "task_idx": "1", "metric": "AIA", "value": "0.9"},
        {"run_id": "r2", "dataset": "d", "method": "m", "scenario": "s", "seed": "2", "task_idx": "0", "metric": "AIA", "value": "0.7"},
    ]
    summary = build_summary(rows)
    assert math.isclose(summary[("m", "s", "d")]["AIA"], (0.9 + 0.7) / 2)


def test_build_summary_excludes_nan_values_from_average():
    rows = [
        {"run_id": "r1", "dataset": "d", "method": "m", "scenario": "s", "seed": "1", "task_idx": "0", "metric": "BWT", "value": "nan"},
        {"run_id": "r2", "dataset": "d", "method": "m", "scenario": "s", "seed": "2", "task_idx": "0", "metric": "BWT", "value": "0.5"},
    ]
    summary = build_summary(rows)
    assert math.isclose(summary[("m", "s", "d")]["BWT"], 0.5)


def test_write_summary_csv_writes_expected_columns(tmp_path):
    summary = {("m", "s", "d"): {"AIA": 0.85, "BWT": -0.1}}
    out_path = str(tmp_path / "summary.csv")
    write_summary_csv(summary, out_path)
    with open(out_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["method"] == "m"
    assert rows[0]["scenario"] == "s"
    assert rows[0]["dataset"] == "d"
    assert math.isclose(float(rows[0]["AIA"]), 0.85)
    assert math.isclose(float(rows[0]["BWT"]), -0.1)
