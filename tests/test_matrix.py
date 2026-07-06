"""``run-matrix`` orchestrator correctness (SPEC §11.1): grid enumeration,
status/staleness classification, and dispatch planning -- all pure,
hand-crafted-fixture-testable, no real subprocesses needed here (see
``tests/test_cli_run_matrix.py`` for one real end-to-end subprocess test).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from clover.config.schema import MatrixSection
from clover.matrix import (
    STALE_HEARTBEAT_SECONDS,
    cell_config,
    enumerate_cells,
    is_stale,
    plan_dispatch,
    read_status,
    run_dir_for,
)


def _matrix(**overrides):
    defaults = dict(
        methods=["simplecil"],
        scenarios=["disjoint_baseline"],
        datasets=["synthetic"],
        seeds=[42],
    )
    defaults.update(overrides)
    return MatrixSection.from_dict(defaults)


def test_enumerate_cells_is_the_full_cross_product():
    matrix = _matrix(methods=["a", "b"], scenarios=["s1"], datasets=["d1", "d2"], seeds=[1, 2])
    cells = enumerate_cells(matrix)
    assert len(cells) == 2 * 1 * 2 * 2
    run_ids = {c.run_id for c in cells}
    assert "d1__a__s1__seed1" in run_ids
    assert "d2__b__s1__seed2" in run_ids


def test_cell_config_fills_in_dataset_scenario_seed():
    matrix = _matrix(stream={"init_cls": 5, "increment": 5}, training={"epochs": 3})
    cell = enumerate_cells(matrix)[0]
    cfg = cell_config(matrix, cell)
    assert cfg["stream"]["dataset"] == "synthetic"
    assert cfg["stream"]["scenario"] == "disjoint_baseline"
    assert cfg["stream"]["stream_seed"] == 42
    assert cfg["stream"]["init_cls"] == 5
    assert cfg["method"]["name"] == "simplecil"
    assert cfg["training"]["epochs"] == 3
    assert cfg["run"]["name"] == cell.run_id


def test_cell_config_applies_method_overrides():
    matrix = _matrix(methods=["l2p"], method_overrides={"l2p": {"prompt_pool_size": 7}})
    cell = enumerate_cells(matrix)[0]
    cfg = cell_config(matrix, cell)
    assert cfg["method"] == {"name": "l2p", "prompt_pool_size": 7}


def test_read_status_missing_file_is_pending(tmp_path):
    assert read_status(str(tmp_path / "nonexistent")) == {"state": "pending"}


def test_read_status_reads_existing_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "status.json", "w") as fh:
        json.dump({"state": "done", "last_completed_task": 4}, fh)
    assert read_status(str(run_dir))["state"] == "done"


def test_is_stale_false_for_non_running_states():
    now = datetime.now(timezone.utc)
    assert not is_stale({"state": "done"}, now)
    assert not is_stale({"state": "pending"}, now)
    assert not is_stale({"state": "failed"}, now)


def test_is_stale_false_for_fresh_heartbeat():
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(minutes=1)).isoformat()
    assert not is_stale({"state": "running", "heartbeat": fresh}, now)


def test_is_stale_true_for_old_heartbeat():
    now = datetime.now(timezone.utc)
    old = (now - timedelta(seconds=STALE_HEARTBEAT_SECONDS + 60)).isoformat()
    assert is_stale({"state": "running", "heartbeat": old}, now)


def test_is_stale_true_for_missing_or_unparseable_heartbeat():
    now = datetime.now(timezone.utc)
    assert is_stale({"state": "running"}, now)
    assert is_stale({"state": "running", "heartbeat": "not-a-date"}, now)


def test_plan_dispatch_skips_done_dispatches_everything_else(tmp_path):
    matrix = _matrix(
        methods=["a", "b", "c", "d"], scenarios=["s"], datasets=["synthetic"], seeds=[1], output_dir=str(tmp_path)
    )
    cells = enumerate_cells(matrix)  # a, b, c, d -- one per state below
    now = datetime.now(timezone.utc)

    _write_status(run_dir_for(matrix, cells[0]), {"state": "done", "last_completed_task": 0})
    _write_status(run_dir_for(matrix, cells[1]), {"state": "failed", "error": "boom"})
    # cells[2] has no status.json at all (pending)
    _write_status(
        run_dir_for(matrix, cells[3]),
        {"state": "running", "heartbeat": (now - timedelta(hours=1)).isoformat()},
    )

    to_run, skipped = plan_dispatch(matrix, cells, now)
    assert skipped == [cells[0]]
    assert set(to_run) == {cells[1], cells[2], cells[3]}


def test_plan_dispatch_relabels_stale_running_as_failed_before_retry(tmp_path):
    matrix = _matrix(output_dir=str(tmp_path))
    cell = enumerate_cells(matrix)[0]
    now = datetime.now(timezone.utc)
    run_dir = run_dir_for(matrix, cell)
    _write_status(run_dir, {"state": "running", "heartbeat": (now - timedelta(hours=1)).isoformat()})

    plan_dispatch(matrix, [cell], now)

    with open(os.path.join(run_dir, "status.json")) as fh:
        relabeled = json.load(fh)
    assert relabeled["state"] == "failed"
    assert relabeled["error"] == "stale heartbeat at launch"


def _write_status(run_dir, payload):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "status.json"), "w") as fh:
        json.dump(payload, fh)
