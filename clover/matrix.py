"""Matrix-sweep orchestrator (SPEC §11.1): enumerate a methods × scenarios
× datasets × seeds grid, dispatch each cell as its own ``clover run``
subprocess -- confirmed via read-only research this matches the bench's
actual architecture (crash isolation: one bad run can't take down the
whole matrix; GPU pinning via ``CUDA_VISIBLE_DEVICES`` is fundamentally a
subprocess-level concern anyway). Skip cells whose ``status.json`` already
says ``"done"``; retry ``failed``/``pending``/stale-``running`` ones.
Ported design, not reinvented, per SPEC. GPU-pool parallelism (the bench's
``ThreadPoolExecutor`` sized to detected CUDA devices) is a deferred
follow-up -- this machine has no GPU to run or test it against, and
sequential dispatch already satisfies resume/retry/stale detection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import yaml

from clover.config.schema import MatrixSection

#: Matches the bench's own threshold (confirmed via research): a "running"
#: status whose heartbeat hasn't updated in this long is presumed dead.
STALE_HEARTBEAT_SECONDS = 15 * 60


@dataclass(frozen=True)
class MatrixCell:
    dataset: str
    method: str
    scenario: str
    seed: int

    @property
    def run_id(self) -> str:
        """``{dataset}__{method}__{scenario}__seed{seed}`` -- the bench's
        own run-id convention; ``clover.reporting.parse_run_id`` parses
        this exact shape back."""
        return f"{self.dataset}__{self.method}__{self.scenario}__seed{self.seed}"


def enumerate_cells(matrix: MatrixSection) -> List[MatrixCell]:
    return [
        MatrixCell(dataset=d, method=m, scenario=s, seed=seed)
        for d in matrix.datasets
        for m in matrix.methods
        for s in matrix.scenarios
        for seed in matrix.seeds
    ]


def cell_config(matrix: MatrixSection, cell: MatrixCell) -> Dict[str, Any]:
    """Synthesize a single-run config dict for *cell* -- the same raw
    shape ``clover.config.resolve_config`` expects."""
    stream = dict(matrix.stream)
    stream["dataset"] = cell.dataset
    stream["scenario"] = cell.scenario
    stream.setdefault("stream_seed", cell.seed)

    method_cfg = {"name": cell.method, **matrix.method_overrides.get(cell.method, {})}

    return {
        "run": {"name": cell.run_id, "seed": cell.seed, "output_dir": matrix.output_dir},
        "stream": stream,
        "method": method_cfg,
        "training": dict(matrix.training),
    }


def run_dir_for(matrix: MatrixSection, cell: MatrixCell) -> str:
    return os.path.join(matrix.output_dir, cell.run_id)


def read_status(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "status.json")
    if not os.path.exists(path):
        return {"state": "pending"}
    with open(path) as fh:
        return json.load(fh)


def is_stale(
    status: Dict[str, Any], now: datetime, threshold_seconds: int = STALE_HEARTBEAT_SECONDS
) -> bool:
    """A ``"running"`` status with no recent heartbeat is presumed dead
    (the process that was writing it crashed without updating
    ``status.json`` to ``"failed"``)."""
    if status.get("state") != "running":
        return False
    heartbeat_str = status.get("heartbeat")
    if not heartbeat_str:
        return True
    try:
        heartbeat = datetime.fromisoformat(heartbeat_str)
    except ValueError:
        return True
    if heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=timezone.utc)
    return (now - heartbeat).total_seconds() > threshold_seconds


def _mark_stale_as_failed(run_dir: str, status: Dict[str, Any]) -> None:
    """Force-relabel a stale ``running`` status to ``failed`` before
    retrying -- purely informational bookkeeping for the orchestrator's
    own skip/retry decisions and ``clover report``'s "only done runs
    count"; the retried subprocess's ``Trainer`` resumes from whatever
    checkpoint actually exists regardless of this label."""
    path = os.path.join(run_dir, "status.json")
    tmp = path + ".tmp"
    payload = {**status, "state": "failed", "error": "stale heartbeat at launch"}
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def plan_dispatch(
    matrix: MatrixSection, cells: List[MatrixCell], now: datetime
) -> Tuple[List[MatrixCell], List[MatrixCell]]:
    """Split *cells* into ``(to_run, skipped)``: only ``"done"`` is
    skipped; ``pending``/``failed``/stale-``running`` are (re)dispatched.
    A stale ``running`` status is force-relabeled ``failed`` first,
    matching the bench's "before launching, mark stale runs as failed"
    behavior."""
    to_run: List[MatrixCell] = []
    skipped: List[MatrixCell] = []
    for cell in cells:
        run_dir = run_dir_for(matrix, cell)
        status = read_status(run_dir)
        if status.get("state") == "done":
            skipped.append(cell)
            continue
        if is_stale(status, now):
            _mark_stale_as_failed(run_dir, status)
        to_run.append(cell)
    return to_run, skipped


def dispatch_cell(matrix: MatrixSection, cell: MatrixCell) -> "subprocess.CompletedProcess[bytes]":
    """Write this cell's synthesized config and run it via ``clover run``
    as a subprocess -- crash isolation; a hung/crashed cell can't take
    down the rest of the matrix."""
    run_dir = run_dir_for(matrix, cell)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as fh:
        yaml.safe_dump(cell_config(matrix, cell), fh, sort_keys=False)
    return subprocess.run([sys.executable, "-m", "clover.cli", "run", config_path])


def run_matrix(matrix: MatrixSection, confirm: bool = False) -> Dict[str, int]:
    """Enumerate, plan, and sequentially dispatch the whole grid. Returns
    ``{"done": n, "failed": n, "skipped": n}``."""
    cells = enumerate_cells(matrix)
    to_run, skipped = plan_dispatch(matrix, cells, datetime.now(timezone.utc))

    if not confirm:
        answer = input(
            f"About to launch {len(to_run)} run(s) ({len(skipped)} already done). "
            "Type 'yes' to launch: "
        )
        if answer.strip().lower() != "yes":
            print("run-matrix: aborted.")
            return {"done": 0, "failed": 0, "skipped": len(skipped)}

    done = 0
    failed = 0
    for cell in to_run:
        result = dispatch_cell(matrix, cell)
        if result.returncode == 0:
            done += 1
        else:
            failed += 1
            print(f"run-matrix: {cell.run_id} FAILED (exit {result.returncode})", file=sys.stderr)

    return {"done": done, "failed": failed, "skipped": len(skipped)}
