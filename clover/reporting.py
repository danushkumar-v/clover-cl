"""Report aggregation (SPEC §10-§11): rebuilds the aggregate long CSV from
each run's ``per_task.csv`` (the authoritative source -- confirmed via
read-only research this is exactly what the bench's own
``scripts/04_rebuild_long_csv.py`` does, for the exact reason `AUDIT.md`
documents: "aggregate CSV drifts during failed-run debugging; per-run
artifacts must remain the authoritative source, rebuild aggregates from
them"), plus a method×scenario×dataset summary pivot. No pandas --
CLAUDE.md pins runtime deps to exactly
torch/torchvision/timm/numpy/pyyaml/pillow.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

LONG_CSV_COLUMNS = ["run_id", "dataset", "method", "scenario", "seed", "task_idx", "metric", "value"]


def parse_run_id(dirname: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse ``{dataset}__{method}__{scenario}__seed{seed}`` -- the
    bench's own run-id convention (confirmed via read-only research), used
    here purely for grouping in the summary pivot. An unparseable name
    still gets its raw metrics folded into the long CSV (nothing lost),
    just not grouped in the summary -- mirroring the bench's own
    "unparseable dirs are skipped [from grouping] with a printed warning."
    """
    parts = dirname.split("__")
    if len(parts) != 4 or not parts[3].startswith("seed"):
        return None
    dataset, method, scenario, seed_part = parts
    seed = seed_part[len("seed") :]
    if not seed:
        return None
    return dataset, method, scenario, seed


def _read_status(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "status.json")
    if not os.path.exists(path):
        return {"state": "pending"}
    with open(path) as fh:
        return json.load(fh)


def _read_per_task_csv(run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, "per_task.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def rebuild_long_csv(run_dirs: List[str]) -> List[Dict[str, Any]]:
    """Rebuild the aggregate long-format rows from scratch -- never trust
    or append to a stale existing aggregate. Only runs whose
    ``status.json`` says ``"done"`` contribute rows; a run still `pending`/
    `failed`/`running` is silently excluded (its own `per_task.csv` may be
    incomplete or about to be overwritten by a retry).
    """
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        status = _read_status(run_dir)
        if status.get("state") != "done":
            continue
        run_id = os.path.basename(os.path.normpath(run_dir))
        parsed = parse_run_id(run_id)
        dataset, method, scenario, seed = parsed if parsed else ("unknown", "unknown", "unknown", "unknown")
        for row in _read_per_task_csv(run_dir):
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "method": method,
                    "scenario": scenario,
                    "seed": seed,
                    "task_idx": row["task_idx"],
                    "metric": row["metric"],
                    "value": row["value"],
                }
            )
    return rows


def write_long_csv(rows: List[Dict[str, Any]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LONG_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """``(method, scenario, dataset) -> {metric: value}`` -- the last
    recorded value per ``(run_id, metric)`` (max ``task_idx``), averaged
    over seeds within each ``(method, scenario, dataset)`` group. Mirrors
    the bench's ``by_method`` sheet without pandas. NaN values are
    excluded from the average (a metric that never applied for a given
    seed shouldn't pull the average toward NaN)."""
    latest: Dict[Tuple[str, str], Tuple[int, float]] = {}
    meta: Dict[str, Tuple[str, str, str]] = {}
    for row in rows:
        run_id = row["run_id"]
        metric = row["metric"]
        task_idx = int(row["task_idx"])
        value = float(row["value"])
        key = (run_id, metric)
        if key not in latest or task_idx > latest[key][0]:
            latest[key] = (task_idx, value)
        meta[run_id] = (row["method"], row["scenario"], row["dataset"])

    grouped: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
    for (run_id, metric), (_task_idx, value) in latest.items():
        if math.isnan(value):
            continue
        group_key = meta[run_id]
        grouped.setdefault(group_key, {}).setdefault(metric, []).append(value)

    return {
        group_key: {metric: statistics.mean(values) for metric, values in metrics.items()}
        for group_key, metrics in grouped.items()
    }


def write_summary_csv(summary: Dict[Tuple[str, str, str], Dict[str, float]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    metric_names = sorted({m for metrics in summary.values() for m in metrics})
    fieldnames = ["method", "scenario", "dataset", *metric_names]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (method, scenario, dataset), metrics in sorted(summary.items()):
            row: Dict[str, Any] = {"method": method, "scenario": scenario, "dataset": dataset}
            row.update({m: metrics.get(m, "") for m in metric_names})
            writer.writerow(row)
