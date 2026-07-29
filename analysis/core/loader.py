"""Batch discovery and tidy-DataFrame loading for CLOVER analysis.

A "batch" is one immutable results drop: a ``batch.yaml`` file describing
where its raw ``runs/`` directory lives, plus provenance (dataset,
real_backbone, git_commit, date, notes). A new results drop is a new
``bNN_...`` folder under ``analysis/batches/`` -- existing batch folders are
never mutated in place. Nothing in this module may hardcode a batch name;
every batch-specific fact comes from that batch's ``batch.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Batch:
    name: str
    path: Path
    runs_dir: Path
    results_dir: Path
    dataset: str
    real_backbone: bool
    git_commit: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None


def discover_batches(analysis_root: str | Path) -> List[Batch]:
    """Find every ``batches/*/batch.yaml`` under *analysis_root*.

    ``runs_dir`` / ``results_dir`` in ``batch.yaml`` are resolved relative to
    the repo root (the parent of *analysis_root*).
    """
    analysis_root = Path(analysis_root).resolve()
    repo_root = analysis_root.parent
    batches = []
    for batch_yaml in sorted((analysis_root / "batches").glob("*/batch.yaml")):
        spec = yaml.safe_load(batch_yaml.read_text()) or {}
        batches.append(
            Batch(
                name=batch_yaml.parent.name,
                path=batch_yaml.parent,
                runs_dir=(repo_root / spec["runs_dir"]).resolve(),
                results_dir=(repo_root / spec["results_dir"]).resolve(),
                dataset=spec["dataset"],
                real_backbone=bool(spec.get("real_backbone", False)),
                git_commit=spec.get("git_commit"),
                date=spec.get("date"),
                notes=spec.get("notes"),
            )
        )
    return batches


def _parse_run_id(run_id: str) -> tuple[str, str, str, int]:
    """``{dataset}__{method}__{scenario}__seed{N}`` -> its four parts."""
    dataset, method, scenario, seed_part = run_id.split("__")
    if not seed_part.startswith("seed"):
        raise ValueError(f"unexpected run_id format: {run_id!r}")
    return dataset, method, scenario, int(seed_part[len("seed"):])


def _done_run_dirs(batch: Batch):
    for run_dir in sorted(batch.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("state") == "done":
            yield run_dir


def load_per_task(batch: Batch) -> pd.DataFrame:
    """Tidy per-task metrics for every ``done`` run in *batch*.

    Columns: ``dataset, method, scenario, seed, task_idx, metric, value``.
    """
    columns = ["dataset", "method", "scenario", "seed", "task_idx", "metric", "value"]
    frames = []
    for run_dir in _done_run_dirs(batch):
        per_task_path = run_dir / "per_task.csv"
        if not per_task_path.exists():
            continue
        dataset, method, scenario, seed = _parse_run_id(run_dir.name)
        df = pd.read_csv(per_task_path)
        df.insert(0, "seed", seed)
        df.insert(0, "scenario", scenario)
        df.insert(0, "method", method)
        df.insert(0, "dataset", dataset)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def load_r_matrices(batch: Batch) -> Dict[str, np.ndarray]:
    """``{run_id: R matrix}`` for every ``done`` run in *batch*."""
    out: Dict[str, np.ndarray] = {}
    for run_dir in _done_run_dirs(batch):
        r_path = run_dir / "R_matrix.npy"
        if r_path.exists():
            out[run_dir.name] = np.load(r_path)
    return out
