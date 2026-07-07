"""Core-owned Trainer: loop, AMP, seeding, checkpoints (SPEC §6.2).

Replaces PILOT's per-method self-contained loops. Owns experience iteration,
dataloader construction, per-experience checkpoint/resume, calling the
evaluator, and writing run artifacts. Methods keep their inner epoch loop in
``train_experience`` -- CL methods differ too much to share one.

Checkpoint/resume design ported (bench-proven): atomic write-temp-then-rename
for both ``status.json`` and checkpoint files; the checkpoint glob itself is
the resume-from-task source of truth (no separate resume pointer); a method's
head must be grown to the right size (replaying ``before_experience`` for
every completed experience) *before* ``load_state_dict``. No background
heartbeat thread / stale-run detection here -- that's P7 run-matrix
orchestration territory.
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from clover.core.experience import Experience
from clover.core.stream import Benchmark
from clover.datasets.base import CLDataset
from clover.evaluation import PerClassEvaluator, PerClassHistory, RMatrix
from clover.evaluation.metrics import (
    aggregate_accuracy,
    anchor_classes,
    anchor_retention,
    average_incremental_accuracy,
    backward_transfer,
    classify_task,
    echo_source_ids,
    first_appearance_map,
    forgetting,
    forward_transfer,
    long_range_retention,
    rag_mean,
    repetition_gain,
    revisit_task_map,
)
from clover.methods.base import CLMethod, StreamInfo, TrainContext


class ExperienceDataset(Dataset):
    """Wraps a ``CLDataset`` + one experience's ``image_indices`` into
    ``(image, class_id)`` pairs -- the "dataloader construction" SPEC
    assigns to the Trainer, not the method."""

    def __init__(self, base_dataset: CLDataset, image_indices: Dict[int, List[int]]) -> None:
        self._base = base_dataset
        self._pairs: List[Tuple[int, int]] = [
            (idx, cls) for cls, idxs in image_indices.items() for idx in idxs
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int) -> Tuple[Any, int]:
        idx, cls = self._pairs[i]
        image, _ = self._base[idx]
        return image, cls


@dataclass
class RunConfig:
    run_dir: str
    seed: int = 42
    batch_size: int = 32
    device: str = "cpu"
    amp: bool = False
    optimizer_name: str = "adam"
    optimizer_lr: float = 1e-3
    epochs: int = 1
    cudnn_benchmark: bool = False


#: The one place optimizer names are mapped to classes -- clover/config's
#: OptimizerConfig validates against this same set, so an unknown optimizer
#: name is a config error at resolve time, not a KeyError deep in a run.
OPTIMIZER_FACTORIES = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}


def _seed_everything(seed: int, cudnn_benchmark: bool = False) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # SPEC §11: determinism on by default, cudnn_benchmark opt-in (recorded
    # via TrainingSection.cudnn_benchmark in config_resolved.yaml). No-op on
    # CPU-only runs -- these flags only affect cudnn's GPU kernel selection.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _atomic_torch_save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Trainer:
    """Owns experience iteration, dataloaders, checkpoint/resume, and evaluation."""

    def __init__(
        self,
        method: CLMethod,
        benchmark: Benchmark,
        train_dataset: CLDataset,
        test_dataset: CLDataset,
        config: RunConfig,
    ) -> None:
        self.method = method
        self.benchmark = benchmark
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def _checkpoint_path(self, task: int) -> str:
        return os.path.join(self.config.run_dir, f"ckpt_task{task:02d}.pt")

    def _status_path(self) -> str:
        return os.path.join(self.config.run_dir, "status.json")

    def _r_matrix_path(self) -> str:
        return os.path.join(self.config.run_dir, "R_matrix.npy")

    def _per_task_csv_path(self) -> str:
        return os.path.join(self.config.run_dir, "per_task.csv")

    def _read_per_task_rows(self) -> List[Tuple[int, str, float]]:
        """Rows already written by an earlier (possibly crashed) run --
        read back verbatim on resume rather than recomputed, since some
        CLOVER metrics (e.g. RAG_mean) aggregate over the *whole* history
        and would leak future information into an earlier task's row if
        recomputed from today's fuller history instead of carried forward
        as originally written."""
        path = self._per_task_csv_path()
        if not os.path.exists(path):
            return []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            return [(int(row["task_idx"]), row["metric"], float(row["value"])) for row in reader]

    def _write_per_task_rows(self, rows: List[Tuple[int, str, float]]) -> None:
        path = self._per_task_csv_path()
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["task_idx", "metric", "value"])
            writer.writerows(rows)
        os.replace(tmp, path)

    def _latest_completed_task(self) -> Optional[int]:
        paths = sorted(glob(os.path.join(self.config.run_dir, "ckpt_task*.pt")))
        if not paths:
            return None
        basename = os.path.basename(paths[-1])
        return int(basename[len("ckpt_task") : -len(".pt")])

    def _write_status(self, state: str, last_completed_task: int) -> None:
        _atomic_write_json(
            self._status_path(),
            {
                "state": state,
                "last_completed_task": last_completed_task,
                "heartbeat": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _build_loader(self, exp: Experience) -> DataLoader:
        dataset = ExperienceDataset(self.train_dataset, exp.image_indices)
        # A per-experience generator, not the global RNG: shuffle order must
        # depend only on (seed, experience), never on how many *other*
        # experiences happened to run before it. Otherwise a resumed run
        # (which replays before_experience but skips train_experience for
        # already-completed experiences) draws a different global-RNG
        # position than an uninterrupted run reaches by that point, and
        # floating-point mean() is order-sensitive enough to break exact
        # resume reproducibility.
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + exp.task_label)
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, generator=generator
        )

    def run(self) -> RMatrix:
        os.makedirs(self.config.run_dir, exist_ok=True)
        _seed_everything(self.config.seed, self.config.cudnn_benchmark)

        device = torch.device(self.config.device)
        optimizer_cls = OPTIMIZER_FACTORIES[self.config.optimizer_name]
        optimizer_lr = self.config.optimizer_lr
        ctx = TrainContext(
            device=device,
            amp=self.config.amp,
            optimizer_factory=lambda params: optimizer_cls(params, lr=optimizer_lr),
            epochs=self.config.epochs,
        )

        stream_info = StreamInfo(
            dataset=self.benchmark.spec.dataset,
            nb_experiences=self.benchmark.nb_experiences,
            total_classes=self.benchmark.total_classes,
            input_size=self.train_dataset.input_size,
        )
        self.method.build(stream_info, {})

        train_experiences = list(self.benchmark.train_stream)
        test_experiences = list(self.benchmark.test_stream)

        # Bookkeeping for the CLOVER metrics (SPEC §10): computed once from
        # the resolved stream/plan, not re-derived per experience.
        first_appearance = first_appearance_map(self.benchmark.train_stream)
        revisit_task = revisit_task_map(self.benchmark.train_stream)
        anchor_ids = anchor_classes(self.benchmark.train_stream)
        echo_table = self.benchmark.plan.echo_table
        source_ids = sorted(echo_source_ids(echo_table))
        retention_ids = (
            source_ids if source_ids else (train_experiences[0].classes_in_this_experience if train_experiences else [])
        )
        final_task = self.benchmark.nb_experiences - 1

        latest_task = self._latest_completed_task()
        if latest_task is None:
            resume_from = 0
            r_matrix = RMatrix(self.benchmark.nb_experiences)
            history = PerClassHistory()
        else:
            resume_from = latest_task + 1
            for exp in train_experiences[:resume_from]:
                self.method.before_experience(exp, ctx)
            checkpoint = torch.load(self._checkpoint_path(latest_task), weights_only=False)
            self.method.load_state_dict(checkpoint["method_state"])
            r_matrix = RMatrix.from_array(checkpoint["r_matrix"])
            history = PerClassHistory.from_dict(checkpoint["history"])

        self._write_status("running", resume_from - 1)
        ctx.logger.info(
            f"run starting: {self.benchmark.nb_experiences} experience(s), "
            f"resuming from experience {resume_from}"
        )

        evaluator = PerClassEvaluator(self.method.classifier(), self.test_dataset, device)
        # Only trust rows for tasks the checkpoint confirms are actually
        # done -- per_task.csv is written *before* the checkpoint each
        # round, so a crash between the two writes can leave a stale row
        # for the about-to-be-redone task; without this filter, resuming
        # would duplicate it.
        per_task_rows = [row for row in self._read_per_task_rows() if row[0] < resume_from]

        for exp in train_experiences:
            if exp.task_label < resume_from:
                continue

            self.method.before_experience(exp, ctx)
            loader = self._build_loader(exp)
            self.method.train_experience(exp, loader, ctx)
            self.method.after_experience(exp, ctx)

            seen_test = test_experiences[: exp.task_label + 1]
            per_class_acc = evaluator.evaluate(seen_test)
            history.record(exp.task_label, per_class_acc)
            for test_exp in seen_test:
                first_app = test_exp.first_appearance_of
                if first_app:
                    mean_acc = sum(per_class_acc[c] for c in first_app) / len(first_app)
                    r_matrix.update(test_exp.task_label, exp.task_label, mean_acc)

            t = exp.task_label
            R = r_matrix.to_array()
            returning_ids, fresh_ids = classify_task(exp, echo_table, revisit_task, first_appearance)
            per_task_rows.extend(
                [
                    (t, "A_t", aggregate_accuracy(R, t)),
                    (t, "AIA", average_incremental_accuracy(R, t)),
                    (t, "BWT", backward_transfer(R, t)),
                    (t, "Forgetting", forgetting(R, t)),
                    (t, "FWT", forward_transfer(R, t)),
                    (t, "RAG_mean", rag_mean(history, first_appearance, revisit_task)),
                    (t, "Repetition_Gain", repetition_gain(history, t, returning_ids, fresh_ids)),
                ]
            )
            if t == final_task:
                per_task_rows.extend(
                    [
                        (t, "Anchor_Retention", anchor_retention(history, anchor_ids, t)),
                        (t, "Long_Range_Retention", long_range_retention(history, retention_ids, t)),
                    ]
                )
            self._write_per_task_rows(per_task_rows)

            r_matrix.save(self._r_matrix_path())
            _atomic_torch_save(
                {
                    "task": exp.task_label,
                    "method_state": self.method.state_dict(),
                    "r_matrix": r_matrix.to_array(),
                    "history": history.to_dict(),
                },
                self._checkpoint_path(exp.task_label),
            )
            self._write_status("running", exp.task_label)
            ctx.logger.info(f"experience {exp.task_label} done")

        self._write_status("done", self.benchmark.nb_experiences - 1)
        ctx.logger.info("run complete")
        return r_matrix
