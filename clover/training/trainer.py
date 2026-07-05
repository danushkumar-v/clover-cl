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
from clover.evaluation import PerClassEvaluator, RMatrix
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


#: The one place optimizer names are mapped to classes -- clover/config's
#: OptimizerConfig validates against this same set, so an unknown optimizer
#: name is a config error at resolve time, not a KeyError deep in a run.
OPTIMIZER_FACTORIES = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        _seed_everything(self.config.seed)

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

        latest_task = self._latest_completed_task()
        if latest_task is None:
            resume_from = 0
            r_matrix = RMatrix(self.benchmark.nb_experiences)
        else:
            resume_from = latest_task + 1
            for exp in train_experiences[:resume_from]:
                self.method.before_experience(exp, ctx)
            checkpoint = torch.load(self._checkpoint_path(latest_task), weights_only=False)
            self.method.load_state_dict(checkpoint["method_state"])
            r_matrix = RMatrix.from_array(checkpoint["r_matrix"])

        self._write_status("running", resume_from - 1)

        evaluator = PerClassEvaluator(self.method.classifier(), self.test_dataset, device)

        for exp in train_experiences:
            if exp.task_label < resume_from:
                continue

            self.method.before_experience(exp, ctx)
            loader = self._build_loader(exp)
            self.method.train_experience(exp, loader, ctx)
            self.method.after_experience(exp, ctx)

            seen_test = test_experiences[: exp.task_label + 1]
            per_class_acc = evaluator.evaluate(seen_test)
            for test_exp in seen_test:
                first_app = test_exp.first_appearance_of
                if first_app:
                    mean_acc = sum(per_class_acc[c] for c in first_app) / len(first_app)
                    r_matrix.update(test_exp.task_label, exp.task_label, mean_acc)

            r_matrix.save(self._r_matrix_path())
            _atomic_torch_save(
                {
                    "task": exp.task_label,
                    "method_state": self.method.state_dict(),
                    "r_matrix": r_matrix.to_array(),
                },
                self._checkpoint_path(exp.task_label),
            )
            self._write_status("running", exp.task_label)

        self._write_status("done", self.benchmark.nb_experiences - 1)
        return r_matrix
