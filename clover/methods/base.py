"""CLMethod ABC + lifecycle contract (SPEC §6.1).

``ctx`` (``TrainContext``) provides device/AMP/logging; methods do not track
``_known_classes`` themselves -- ``exp.label_space`` is the source of truth.
``state_dict``/``load_state_dict`` aren't in SPEC's illustrative snippet but
are necessary to make the Trainer's checkpoint/resume generic across
methods (mirrors the bench's ``model._network.state_dict()`` pattern).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, Iterable, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from clover.core.experience import Experience


@dataclass(frozen=True)
class StreamInfo:
    """Lightweight, dataset-agnostic summary a method needs to build itself.

    Deliberately not the full ``Benchmark`` (which also carries plan/image
    data a method has no business touching before training starts).
    """

    dataset: str
    nb_experiences: int
    total_classes: int
    input_size: int
    channels: int


@dataclass
class TrainContext:
    """Per-run context threaded through every lifecycle hook.

    ``optimizer_factory`` builds an optimizer over whatever parameters a
    method passes it, configured from ``training.optimizer`` (SPEC: "ctx
    provides... optimizer/scheduler factories from config"). Unexercised by
    SimpleCIL (no gradient step) but real plumbing for gradient-trained
    methods.

    ``scheduler_factory`` (P11-C) is the matching half: ``training.
    scheduler`` selects a per-epoch learning-rate schedule, which 5 of the
    9 published method configs use (cosine). It is a *factory over an
    optimizer* rather than a ready scheduler because each method builds its
    own optimizer over its own trainable subset. **The method must step it**
    -- the Trainer does not own the epoch loop; every gradient-trained
    method runs its own ``for _epoch in range(ctx.epochs)`` inside
    ``train_experience``, so only the method can step a per-epoch schedule.
    Use :meth:`make_scheduler` and call ``.step()`` at the end of each
    epoch; ``None`` means a constant learning rate.

    ``epochs`` (from ``training.epochs``) was plumbed as a config field in
    P4 but never actually threaded through to a method -- SimpleCIL has no
    per-experience loop to repeat. L2P (P5) is the first method that needs
    it.
    """

    device: torch.device
    amp: bool = False
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("clover.training"))
    optimizer_factory: Optional[Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]] = None
    epochs: int = 1
    scheduler_factory: Optional[
        Callable[[torch.optim.Optimizer], Optional["torch.optim.lr_scheduler.LRScheduler"]]
    ] = None

    def make_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional["torch.optim.lr_scheduler.LRScheduler"]:
        """Build this run's LR schedule for *optimizer*, or ``None`` for a
        constant rate. Always safe to call: a run configured without a
        scheduler simply yields ``None``."""
        if self.scheduler_factory is None:
            return None
        return self.scheduler_factory(optimizer)


class CLMethod(ABC):
    """Plugin surface every continual-learning method implements.

    The framework (``clover.training.Trainer``) owns experience iteration,
    dataloader construction, checkpointing, and evaluation; a method's inner
    epoch loop lives entirely in ``train_experience`` (SPEC §6.2 -- CL
    methods differ too much to share one epoch loop).
    """

    name: ClassVar[str]
    default_backbone: ClassVar[str]
    #: Declared, not guessed (bench lesson): whether this method's backbone
    #: is frozen, so a feature cache could safely reuse its embeddings.
    cacheable_features: ClassVar[bool] = False

    @abstractmethod
    def build(self, stream_info: StreamInfo, cfg: Dict[str, Any]) -> None:
        """Construct networks/heads. Called once before any experience."""

    def before_experience(self, exp: "Experience", ctx: TrainContext) -> None:
        """Expand heads, allocate new prompts, etc. Default: no-op."""

    @abstractmethod
    def train_experience(self, exp: "Experience", loader: "DataLoader", ctx: TrainContext) -> None:
        """The method's inner training loop for one experience."""

    def after_experience(self, exp: "Experience", ctx: TrainContext) -> None:
        """Prototypes, merges, freezes, etc. Default: no-op."""

    @abstractmethod
    def classifier(self) -> nn.Module:
        """Return an ``images -> logits`` module for the shared evaluator."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Full method state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore method state from a checkpoint. ``build()`` and enough
        ``before_experience()`` replay to reach the right head size must
        happen first (the Trainer's responsibility)."""
