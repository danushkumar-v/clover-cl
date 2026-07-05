"""ctx.optimizer_factory built from RunConfig (SPEC §6.1: "optimizer factories
from config"). Unexercised by SimpleCIL (no gradient step) but real
plumbing for gradient-trained methods (P5+)."""

from __future__ import annotations

import tempfile

import torch

from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods.base import CLMethod, StreamInfo, TrainContext
from clover.training import RunConfig, Trainer


class _CapturingMethod(CLMethod):
    """Captures the TrainContext it receives, to inspect optimizer_factory."""

    name = "capturing"
    default_backbone = "tiny_mlp"

    def __init__(self) -> None:
        self.captured_ctx: TrainContext | None = None

    def build(self, stream_info: StreamInfo, cfg) -> None:
        pass

    def before_experience(self, exp, ctx: TrainContext) -> None:
        self.captured_ctx = ctx

    def train_experience(self, exp, loader, ctx: TrainContext) -> None:
        pass

    def classifier(self):
        # Real input shape (8x8 synthetic images) -> 20 logits (final head
        # size for this test's stream) so the Trainer's evaluator can run
        # without crashing; this test only inspects optimizer_factory, not
        # prediction quality.
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 20))

    def state_dict(self):
        return {}

    def load_state_dict(self, state) -> None:
        pass


def test_optimizer_factory_builds_configured_optimizer_instance():
    spec = StreamSpec(dataset="synthetic", init_cls=5, increment=5)
    info = DatasetInfo("synthetic", 20)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())

    method = _CapturingMethod()
    with tempfile.TemporaryDirectory() as run_dir:
        Trainer(
            method,
            benchmark,
            train_ds,
            test_ds,
            RunConfig(run_dir=run_dir, seed=42, optimizer_name="sgd", optimizer_lr=0.02),
        ).run()

    assert method.captured_ctx is not None
    assert method.captured_ctx.optimizer_factory is not None

    params = [torch.nn.Parameter(torch.zeros(2))]
    optimizer = method.captured_ctx.optimizer_factory(params)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["lr"] == 0.02
