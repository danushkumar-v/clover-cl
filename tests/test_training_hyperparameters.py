"""Per-method training hyperparameters (P11-C).

The 9 published method configs disagree on optimizer (6 SGD / 3 Adam),
learning rate (30x spread), batch size (16x) and schedule. Before P11-C the
config layer could express none of that: ``TrainingSection`` had no
``weight_decay``/``scheduler``/``min_lr``, and a matrix applied one shared
``training`` block to every cell. These tests pin both halves of the fix.
"""

from __future__ import annotations

import pytest
import torch

from clover.config import resolve_config
from clover.config.schema import MatrixSection, OptimizerConfig, TrainingSection
from clover.matrix import cell_config, enumerate_cells
from clover.methods.base import TrainContext

# --- Gap A: the knobs exist and round-trip ---------------------------------


def test_optimizer_config_carries_weight_decay():
    cfg = OptimizerConfig.from_dict({"name": "sgd", "lr": 0.03, "weight_decay": 5e-4})
    assert (cfg.name, cfg.lr, cfg.weight_decay) == ("sgd", 0.03, 5e-4)
    assert cfg.to_dict() == {"name": "sgd", "lr": 0.03, "weight_decay": 5e-4}


def test_weight_decay_defaults_to_zero_and_rejects_negative():
    assert OptimizerConfig.from_dict({"name": "adam"}).weight_decay == 0.0
    with pytest.raises(ValueError, match="weight_decay must be >= 0"):
        OptimizerConfig.from_dict({"name": "adam", "weight_decay": -1.0})


def test_training_section_carries_scheduler_and_min_lr():
    section = TrainingSection.from_dict({"epochs": 20, "scheduler": "cosine", "min_lr": 1e-8})
    assert (section.scheduler, section.min_lr) == ("cosine", 1e-8)
    assert TrainingSection.from_dict(section.to_dict()).scheduler == "cosine"


def test_unknown_scheduler_is_rejected_by_name():
    with pytest.raises(ValueError, match="scheduler must be one of"):
        TrainingSection.from_dict({"scheduler": "exponential"})


def test_scheduler_defaults_to_constant():
    assert TrainingSection.from_dict({}).scheduler == "constant"


# --- Gap B: per-method training layering in a matrix -----------------------


def _matrix() -> MatrixSection:
    return MatrixSection.from_dict(
        {
            "methods": ["mos", "l2p"],
            "scenarios": ["disjoint_baseline"],
            "datasets": ["synthetic"],
            "seeds": [1],
            "stream": {"init_cls": 5, "increment": 5},
            "training": {
                "epochs": 5,
                "batch_size": 16,
                "optimizer": {"name": "adam", "lr": 0.001875},
                "scheduler": "constant",
            },
            "training_overrides": {
                "mos": {
                    "epochs": 20,
                    "batch_size": 48,
                    "scheduler": "cosine",
                    "optimizer": {"name": "sgd", "lr": 0.03, "weight_decay": 5e-4},
                }
            },
        }
    )


def test_each_method_gets_its_own_training_block():
    matrix = _matrix()
    got = {c.method: cell_config(matrix, c)["training"] for c in enumerate_cells(matrix)}

    assert got["mos"]["optimizer"] == {"name": "sgd", "lr": 0.03, "weight_decay": 5e-4}
    assert (got["mos"]["epochs"], got["mos"]["batch_size"]) == (20, 48)
    assert got["mos"]["scheduler"] == "cosine"

    # The un-overridden method keeps the shared block untouched -- an
    # override must not leak across cells.
    assert got["l2p"]["optimizer"] == {"name": "adam", "lr": 0.001875}
    assert (got["l2p"]["epochs"], got["l2p"]["batch_size"]) == (5, 16)
    assert got["l2p"]["scheduler"] == "constant"


def test_optimizer_override_merges_key_wise_rather_than_replacing():
    """An override may set just ``lr`` without restating the optimizer name."""
    matrix = MatrixSection.from_dict(
        {
            "methods": ["ease"],
            "scenarios": ["disjoint_baseline"],
            "datasets": ["synthetic"],
            "seeds": [1],
            "stream": {"init_cls": 5, "increment": 5},
            "training": {"optimizer": {"name": "sgd", "lr": 0.01, "weight_decay": 5e-4}},
            "training_overrides": {"ease": {"optimizer": {"lr": 0.025}}},
        }
    )
    cell = enumerate_cells(matrix)[0]
    optimizer = cell_config(matrix, cell)["training"]["optimizer"]
    assert optimizer == {"name": "sgd", "lr": 0.025, "weight_decay": 5e-4}


def test_a_layered_cell_still_passes_full_config_validation():
    matrix = _matrix()
    for cell in enumerate_cells(matrix):
        resolved = resolve_config(cell_config(matrix, cell))
        assert resolved.training.optimizer is not None
    # and an unknown key inside an override is still rejected
    bad = _matrix()
    bad.training_overrides["mos"]["epocs"] = 3  # typo
    with pytest.raises(ValueError, match="epocs"):
        resolve_config(cell_config(bad, enumerate_cells(bad)[0]))


# --- The scheduler actually moves the learning rate ------------------------


def test_cosine_scheduler_anneals_the_learning_rate_across_epochs():
    """``TrainContext.make_scheduler`` is only useful if stepping it changes
    the optimizer's LR -- the methods own the epoch loop, so a scheduler
    that is built but never stepped would silently be a no-op."""
    epochs, lr, min_lr = 4, 0.1, 0.0

    def factory(optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )

    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=lr),
        epochs=epochs,
        scheduler_factory=factory,
    )
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = ctx.optimizer_factory([param])
    scheduler = ctx.make_scheduler(optimizer)
    assert scheduler is not None

    seen = [optimizer.param_groups[0]["lr"]]
    for _ in range(epochs):
        scheduler.step()
        seen.append(optimizer.param_groups[0]["lr"])

    assert seen[0] == pytest.approx(lr)
    assert seen == sorted(seen, reverse=True), f"LR must decrease monotonically, got {seen}"
    assert seen[-1] == pytest.approx(min_lr, abs=1e-9)


def test_make_scheduler_returns_none_for_a_constant_schedule():
    ctx = TrainContext(device=torch.device("cpu"), scheduler_factory=None)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    assert ctx.make_scheduler(optimizer) is None
