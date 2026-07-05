"""Registry-driven revisit-safety gate (SPEC §5.3), via the real Trainer.

P2 built the loss-policy version of this gate against local stub callables,
since no real CLMethod existed yet, and explicitly deferred registry-driven
parametrization to P3. Now that `simplecil` is registered, this fulfills
SPEC's "a new method plugin inherits this test automatically via the
registry" -- add a method, it's covered here with no test-file changes.
"""

from __future__ import annotations

import pytest
import torch

from clover.core.spec import DatasetInfo, RevisitSpec, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method, list_methods
from clover.training import RunConfig, Trainer

NUM_CLASSES = 20
INIT_CLS = 4
INCREMENT = 4


def _disjoint_spec() -> StreamSpec:
    return StreamSpec(dataset="synthetic", init_cls=INIT_CLS, increment=INCREMENT)


def _same_id_mid_stream_spec() -> StreamSpec:
    return StreamSpec(
        dataset="synthetic",
        init_cls=INIT_CLS,
        increment=INCREMENT,
        revisits=[
            RevisitSpec(classes=[0, 1], placement="spaced", label="same", images="new", min_gap=1)
        ],
    )


def _echo_at_end_spec() -> StreamSpec:
    return StreamSpec(
        dataset="synthetic",
        init_cls=INIT_CLS,
        increment=INCREMENT,
        revisits=[
            RevisitSpec(classes="task0", placement="end_of_stream", label="new", images="new")
        ],
    )


SHAPES = {
    "disjoint": _disjoint_spec,
    "same_id_mid_stream": _same_id_mid_stream_spec,
    "echo_at_end": _echo_at_end_spec,
}


def _revisit_check_experience(benchmark, shape_name):
    """Return (test_experience, classes_of_interest) for the shape's revisit round."""
    if shape_name == "disjoint":
        return None, set()
    if shape_name == "same_id_mid_stream":
        train_exp = next(e for e in benchmark.train_stream if e.is_revisit_experience())
        return benchmark.test_stream[train_exp.task_label], set(train_exp.revisiting_classes)
    if shape_name == "echo_at_end":
        train_exp = benchmark.train_stream[-1]
        return benchmark.test_stream[-1], set(train_exp.echo_map)
    raise AssertionError(f"unknown shape {shape_name!r}")


@pytest.mark.parametrize("method_name", sorted(list_methods()))
@pytest.mark.parametrize("shape_name", sorted(SHAPES))
def test_registered_method_passes_revisit_safety_gate(tmp_path, method_name, shape_name):
    spec = SHAPES[shape_name]()
    info = DatasetInfo("synthetic", NUM_CLASSES)
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = build_benchmark(
        spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices()
    )

    method = get_method(method_name)()
    trainer = Trainer(method, benchmark, train_ds, test_ds, RunConfig(run_dir=str(tmp_path), seed=42))
    trainer.run()

    classifier = method.classifier()
    for p in classifier.parameters():
        assert torch.isfinite(p).all(), f"{method_name}/{shape_name}: non-finite classifier params"

    test_exp, classes_of_interest = _revisit_check_experience(benchmark, shape_name)
    if not classes_of_interest:
        return  # disjoint shape: nothing revisited, nothing more to check

    pairs = [(i, c) for c in classes_of_interest for i in test_exp.image_indices.get(c, [])]
    assert pairs, f"{method_name}/{shape_name}: no test images for the classes of interest"
    images = torch.stack([test_ds[i][0] for i, _ in pairs])
    targets = torch.tensor([c for _, c in pairs])
    with torch.no_grad():
        preds = classifier(images).argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    assert accuracy > 1 / NUM_CLASSES, f"{method_name}/{shape_name}: revisit accuracy at chance"
