"""Registry-driven revisit-safety gate (SPEC §5.3), via the real Trainer,
across all 6 core scenarios (SPEC §8) -- P5's DONE criteria names this
combination explicitly ("sane accuracy on all 6 scenarios, incl. same-id
cumulative_drift").

P2 built the loss-policy version of this gate against local stub callables,
since no real CLMethod existed yet, and explicitly deferred registry-driven
parametrization to P3. Now that real methods are registered, this fulfills
SPEC's "a new method plugin inherits this test automatically via the
registry" -- add a method, it's covered here with no test-file changes.
"""

from __future__ import annotations

import pytest
import torch

from clover.core.spec import DatasetInfo
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method, list_methods
from clover.scenarios import get_scenario, list_scenarios
from clover.training import RunConfig, Trainer

NUM_CLASSES = 20
INIT_CLS = 4
INCREMENT = 4


def _build_benchmark(scenario_name, train_ds, test_ds):
    info = DatasetInfo("synthetic", NUM_CLASSES)
    factory = get_scenario(scenario_name)
    # mid_range_revisit's default anchor_task=4 needs an experience *after*
    # the anchor for the echo to land in (5 experiences here isn't enough
    # room for anchor_task=4); anchor_task=2 fits our 5-experience stream.
    kwargs = {"anchor_task": 2} if scenario_name == "mid_range_revisit" else {}
    spec = factory(info, INIT_CLS, INCREMENT, seed=42, **kwargs)
    return build_benchmark(spec, info, train_ds.get_class_to_indices(), test_ds.get_class_to_indices())


def _revisit_check_experience(benchmark):
    """Return (test_experience, classes_of_interest) for the last train
    experience's revisit/echo round -- every core scenario except
    disjoint_baseline concentrates its revisit signal there (end_of_stream
    or every_task placement always includes the final experience)."""
    train_exp = benchmark.train_stream[-1]
    classes_of_interest = set(train_exp.revisiting_classes) | set(train_exp.echo_map)
    if not classes_of_interest:
        return None, set()
    return benchmark.test_stream[-1], classes_of_interest


@pytest.mark.parametrize("method_name", sorted(list_methods()))
@pytest.mark.parametrize("scenario_name", sorted(list_scenarios()))
def test_registered_method_passes_revisit_safety_gate(tmp_path, method_name, scenario_name):
    train_ds = SyntheticDataset(train=True)
    test_ds = SyntheticDataset(train=False)
    benchmark = _build_benchmark(scenario_name, train_ds, test_ds)

    method = get_method(method_name)()
    # epochs/lr tuned for gradient-trained methods to reliably clear chance
    # on this tiny, randomly-initialized-backbone setup; harmless for
    # SimpleCIL, which ignores both (single closed-form pass). lr=3e-2
    # specifically needed for coda_prompt/exact_replay-style echo shapes:
    # its soft, query-only-dependent pool combination gives near-identical
    # prompts to an echo id and its source id (same underlying synthetic
    # pattern, only the noise differs), so distinguishing them relies
    # entirely on the head separating two very similar feature vectors --
    # 1e-2 wasn't reliably enough signal for that specific split.
    run_config = RunConfig(run_dir=str(tmp_path), seed=42, epochs=40, optimizer_lr=3e-2)
    trainer = Trainer(method, benchmark, train_ds, test_ds, run_config)
    trainer.run()

    classifier = method.classifier()
    for p in classifier.parameters():
        assert torch.isfinite(p).all(), f"{method_name}/{scenario_name}: non-finite classifier params"

    test_exp, classes_of_interest = _revisit_check_experience(benchmark)
    if not classes_of_interest:
        return  # disjoint_baseline: nothing revisited, nothing more to check

    pairs = [(i, c) for c in classes_of_interest for i in test_exp.image_indices.get(c, [])]
    assert pairs, f"{method_name}/{scenario_name}: no test images for the classes of interest"
    images = torch.stack([test_ds[i][0] for i, _ in pairs])
    targets = torch.tensor([c for _, c in pairs])
    with torch.no_grad():
        preds = classifier(images).argmax(dim=-1)
    accuracy = (preds == targets).float().mean().item()
    assert accuracy > 1 / NUM_CLASSES, f"{method_name}/{scenario_name}: revisit accuracy at chance"
