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

import inspect

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from clover.backbones import get_backbone
from clover.backbones.timm_vit import TimmViTHooks
from clover.core.spec import DatasetInfo
from clover.core.stream import build_benchmark
from clover.datasets.synthetic import SyntheticDataset
from clover.methods import get_method, list_methods
from clover.methods.base import StreamInfo, TrainContext
from clover.scenarios import get_scenario, list_scenarios
from clover.training import RunConfig, Trainer

NUM_CLASSES = 20
INIT_CLS = 4
INCREMENT = 4

#: Smallest real timm ViT: the same `Block`/`Attention` classes and the same
#: 12-block structure as ViT-B/16, at 1/16 the width. `pretrained=False`
#: builds the architecture only -- no download, no network, CPU-only.
REAL_TIMM_BASE = "vit_tiny_patch16_224"
REAL_INPUT_SIZE = 224
REAL_CHANNELS = 3


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


# --- the same gate, on a real timm ViT (P11-A) --------------------------


def _base_model_key(method_cls: type) -> str:
    """``backbone:`` or ``base_model:`` for this method, read off the
    registry rather than a hardcoded list, so a new method is classified
    automatically.

    A method whose ``default_backbone`` is a *mechanism wrapper* (prompt
    pool, prefix, adapter stack) takes ``base_model``, which swaps the base
    underneath the mechanism. A method whose default is a plain base model
    (SimpleCIL) takes ``backbone``: its base *is* its backbone. Getting this
    backwards silently deletes the mechanism and degrades the method to a
    linear probe -- it doesn't crash, which is exactly why it's derived here
    instead of written down twice.
    """
    factory = get_backbone(method_cls.default_backbone)
    return "base_model" if "base_model" in inspect.signature(factory).parameters else "backbone"


def _real_base_loader(classes, generator):
    targets = torch.tensor([c for c in classes for _ in range(2)])
    images = torch.randn(
        len(targets), REAL_CHANNELS, REAL_INPUT_SIZE, REAL_INPUT_SIZE, generator=generator
    )
    return DataLoader(TensorDataset(images, targets), batch_size=4), images


@pytest.mark.parametrize("method_name", sorted(list_methods()))
def test_registered_method_is_finite_on_a_real_timm_base(method_name, make_experience):
    """Every method must also survive a *real* timm ViT, not just TinyViT.

    The 6-scenario x 40-epoch product above stays on TinyViT: it asserts
    above-chance accuracy, which needs a learnable dataset and a budget no
    12-block ViT can be given on CPU. This companion gate keeps the same
    registry-driven coverage (every method, automatically) and the same
    revisit shape -- experience 1 re-presents class 0 as a same-id revisit --
    but asserts only what a real backbone can be held to offline: the
    mechanism splices in, the whole lifecycle runs, and every parameter and
    logit stays finite.
    """
    method_cls = get_method(method_name)
    method = method_cls()
    stream_info = StreamInfo(
        dataset="synthetic",
        nb_experiences=2,
        total_classes=4,
        input_size=REAL_INPUT_SIZE,
        channels=REAL_CHANNELS,
    )
    key = _base_model_key(method_cls)
    method.build(stream_info, {key: REAL_TIMM_BASE, "pretrained": False})
    if key == "base_model":
        # Guards the footgun the key distinction exists for: had `backbone:`
        # been used, this would be a bare ViT and the mechanism -- the thing
        # under test -- would be silently gone, with everything still green.
        assert not isinstance(method.backbone, TimmViTHooks), f"{method_name}: mechanism dropped"

    ctx = TrainContext(
        device=torch.device("cpu"),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
        epochs=1,
    )
    generator = torch.Generator().manual_seed(42)

    # Experience 1 re-presents class 0 under its original id -- the same-id
    # revisit shape the TinyViT gate above exercises across every scenario.
    for task_label, (classes, seen) in enumerate([([0, 1], set()), ([0, 2, 3], {0, 1})]):
        exp = make_experience(task_label, classes, seen, head_size=2 + 2 * task_label)
        loader, images = _real_base_loader(classes, generator)
        method.before_experience(exp, ctx)
        method.train_experience(exp, loader, ctx)
        method.after_experience(exp, ctx)

        classifier = method.classifier()
        for p in classifier.parameters():
            assert torch.isfinite(p).all(), f"{method_name}: non-finite param after exp {task_label}"
        with torch.no_grad():
            logits = classifier(images)
        assert logits.shape == (len(images), exp.label_space.head_size)
        assert torch.isfinite(logits).all(), f"{method_name}: non-finite logits after exp {task_label}"
