"""Shipped example configs (P9, SPEC §2/§9) validate for shape/schema
correctness. Real training on any real dataset currently crashes on the
first batch (no method/backbone threads a channel count through -- see
PLAN.md's P10 phase), so verification here stops at `clover preflight`/
`clover inspect` (config resolution + plan feasibility), never a full
`clover run`. `patched_cifar100` (tests/conftest.py) keeps this fully
offline -- no real CIFAR-100 download.
"""

from __future__ import annotations

import glob
import os

from clover.cli import main
from clover.config import load_yaml, resolve_config
from clover.config.schema import MatrixSection
from clover.core.planner import resolve as resolve_plan
from clover.matrix import cell_config, enumerate_cells

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
_SINGLE_RUN_CONFIGS = sorted(
    p
    for p in glob.glob(os.path.join(_CONFIGS_DIR, "*.yaml"))
    if not os.path.basename(p).startswith("matrix_")
)


def test_configs_directory_has_the_expected_shipped_files():
    names = {os.path.basename(p) for p in glob.glob(os.path.join(_CONFIGS_DIR, "*.yaml"))}
    assert names == {
        "cifar100_disjoint_baseline_simplecil.yaml",
        "cifar100_exact_replay_l2p.yaml",
        "cifar100_partial_overlap_dualprompt.yaml",
        "cifar100_long_range_revisit_coda_prompt.yaml",
        "cifar100_mid_range_revisit_ease.yaml",
        "cifar100_cumulative_drift_ranpac.yaml",
        "cifar100_distribution_shift_mos.yaml",
        "matrix_full.yaml",
        "matrix_simplecil_vitb16.yaml",
        "matrix_simplecil_vitb16_inr.yaml",
        "matrix_all9_vitb16_cifar224.yaml",
        "matrix_all9_vitb16_inr.yaml",
        "pilot_gpu_mos.yaml",
    }


def test_the_9_method_matrices_give_every_method_its_own_published_training_block():
    """P11-C: the whole point of these two configs is that a shared
    ``training`` block cannot represent a 9-method comparison -- the
    published configs disagree on optimizer, LR (30x), batch (16x) and
    schedule. Validated offline via ``dataset_num_classes``, so this needs
    neither CIFAR-100 nor a staged ImageNet-R.
    """
    for name in ("matrix_all9_vitb16_cifar224", "matrix_all9_vitb16_inr"):
        matrix = MatrixSection.from_dict(load_yaml(os.path.join(_CONFIGS_DIR, f"{name}.yaml")))
        assert len(matrix.methods) == 9
        cells = enumerate_cells(matrix)
        assert len(cells) == 9 * 7 * 3

        seen: dict = {}
        for cell in cells:
            resolved = resolve_config(cell_config(matrix, cell))
            assert resolved.training.optimizer is not None
            seen[cell.method] = (
                resolved.training.optimizer.name,
                resolved.training.optimizer.lr,
                resolved.training.batch_size,
                resolved.training.scheduler,
            )

        # every method carries a distinct, published training setup
        assert seen["mos"] == ("sgd", 0.03, 48, "cosine")
        assert seen["l2p"] == ("adam", 0.001875, 16, "constant")
        assert seen["coda_prompt"] == ("adam", 0.001, 128, "cosine")
        assert seen["ease"] == ("sgd", 0.025, 48, "cosine")
        # both optimizers are genuinely in use across the 9
        assert {v[0] for v in seen.values()} == {"sgd", "adam"}

        # SimpleCIL's base IS its backbone; the other 8 keep their mechanism
        # and swap the base underneath -- using the wrong key silently
        # deletes the mechanism.
        assert "backbone" in matrix.method_overrides["simplecil"]
        for method in matrix.methods:
            if method != "simplecil":
                assert "base_model" in matrix.method_overrides[method], method
                assert "backbone" not in matrix.method_overrides[method], method


def test_every_shipped_single_run_config_passes_preflight(patched_cifar100, tmp_path, capsys):
    for config_path in _SINGLE_RUN_CONFIGS:
        exit_code = main(["preflight", config_path])
        assert exit_code == 0, f"{config_path} failed preflight: {capsys.readouterr().err}"


def test_every_shipped_single_run_config_inspects_to_a_feasible_plan(patched_cifar100, capsys):
    for config_path in _SINGLE_RUN_CONFIGS:
        exit_code = main(["inspect", config_path])
        assert exit_code == 0, f"{config_path} failed inspect: {capsys.readouterr().err}"
        out = capsys.readouterr().out
        assert "dataset: cifar100 (100 classes)" in out
        assert "experiences: 10" in out


def test_matrix_full_yaml_every_cell_resolves_to_a_feasible_plan(patched_cifar100):
    from clover.core.spec import DatasetInfo
    from clover.datasets import get_dataset

    raw = load_yaml(os.path.join(_CONFIGS_DIR, "matrix_full.yaml"))
    matrix = MatrixSection.from_dict(raw)
    assert len(matrix.methods) == 9
    assert len(matrix.scenarios) == 7
    assert matrix.datasets == ["cifar100"]

    cells = enumerate_cells(matrix)
    assert len(cells) == 9 * 7 * 3

    # every cell shares the same dataset -- resolve num_classes once, not per cell
    info = DatasetInfo("cifar100", get_dataset("cifar100")(root="./data").num_classes)

    for cell in cells:
        resolved = resolve_config(cell_config(matrix, cell))
        resolve_plan(resolved.stream_spec, info)


def test_matrix_simplecil_vitb16_every_cell_resolves_to_a_feasible_plan(patched_cifar100):
    from clover.core.spec import DatasetInfo
    from clover.datasets import get_dataset

    raw = load_yaml(os.path.join(_CONFIGS_DIR, "matrix_simplecil_vitb16.yaml"))
    matrix = MatrixSection.from_dict(raw)
    assert matrix.methods == ["simplecil"]
    assert matrix.datasets == ["cifar224"]
    assert matrix.method_overrides["simplecil"]["backbone"] == "vit_base_patch16_224"
    assert matrix.method_overrides["simplecil"]["pretrained"] is True

    cells = enumerate_cells(matrix)
    assert len(cells) == 1 * 7 * 3

    ds = get_dataset("cifar224")(root="./data")
    assert ds.input_size == 224  # what a pretrained ViT-B/16 checkpoint expects
    info = DatasetInfo("cifar224", ds.num_classes)

    for cell in cells:
        resolved = resolve_config(cell_config(matrix, cell))
        resolve_plan(resolved.stream_spec, info)
