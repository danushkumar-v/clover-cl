"""``clover`` entry point: run / smoke / inspect / preflight / report /
run-matrix (SPEC §9, §10-§11).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from torchvision import transforms

from clover.config import MatrixSection, ResolvedConfig, load_yaml, resolve_config
from clover.core.planner import resolve as resolve_plan
from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets import get_dataset
from clover.matrix import cell_config, enumerate_cells, run_matrix
from clover.methods import get_method, list_methods
from clover.reporting import build_summary, rebuild_long_csv, write_long_csv, write_summary_csv
from clover.training import RunConfig, Trainer

_SMOKE_METHOD_INIT_CLS = 4
_SMOKE_METHOD_INCREMENT = 4


def _dataset_info(
    dataset_name: str, root: str = "./data", num_classes: Optional[int] = None
) -> DatasetInfo:
    if num_classes is None:
        # Config-only datasets (SPEC §7: image_folder) can't be
        # constructed with zero args at all -- callers with a declared
        # count pass it directly instead of relying on this fallback.
        dataset_cls = get_dataset(dataset_name)
        num_classes = dataset_cls(root=root).num_classes
    return DatasetInfo(dataset_name, num_classes)


def _apply_default_transforms(dataset: Any) -> None:
    # Datasets declare their own transform presets (SPEC §7) but never
    # apply them -- v1's DataManager did this composition centrally
    # (train_trsf/test_trsf + common_trsf); v2 has no equivalent yet, so
    # every real (non-synthetic) dataset silently trained on raw PIL
    # images until this was wired up here.
    trsf = list(dataset.train_trsf if dataset.train else dataset.test_trsf) + list(dataset.common_trsf)
    if trsf:
        dataset.transform = transforms.Compose(trsf)


def _run_dir_for(resolved: ResolvedConfig, config_path: str) -> str:
    name = resolved.run.name or Path(config_path).stem
    return os.path.join(resolved.run.output_dir, name)


def _build_run_config(resolved: ResolvedConfig, run_dir: str, smoke: bool = False) -> RunConfig:
    optimizer = resolved.training.optimizer
    # The smoke profile is CPU-only by definition (SPEC R7) regardless of
    # what hardware happens to run it; a real run auto-detects a GPU if one
    # is present -- there is otherwise no way to ever select one (this was
    # hardcoded to "cpu" unconditionally until P9, silently blocking every
    # real cluster run).
    device = "cpu" if smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    return RunConfig(
        run_dir=run_dir,
        seed=resolved.run.seed,
        batch_size=resolved.training.batch_size,
        device=device,
        amp=resolved.training.amp != "none",
        optimizer_name=optimizer.name if optimizer else "adam",
        optimizer_lr=optimizer.lr if optimizer else 1e-3,
        epochs=resolved.training.epochs,
        cudnn_benchmark=resolved.training.cudnn_benchmark,
    )


def _attach_run_log_handler(run_dir: str) -> None:
    """Write ``TrainContext.logger`` output into ``<run_dir>/log.txt``
    (SPEC §11's run-artifact layout). Clears any handler left by a prior
    in-process ``main()`` call first -- otherwise repeated calls within one
    process (tests, or any future in-process matrix dispatch) would
    accumulate handlers and leak log lines into the wrong run's file."""
    logger = logging.getLogger("clover.training")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(run_dir, "log.txt"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def _error_message(exc: BaseException) -> str:
    # KeyError's str() wraps the message in an extra pair of quotes (a
    # well-known repr quirk); args[0] is the clean message we actually raised.
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def cmd_run(args: argparse.Namespace) -> int:
    smoke = args.profile == "smoke"
    raw = load_yaml(args.config)
    resolved = resolve_config(raw, smoke=smoke)

    run_dir = _run_dir_for(resolved, args.config)
    os.makedirs(run_dir, exist_ok=True)
    resolved.save(os.path.join(run_dir, "config_resolved.yaml"))
    _attach_run_log_handler(run_dir)

    dataset_cls = get_dataset(resolved.stream_spec.dataset)
    dataset_kwargs: dict[str, Any] = {"root": resolved.stream_spec.data_root}
    if resolved.stream_spec.dataset_num_classes is not None:
        dataset_kwargs["num_classes"] = resolved.stream_spec.dataset_num_classes
    train_dataset = dataset_cls(train=True, **dataset_kwargs)
    test_dataset = dataset_cls(train=False, **dataset_kwargs)
    _apply_default_transforms(train_dataset)
    _apply_default_transforms(test_dataset)
    info = DatasetInfo(resolved.stream_spec.dataset, train_dataset.num_classes)

    benchmark = build_benchmark(
        resolved.stream_spec, info, train_dataset.get_class_to_indices(), test_dataset.get_class_to_indices()
    )
    benchmark.save_manifest(os.path.join(run_dir, "manifest.json"))
    method = get_method(resolved.method_name)()
    run_config = _build_run_config(resolved, run_dir, smoke=smoke)

    r_matrix = Trainer(method, benchmark, train_dataset, test_dataset, run_config).run()

    print(f"run complete: {run_dir}")
    array = r_matrix.to_array()
    for t in range(benchmark.nb_experiences):
        print(f"  after experience {t}: R[{t},{t}] = {array[t, t]:.4f}")
    return 0


def _smoke_check_method(method_name: str) -> None:
    spec = StreamSpec(
        dataset="synthetic", init_cls=_SMOKE_METHOD_INIT_CLS, increment=_SMOKE_METHOD_INCREMENT
    )
    info = _dataset_info("synthetic")
    dataset_cls = get_dataset("synthetic")
    train_dataset = dataset_cls(train=True)
    test_dataset = dataset_cls(train=False)
    _apply_default_transforms(train_dataset)
    _apply_default_transforms(test_dataset)
    benchmark = build_benchmark(
        spec, info, train_dataset.get_class_to_indices(), test_dataset.get_class_to_indices()
    )
    method = get_method(method_name)()

    with tempfile.TemporaryDirectory() as run_dir:
        r_matrix = Trainer(
            method, benchmark, train_dataset, test_dataset, RunConfig(run_dir=run_dir, seed=42)
        ).run()

    array = r_matrix.to_array()
    for t in range(benchmark.nb_experiences):
        for te in range(t + 1):
            if not math.isfinite(array[te, t]):
                raise RuntimeError(f"non-finite R[{te},{t}] after experience {t}")


def cmd_smoke(args: argparse.Namespace) -> int:
    methods = sorted(list_methods())
    if not methods:
        print("smoke: no methods registered.")
        return 0
    for name in methods:
        print(f"smoke: {name} ...")
        try:
            _smoke_check_method(name)
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            print(f"smoke: {name} FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"smoke: {name} OK")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    resolved = resolve_config(raw)
    info = _dataset_info(
        resolved.stream_spec.dataset,
        root=resolved.stream_spec.data_root,
        num_classes=resolved.stream_spec.dataset_num_classes,
    )
    plan = resolve_plan(resolved.stream_spec, info)

    print(f"dataset: {resolved.stream_spec.dataset} ({info.num_classes} classes)")
    print(f"experiences: {len(plan.task_class_lists)}")
    for t, cls_list in enumerate(plan.task_class_lists):
        echoes = sorted(c for c in cls_list if c in plan.echo_table)
        revisits = sorted(c for c in cls_list if c in plan.revisit_ids)
        print(f"  [{t}] {len(cls_list)} classes -- echoes={echoes} same-id revisits={revisits}")
    print(f"head size schedule: {plan.head_size_schedule}")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    resolved = resolve_config(raw)
    get_method(resolved.method_name)  # raises if unregistered
    info = _dataset_info(
        resolved.stream_spec.dataset,
        root=resolved.stream_spec.data_root,
        num_classes=resolved.stream_spec.dataset_num_classes,
    )
    resolve_plan(resolved.stream_spec, info)  # raises on infeasible placement/budget overflow
    print("preflight OK")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = rebuild_long_csv(args.run_dirs)
    long_csv_path = os.path.join(args.out, "all_runs_long.csv")
    write_long_csv(rows, long_csv_path)

    summary = build_summary(rows)
    summary_path = os.path.join(args.out, "summary.csv")
    write_summary_csv(summary, summary_path)

    done_count = len({row["run_id"] for row in rows})
    print(f"report: {done_count}/{len(args.run_dirs)} run(s) done, {len(rows)} row(s) aggregated")
    print(f"  {long_csv_path}")
    print(f"  {summary_path}")
    return 0


def cmd_run_matrix(args: argparse.Namespace) -> int:
    raw = load_yaml(args.matrix)
    matrix = MatrixSection.from_dict(raw)

    # Preflight the first cell before dispatching the whole grid -- the
    # bench lesson (AUDIT.md): this catches a shared config typo/unknown
    # dataset fast instead of discovering it only after several cells
    # have already been dispatched.
    cells = enumerate_cells(matrix)
    if cells:
        resolve_config(cell_config(matrix, cells[0]))

    result = run_matrix(matrix, confirm=args.confirm)
    print(
        f"run-matrix: {result['done']} done, {result['failed']} failed, {result['skipped']} skipped"
    )
    return 1 if result["failed"] else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clover")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one config.")
    run_parser.add_argument("config")
    run_parser.add_argument("--profile", choices=["smoke"], default=None)
    run_parser.set_defaults(func=cmd_run)

    smoke_parser = subparsers.add_parser("smoke", help="Run every registered method through the smoke profile.")
    smoke_parser.set_defaults(func=cmd_smoke)

    inspect_parser = subparsers.add_parser("inspect", help="Resolve and print a stream plan, no training.")
    inspect_parser.add_argument("config")
    inspect_parser.set_defaults(func=cmd_inspect)

    preflight_parser = subparsers.add_parser("preflight", help="Validate a config before submitting a long run.")
    preflight_parser.add_argument("config")
    preflight_parser.set_defaults(func=cmd_preflight)

    report_parser = subparsers.add_parser("report", help="Aggregate per-run artifacts into a long CSV + summary.")
    report_parser.add_argument("run_dirs", nargs="+")
    report_parser.add_argument("--out", default="results")
    report_parser.set_defaults(func=cmd_report)

    run_matrix_parser = subparsers.add_parser(
        "run-matrix", help="Sweep methods x scenarios x datasets x seeds, with resume/retry/stale detection."
    )
    run_matrix_parser.add_argument("matrix")
    run_matrix_parser.add_argument("--confirm", action="store_true")
    run_matrix_parser.set_defaults(func=cmd_run_matrix)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func: Any = args.func
    try:
        return func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
