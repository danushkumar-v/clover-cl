"""``clover`` entry point: run / smoke / inspect / preflight (SPEC §9, §11).

``report`` (aggregating run artifacts) and ``run-matrix`` (sweeps,
resume/retry/stale/GPU-pool orchestration) are P7 scope, not this phase's.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clover.config import ResolvedConfig, load_yaml, resolve_config
from clover.core.planner import resolve as resolve_plan
from clover.core.spec import DatasetInfo, StreamSpec
from clover.core.stream import build_benchmark
from clover.datasets import get_dataset
from clover.methods import get_method, list_methods
from clover.training import RunConfig, Trainer

_SMOKE_METHOD_INIT_CLS = 4
_SMOKE_METHOD_INCREMENT = 4


def _dataset_info(dataset_name: str) -> DatasetInfo:
    dataset_cls = get_dataset(dataset_name)
    return DatasetInfo(dataset_name, dataset_cls().num_classes)


def _run_dir_for(resolved: ResolvedConfig, config_path: str) -> str:
    name = resolved.run.name or Path(config_path).stem
    return os.path.join(resolved.run.output_dir, name)


def _build_run_config(resolved: ResolvedConfig, run_dir: str) -> RunConfig:
    optimizer = resolved.training.optimizer
    return RunConfig(
        run_dir=run_dir,
        seed=resolved.run.seed,
        batch_size=resolved.training.batch_size,
        device="cpu",
        amp=resolved.training.amp != "none",
        optimizer_name=optimizer.name if optimizer else "adam",
        optimizer_lr=optimizer.lr if optimizer else 1e-3,
        epochs=resolved.training.epochs,
    )


def _error_message(exc: BaseException) -> str:
    # KeyError's str() wraps the message in an extra pair of quotes (a
    # well-known repr quirk); args[0] is the clean message we actually raised.
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def cmd_run(args: argparse.Namespace) -> int:
    raw = load_yaml(args.config)
    resolved = resolve_config(raw, smoke=(args.profile == "smoke"))

    run_dir = _run_dir_for(resolved, args.config)
    os.makedirs(run_dir, exist_ok=True)
    resolved.save(os.path.join(run_dir, "config_resolved.yaml"))

    dataset_cls = get_dataset(resolved.stream_spec.dataset)
    train_dataset = dataset_cls(train=True)
    test_dataset = dataset_cls(train=False)
    info = DatasetInfo(resolved.stream_spec.dataset, train_dataset.num_classes)

    benchmark = build_benchmark(
        resolved.stream_spec, info, train_dataset.get_class_to_indices(), test_dataset.get_class_to_indices()
    )
    method = get_method(resolved.method_name)()
    run_config = _build_run_config(resolved, run_dir)

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
    info = _dataset_info(resolved.stream_spec.dataset)
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
    info = _dataset_info(resolved.stream_spec.dataset)
    resolve_plan(resolved.stream_spec, info)  # raises on infeasible placement/budget overflow
    print("preflight OK")
    return 0


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
