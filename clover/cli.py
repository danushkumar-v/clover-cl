"""``clover`` entry point: run / smoke / inspect / preflight / report (SPEC §9, §11).

Subcommands are added in P4. For now this only wires the console-script
entry point (``pyproject.toml``: ``clover = "clover.cli:main"``) to a no-op
parser so ``clover --help`` resolves.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clover")
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
