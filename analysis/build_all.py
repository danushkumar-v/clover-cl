"""Rebuild and execute every batch's notebooks, in place.

Run: .venv\\Scripts\\python.exe analysis/build_all.py

For each `analysis/batches/*/`: regenerate its notebooks from
`_build_notebooks.py` (if present), then execute every `*.ipynb` with
`jupyter nbconvert --execute --inplace`, run from that batch's own
directory so its `sys.path.insert(0, '../..')` resolves correctly.
Ported from ``../clover-pilot-bench/analysis/v2_fixed_size/_build_notebooks.py``'s
build-then-execute pattern, folded into one script since v2 batches are
just directories to loop over rather than a single fixed notebook set.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str], cwd: Path) -> None:
    print(f"$ {' '.join(cmd)}   (in {cwd.relative_to(HERE.parent)})")
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    batch_dirs = sorted(p.parent for p in (HERE / "batches").glob("*/batch.yaml"))
    if not batch_dirs:
        print("no batches found under analysis/batches/*/batch.yaml")
        sys.exit(1)

    for batch_dir in batch_dirs:
        print(f"\n=== {batch_dir.name} ===")
        generator = batch_dir / "_build_notebooks.py"
        if generator.exists():
            run([PYTHON, generator.name], cwd=batch_dir)

        notebooks = sorted(batch_dir.glob("*.ipynb"))
        if not notebooks:
            print(f"  (no notebooks in {batch_dir.name}, skipping execution)")
            continue
        run(
            [PYTHON, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace", *(nb.name for nb in notebooks)],
            cwd=batch_dir,
        )

    print("\nbuild_all: all batches rebuilt and executed.")


if __name__ == "__main__":
    main()
