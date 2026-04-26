"""Entry point: python -m clover.cli <command> [args]

Commands
--------
inspect <config.yaml>
    Load a config, print the overlap matrix as ASCII, and save diagnostic
    plots to ./inspect_output/.
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m clover.cli <command> [args]")
        print("Commands: inspect <config.yaml>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "inspect":
        if len(sys.argv) < 3:
            print("Usage: python -m clover.cli inspect <config.yaml>")
            sys.exit(1)
        _cmd_inspect(sys.argv[2])
    else:
        print(f"Unknown command {command!r}. Available: inspect")
        sys.exit(1)


def _cmd_inspect(config_path: str) -> None:
    import os

    from clover.core.data_manager import OverlapDataManager
    from clover.utils.visualizer import plot_class_frequency, plot_overlap_matrix

    print(f"Loading config: {config_path}")
    dm = OverlapDataManager.from_yaml(config_path)

    mat = dm.get_overlap_matrix()
    T = mat.shape[0]

    print(f"\nDataset   : {dm.dataset_name}")
    print(f"Tasks     : {T}")
    print(f"Classes   : {dm.total_classes}")
    print(f"Scenario  : {dm.overlap_spec.mode if dm.overlap_spec else 'none'}\n")

    # ASCII overlap matrix
    header = "     " + "  ".join(f"T{i:02d}" for i in range(T))
    print(header)
    for i in range(T):
        row = f"T{i:02d}  " + "  ".join(f"{mat[i, j]:4d}" for j in range(T))
        print(row)

    out_dir = "./inspect_output"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(config_path))[0]

    mat_path = os.path.join(out_dir, f"{base}_overlap_matrix.png")
    freq_path = os.path.join(out_dir, f"{base}_class_frequency.png")

    try:
        plot_overlap_matrix(dm, mat_path)
        plot_class_frequency(dm, freq_path)
        print(f"\nPlots saved to {out_dir}/")
    except ImportError:
        print("\nNote: matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
