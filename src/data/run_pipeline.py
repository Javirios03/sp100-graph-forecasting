"""Run the data pipeline in dependency order.

Execute from the repository root with:
    uv run python -m src.data.run_pipeline
"""

from __future__ import annotations

import argparse
import runpy


PIPELINE_STEPS = [
    ("01_load_clean_sp100", "Download and clean S&P 100 prices"),
    ("02_build_metadata", "Download ticker metadata"),
    ("03_build_base_dataset", "Merge prices and metadata"),
    ("04_compute_features", "Compute technical features"),
    ("05_compute_targets", "Compute future-return targets"),
    ("06_build_splits", "Assign train/val/test splits"),
    ("07_build_tabular_dataset", "Build tabular baseline dataset"),
    ("08_build_temporal_dataset", "Build temporal baseline dataset"),
    ("09_build_graphs", "Build graph edges and PyG tensors"),
    ("10_build_gnn_dataset", "Build GNN snapshot dataset"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the S&P 100 data pipeline.")
    parser.add_argument(
        "--from-step",
        choices=[name for name, _ in PIPELINE_STEPS],
        default=PIPELINE_STEPS[0][0],
        help="First step to run.",
    )
    parser.add_argument(
        "--to-step",
        choices=[name for name, _ in PIPELINE_STEPS],
        default=PIPELINE_STEPS[-1][0],
        help="Last step to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pipeline steps without running them.",
    )
    return parser.parse_args()


def selected_steps(from_step: str, to_step: str) -> list[tuple[str, str]]:
    names = [name for name, _ in PIPELINE_STEPS]
    start = names.index(from_step)
    end = names.index(to_step)
    if start > end:
        raise ValueError("--from-step must come before or equal --to-step")
    return PIPELINE_STEPS[start : end + 1]


def main() -> None:
    args = parse_args()

    if args.list:
        for name, description in PIPELINE_STEPS:
            print(f"{name}: {description}")
        return

    for step_name, description in selected_steps(args.from_step, args.to_step):
        module_name = f"src.data.{step_name}"
        print(f"\n=== {step_name}: {description} ===")
        runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
