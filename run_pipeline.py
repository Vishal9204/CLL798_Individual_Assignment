from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplcache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from software_dependency_networks import ExperimentConfig, build_report_pdf, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the software dependency network complexity-science project."
    )
    parser.add_argument(
        "--connectivity",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0, 5.0],
        help="Target mean dependency counts to sweep across.",
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--node-count", type=int, default=450)
    parser.add_argument("--steps", type=int, default=5500)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--base-buffer", type=float, default=1.5)
    parser.add_argument("--buffer-slope", type=float, default=0.4)
    parser.add_argument("--large-avalanche-threshold", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Only run the simulation and generate figures/CSVs, skip PDF rendering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        connectivity_values=args.connectivity,
        replicates=args.replicates,
        node_count=args.node_count,
        steps=args.steps,
        burn_in=args.burn_in,
        sample_interval=args.sample_interval,
        base_buffer=args.base_buffer,
        buffer_slope=args.buffer_slope,
        large_avalanche_threshold=args.large_avalanche_threshold,
        seed=args.seed,
    )

    metrics = run_experiment(PROJECT_ROOT, config=config)
    if not args.skip_pdf:
        build_report_pdf(PROJECT_ROOT, metrics=metrics)

    print("Pipeline finished.")
    print(f"Summary file: {PROJECT_ROOT / 'results' / 'summary_by_connectivity.csv'}")
    print(f"Report source: {PROJECT_ROOT / 'report' / 'manuscript.tex'}")
    print(
        "Report PDF: "
        f"{PROJECT_ROOT / 'report' / 'software_dependency_networks_manuscript.pdf'}"
    )


if __name__ == "__main__":
    main()
