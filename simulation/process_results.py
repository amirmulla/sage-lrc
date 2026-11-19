"""CLI entrypoint for post-processing residual error-weight distribution CSV files."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from tamo_barg_lrc.simulation import process_error_distribution_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``process_results.py``."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a *_err.csv error-weight histogram using exact "
            "(design_distance - 1) // 2 global correction radius."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Direct path to *_err.csv (overrides default results directory lookup).",
    )
    parser.add_argument(
        "--design-distance",
        type=int,
        default=15,
        help="Code design distance used for global unique-decoding threshold.",
    )
    parser.add_argument(
        "--real-distance",
        type=int,
        default=45,
        help="Maximum error weight (number of bins) to aggregate.",
    )
    parser.add_argument(
        "--sim-itr",
        type=int,
        default=5000,
        help="Number of Monte Carlo trials per error weight used when generating the CSV.",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=99.0,
        help="Success percentage threshold to report the first sub-threshold index.",
    )
    parser.add_argument("--q", type=int, default=64, help="Field size for default path lookup.")
    parser.add_argument(
        "--r",
        nargs=2,
        type=int,
        default=[4, 3],
        metavar=("R1", "R2"),
        help="Localities for default path lookup.",
    )
    parser.add_argument(
        "--local-min-dist",
        nargs=2,
        type=int,
        default=[5, 5],
        metavar=("D1", "D2"),
        help="Local minimum distances for default path lookup.",
    )
    parser.add_argument(
        "--subgroup-types",
        nargs=2,
        choices=("add", "mult"),
        default=["add", "mult"],
        metavar=("T1", "T2"),
        help="Subgroup types for default path lookup.",
    )
    parser.add_argument(
        "--use-erasure-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select *_erasure directory suffix for default path lookup.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Base results directory for default path lookup.",
    )
    return parser


def _resolve_csv_path(args: argparse.Namespace) -> Path:
    """Resolve explicit ``--csv-path`` or construct the conventional path."""
    if args.csv_path is not None:
        return Path(args.csv_path)
    sg0_size = args.r[0] + args.local_min_dist[0] - 1
    sg1_size = args.r[1] + args.local_min_dist[1] - 1
    sim_name = (
        f"GF_{args.q}_{args.subgroup_types[0]}_{sg0_size}_{args.subgroup_types[1]}_{sg1_size}"
    )
    sub_dir = f"{sim_name}_erasure" if args.use_erasure_decoder else sim_name
    return Path(args.output_dir) / sub_dir / f"{sim_name}_err.csv"


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and post-process the residual error distribution CSV."""
    parser = build_parser()
    args = parser.parse_args(argv)
    csv_path = _resolve_csv_path(args)

    summary = process_error_distribution_csv(
        err_csv_path=csv_path,
        design_distance=args.design_distance,
        real_distance=args.real_distance,
        sim_itr=args.sim_itr,
        threshold_pct=args.threshold_pct,
    )

    print(summary["succ_count"])
    if summary["threshold_index"] is not None:
        print(summary["threshold_index"])
    for idx, pct in enumerate(summary["prob"]):
        print(f"{idx + 1}: {pct}")


if __name__ == "__main__":
    main()
