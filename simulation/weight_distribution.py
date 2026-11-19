"""CLI entrypoint for computing exact weight distributions of two-set Tamo-Barg codes."""

import argparse
from collections.abc import Sequence

from optimal_lrc.simulation import run_weight_distribution_analysis


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``weight_distribution.py``."""
    parser = argparse.ArgumentParser(
        description="Compute the exact Hamming weight distribution and minimum distance."
    )
    parser.add_argument("--q", type=int, default=64, help="Finite field size q.")
    parser.add_argument("--n", type=int, default=64, help="Codeword length n.")
    parser.add_argument("--k", type=int, default=6, help="Message dimension k.")
    parser.add_argument(
        "--r",
        nargs=2,
        type=int,
        default=[2, 5],
        metavar=("R1", "R2"),
        help="Localities of the two partitions.",
    )
    parser.add_argument(
        "--local-min-dist",
        nargs=2,
        type=int,
        default=[3, 3],
        metavar=("D1", "D2"),
        help="Local minimum distances of the two partitions.",
    )
    parser.add_argument(
        "--subgroup-types",
        nargs=2,
        choices=("add", "mult"),
        default=["add", "mult"],
        metavar=("T1", "T2"),
        help="Subgroup types ('add' or 'mult') for the two partitions.",
    )
    parser.add_argument(
        "--add-idx",
        type=int,
        default=None,
        help="Optional index into enumerated additive subgroups overriding canonical selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for weight distribution CSV and graph PNG.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the weight distribution computation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    run_weight_distribution_analysis(
        q=args.q,
        n=args.n,
        k=args.k,
        locality=args.r,
        local_minimum_distance=args.local_min_dist,
        sub_group_type=args.subgroup_types,
        add_idx=args.add_idx,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
