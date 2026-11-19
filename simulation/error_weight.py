"""CLI entrypoint for residual error-weight transition simulations."""

import argparse
from collections.abc import Sequence

from optimal_lrc.simulation import add_common_simulation_args, run_error_weight_simulation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``error_weight.py``."""
    parser = argparse.ArgumentParser(
        description="Simulate residual error-weight transitions for two-set Tamo-Barg decoders."
    )
    add_common_simulation_args(
        parser,
        q=64,
        n=64,
        k=6,
        r=(2, 2),
        local_min_dist=(3, 3),
        subgroup_types=("add", "add"),
        decoder="IterativeDecoder",
        sim_itr=10,
        max_err=16,
        max_itr=10,
        use_enc=False,
    )
    parser.add_argument(
        "--start-err",
        type=int,
        default=1,
        help="Minimum injected error weight to simulate.",
    )
    parser.add_argument(
        "--exact-distance",
        type=int,
        default=None,
        help="Optional exact code distance override (sets max_err to exact_distance + 1).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the error-weight simulation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    max_err = args.exact_distance + 1 if args.exact_distance is not None else args.max_err

    run_error_weight_simulation(
        q=args.q,
        n=args.n,
        k=args.k,
        locality=args.r,
        local_minimum_distance=args.local_min_dist,
        sub_group_type=args.subgroup_types,
        decoder_name=args.decoder,
        sim_itr=args.sim_itr,
        max_err=max_err,
        start_err=args.start_err,
        max_num_of_itr=args.max_itr,
        erasure_start_itr=args.erasure_start_itr,
        use_enc=args.use_enc,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
