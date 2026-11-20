"""CLI entrypoint for uniform vs. biased coset-expander error-weight simulations."""

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``expander_sim.py``."""
    from tamo_barg_lrc.simulation import add_common_simulation_args

    parser = argparse.ArgumentParser(
        description="Simulate uniform or biased error distributions on two-set expander graphs."
    )
    add_common_simulation_args(
        parser,
        q=256,
        n=256,
        k=6,
        r=(4, 4),
        local_min_dist=(5, 5),
        subgroup_types=("add", "add"),
        decoder="IterativeDecoder",
        sim_itr=10000,
        max_err=64,
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
        "--biased",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict injected errors to the prefix [0, error_boundary).",
    )
    parser.add_argument(
        "--error-boundary",
        "--t",
        dest="error_boundary",
        type=int,
        default=64,
        help="Upper coordinate bound t when --biased is active.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the expander error-weight simulation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from tamo_barg_lrc.simulation import run_error_weight_simulation

    run_error_weight_simulation(
        q=args.q,
        n=args.n,
        k=args.k,
        locality=args.r,
        local_minimum_distance=args.local_min_dist,
        sub_group_type=args.subgroup_types,
        decoder_name=args.decoder,
        sim_itr=args.sim_itr,
        max_err=args.max_err,
        start_err=args.start_err,
        max_num_of_itr=args.max_itr,
        erasure_start_itr=args.erasure_start_itr,
        use_enc=args.use_enc,
        output_dir=args.output_dir,
        seed=args.seed,
        biased=bool(args.biased),
        error_boundary=args.error_boundary,
        plot_graph=not args.no_plot,
        workers=args.workers,
        stop_after_zero_runs=args.stop_after_zero_runs,
    )


if __name__ == "__main__":
    main()
