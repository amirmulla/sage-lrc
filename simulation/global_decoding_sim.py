"""CLI entrypoint for global-cleanup threshold and residual error distribution simulations."""

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``global_decoding_sim.py``."""
    from tamo_barg_lrc.simulation import add_common_simulation_args

    parser = argparse.ArgumentParser(
        description=(
            "Simulate iterative decoding followed by global unique-decoding threshold "
            "evaluation and export residual error-weight distributions."
        )
    )
    add_common_simulation_args(
        parser,
        q=256,
        n=256,
        k=6,
        r=(2, 4),
        local_min_dist=(3, 5),
        subgroup_types=("add", "add"),
        decoder=None,
        sim_itr=5000,
        max_err=250,
        max_itr=10,
        use_enc=False,
    )
    parser.add_argument(
        "--use-erasure-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use IterativeErasureErrorDecoder when --decoder is not explicitly specified.",
    )
    parser.add_argument(
        "--exact-distance",
        type=int,
        default=None,
        help="Optional exact code minimum distance override for the global threshold.",
    )
    parser.add_argument(
        "--print-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-trial error and decoding logs to *_log.txt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the global-cleanup threshold simulation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.decoder is not None:
        decoder_name = args.decoder
    elif args.use_erasure_decoder:
        decoder_name = "IterativeErasureErrorDecoder"
    else:
        decoder_name = "IterativeDecoder"

    from tamo_barg_lrc.simulation import run_global_threshold_simulation

    run_global_threshold_simulation(
        q=args.q,
        n=args.n,
        k=args.k,
        locality=args.r,
        local_minimum_distance=args.local_min_dist,
        sub_group_type=args.subgroup_types,
        decoder_name=decoder_name,
        sim_itr=args.sim_itr,
        max_err=args.max_err,
        max_num_of_itr=args.max_itr,
        erasure_start_itr=args.erasure_start_itr,
        use_enc=args.use_enc,
        exact_distance=args.exact_distance,
        print_log=args.print_log,
        output_dir=args.output_dir,
        seed=args.seed,
        plot_graph=not args.no_plot,
        workers=args.workers,
        stop_after_zero_runs=args.stop_after_zero_runs,
    )


if __name__ == "__main__":
    main()
