"""CLI entrypoint for iterative two-set decoding success and iteration-count simulations."""

import argparse
from collections.abc import Sequence

from optimal_lrc.simulation import add_common_simulation_args, run_decoding_success_simulation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``iterative_decoding_sim.py``."""
    parser = argparse.ArgumentParser(
        description="Simulate iterative decoding success probabilities and average iterations."
    )
    add_common_simulation_args(
        parser,
        q=256,
        n=256,
        k=None,
        r=(14, 3),
        local_min_dist=(3, 3),
        subgroup_types=("add", "mult"),
        decoder=None,
        sim_itr=5,
        max_err=3,
        max_itr=10,
        use_enc=True,
    )
    parser.add_argument(
        "--use-erasure-decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use IterativeErasureErrorDecoder when --decoder is not explicitly specified.",
    )
    parser.add_argument(
        "--print-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-trial error and decoding logs to *_log.txt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the iterative decoding simulation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.decoder is not None:
        decoder_name = args.decoder
    elif args.use_erasure_decoder:
        decoder_name = "IterativeErasureErrorDecoder"
    else:
        decoder_name = "IterativeDecoder"

    run_decoding_success_simulation(
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
        print_log=args.print_log,
        output_dir=args.output_dir,
        seed=args.seed,
        legacy_iter_dir_naming=True,
    )


if __name__ == "__main__":
    main()
