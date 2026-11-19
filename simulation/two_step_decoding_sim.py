"""CLI entrypoint for two-step decoding success and iteration-count simulations."""

import argparse
from collections.abc import Sequence

from tamo_barg_lrc.simulation import add_common_simulation_args, run_decoding_success_simulation


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for ``two_step_decoding_sim.py``."""
    parser = argparse.ArgumentParser(
        description="Simulate two-step (iterative + global) decoding success probabilities."
    )
    add_common_simulation_args(
        parser,
        q=64,
        n=64,
        k=6,
        r=(2, 4),
        local_min_dist=(3, 4),
        subgroup_types=("add", "mult"),
        decoder="TwoStepsDecoder",
        sim_itr=5000,
        max_err=125,
        max_itr=10,
        use_enc=False,
    )
    parser.add_argument(
        "--print-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-trial error and decoding logs to *_log.txt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI flags and execute the two-step decoding simulation."""
    parser = build_parser()
    args = parser.parse_args(argv)

    run_decoding_success_simulation(
        q=args.q,
        n=args.n,
        k=args.k,
        locality=args.r,
        local_minimum_distance=args.local_min_dist,
        sub_group_type=args.subgroup_types,
        decoder_name=args.decoder,
        sim_itr=args.sim_itr,
        max_err=args.max_err,
        max_num_of_itr=args.max_itr,
        erasure_start_itr=args.erasure_start_itr,
        use_enc=args.use_enc,
        print_log=args.print_log,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
