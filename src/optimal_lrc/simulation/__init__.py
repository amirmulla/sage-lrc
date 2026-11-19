"""Simulation runners, decoder wrappers, and post-processing utilities."""

from optimal_lrc.simulation.postprocess import process_error_distribution_csv
from optimal_lrc.simulation.runner import (
    VALID_DECODER_NAMES,
    DecodeOutcome,
    add_common_simulation_args,
    build_candidate_error_positions,
    build_decoder,
    compute_code_weight_distribution,
    compute_print_frequency,
    invoke_decoder,
    run_decoding_success_simulation,
    run_error_weight_simulation,
    run_global_threshold_simulation,
    run_weight_distribution_analysis,
    sample_fixed_weight_error,
    seed_simulation_rngs,
    write_code_metadata_csv,
)

__all__ = [
    "DecodeOutcome",
    "VALID_DECODER_NAMES",
    "add_common_simulation_args",
    "build_candidate_error_positions",
    "build_decoder",
    "compute_code_weight_distribution",
    "compute_print_frequency",
    "invoke_decoder",
    "process_error_distribution_csv",
    "run_decoding_success_simulation",
    "run_error_weight_simulation",
    "run_global_threshold_simulation",
    "run_weight_distribution_analysis",
    "sample_fixed_weight_error",
    "seed_simulation_rngs",
    "write_code_metadata_csv",
]
