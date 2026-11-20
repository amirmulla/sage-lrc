"""Simulation runners, decoder wrappers, and post-processing utilities."""

import argparse
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tamo_barg_lrc.simulation.postprocess import process_error_distribution_csv
    from tamo_barg_lrc.simulation.runner import (
        DecodeOutcome,
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

VALID_DECODER_NAMES: tuple[str, ...] = (
    "GlobalDecoder",
    "GlobalErasureErrorDecoder",
    "IterativeDecoder",
    "IterativeErasureErrorDecoder",
    "TwoStepsDecoder",
    "TwoStepsErasureErrorDecoder",
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

_EXPORT_TO_MODULE: dict[str, str] = {
    "process_error_distribution_csv": "tamo_barg_lrc.simulation.postprocess",
    "DecodeOutcome": "tamo_barg_lrc.simulation.runner",
    "build_candidate_error_positions": "tamo_barg_lrc.simulation.runner",
    "build_decoder": "tamo_barg_lrc.simulation.runner",
    "compute_code_weight_distribution": "tamo_barg_lrc.simulation.runner",
    "compute_print_frequency": "tamo_barg_lrc.simulation.runner",
    "invoke_decoder": "tamo_barg_lrc.simulation.runner",
    "run_decoding_success_simulation": "tamo_barg_lrc.simulation.runner",
    "run_error_weight_simulation": "tamo_barg_lrc.simulation.runner",
    "run_global_threshold_simulation": "tamo_barg_lrc.simulation.runner",
    "run_weight_distribution_analysis": "tamo_barg_lrc.simulation.runner",
    "sample_fixed_weight_error": "tamo_barg_lrc.simulation.runner",
    "seed_simulation_rngs": "tamo_barg_lrc.simulation.runner",
    "write_code_metadata_csv": "tamo_barg_lrc.simulation.runner",
}


def add_common_simulation_args(
    parser: argparse.ArgumentParser,
    *,
    q: int,
    n: int,
    k: int | None,
    r: tuple[int, int],
    local_min_dist: tuple[int, int],
    subgroup_types: tuple[str, str],
    decoder: str | None,
    sim_itr: int,
    max_err: int,
    max_itr: int = 10,
    use_enc: bool = False,
) -> None:
    """Register standard code, decoder, and Monte Carlo CLI options on ``parser``.

    Args:
        parser: Target ``argparse.ArgumentParser`` instance.
        q: Default field cardinality.
        n: Default codeword length.
        k: Default message dimension (or ``None`` for ``max_dimension()``).
        r: Default two-partition locality pair ``(r_1, r_2)``.
        local_min_dist: Default local minimum distance pair ``(d_1, d_2)``.
        subgroup_types: Default subgroup types ``(type_1, type_2)``.
        decoder: Default decoder key, or ``None`` when controlled by a secondary flag.
        sim_itr: Default number of Monte Carlo trials per error weight.
        max_err: Default maximum error weight to simulate.
        max_itr: Default maximum iterations per trial.
        use_enc: Default flag for encoding random messages vs. the zero codeword.
    """
    parser.add_argument("--q", type=int, default=q, help="Finite field size q.")
    parser.add_argument("--n", type=int, default=n, help="Codeword length n.")
    parser.add_argument(
        "--k",
        type=int,
        default=k,
        help="Message dimension k (defaults to max_dimension if None).",
    )
    parser.add_argument(
        "--r",
        nargs=2,
        type=int,
        default=list(r),
        metavar=("R1", "R2"),
        help="Localities of the two partitions.",
    )
    parser.add_argument(
        "--local-min-dist",
        nargs=2,
        type=int,
        default=list(local_min_dist),
        metavar=("D1", "D2"),
        help="Local minimum distances of the two partitions.",
    )
    parser.add_argument(
        "--subgroup-types",
        nargs=2,
        choices=("add", "mult"),
        default=list(subgroup_types),
        metavar=("T1", "T2"),
        help="Subgroup types ('add' or 'mult') for the two partitions.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=VALID_DECODER_NAMES,
        default=decoder,
        help="Registered two-set decoder name.",
    )
    parser.add_argument(
        "--sim-itr",
        type=int,
        default=sim_itr,
        help="Number of Monte Carlo trials per error weight.",
    )
    parser.add_argument(
        "--max-err",
        type=int,
        default=max_err,
        help="Maximum injected error weight to simulate.",
    )
    parser.add_argument(
        "--max-itr",
        type=int,
        default=max_itr,
        help="Maximum number of iterations for iterative/two-step decoders.",
    )
    parser.add_argument(
        "--erasure-start-itr",
        type=int,
        default=None,
        help="Iteration index at which DUE erasure marking begins.",
    )
    parser.add_argument(
        "--use-enc",
        action=argparse.BooleanOptionalAction,
        default=use_enc,
        help="Encode a random message instead of using the all-zero codeword.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for CSV, graph PNG, and log files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible Python and SageMath sampling.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip rendering and saving the bipartite coset-graph PNG.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel Monte Carlo trials (default: 1).",
    )
    parser.add_argument(
        "--stop-after-zero-runs",
        type=int,
        default=None,
        help=(
            "Stop early after this many consecutive error weights observe zero "
            "successes/decreases (disabled by default)."
        ),
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve and cache public simulation exports per PEP 562."""
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return sorted module attributes including lazy PEP 562 exports."""
    return sorted({*globals().keys(), *__all__})
