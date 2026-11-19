"""Reusable simulation runners, decoder dispatchers, and CLI utilities."""

import argparse
import csv
import random
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sage.coding.channel import random_error_vector
from sage.coding.decoder import DecodingError
from sage.misc.randstate import set_random_seed
from sage.modules.free_module import VectorSpace
from sage.modules.free_module_element import zero_vector
from sage.rings.finite_rings.finite_field_constructor import GF

from optimal_lrc.expander import compute_expansion_metrics, save_bipartite_graph_plot
from optimal_lrc.subgroups import SubgroupType, find_additive_subgroups
from optimal_lrc.tamo_barg_two_sets import TamoBargCodeTwoSets

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
    "run_decoding_success_simulation",
    "run_error_weight_simulation",
    "run_global_threshold_simulation",
    "run_weight_distribution_analysis",
    "sample_fixed_weight_error",
    "seed_simulation_rngs",
    "write_code_metadata_csv",
]


@dataclass(frozen=True)
class DecodeOutcome:
    """Normalized decoding outcome across all two-set Tamo-Barg decoders.

    Attributes:
        codeword: Decoded codeword vector in ``GF(q)^n``.
        num_iterations: Number of decoder iterations executed (``1`` for global).
        remaining_erasures: Count of unresolved DUE erasures after decoding.
    """

    codeword: Any
    num_iterations: int
    remaining_erasures: int = 0


def build_decoder(
    code: TamoBargCodeTwoSets,
    decoder_name: str,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
) -> Any:
    """Instantiate a registered two-set Tamo-Barg decoder with matching arguments.

    Args:
        code: Two-set Tamo-Barg code instance.
        decoder_name: Registered decoder key in ``VALID_DECODER_NAMES``.
        max_num_of_itr: Maximum iterations for iterative and two-step decoders.
        erasure_start_itr: Optional iteration index at which DUE erasure marking
            starts for erasure-and-error iterative/two-step decoders.

    Returns:
        Configured SageMath ``Decoder`` instance.

    Raises:
        ValueError: If ``decoder_name`` is not a recognized two-set decoder name.
    """
    if decoder_name in ("GlobalDecoder", "GlobalErasureErrorDecoder"):
        return code.decoder(decoder_name)
    if decoder_name in ("IterativeDecoder", "TwoStepsDecoder"):
        return code.decoder(decoder_name, max_num_of_itr)
    if decoder_name in ("IterativeErasureErrorDecoder", "TwoStepsErasureErrorDecoder"):
        return code.decoder(decoder_name, max_num_of_itr, erasure_start_itr)
    raise ValueError(
        f"Unknown decoder_name {decoder_name!r}; expected one of {VALID_DECODER_NAMES}"
    )


def invoke_decoder(
    decoder: Any,
    decoder_name: str,
    received_word: Any,
) -> DecodeOutcome:
    """Execute ``decoder`` on ``received_word`` and normalize its return value.

    Args:
        decoder: Instantiated two-set Tamo-Barg decoder.
        decoder_name: Registered decoder key matching ``decoder``.
        received_word: Received vector in ``GF(q)^n``.

    Returns:
        Normalized :class:`DecodeOutcome` containing the decoded vector, iteration
        count, and remaining erasure weight.

    Raises:
        ValueError: If ``decoder_name`` is not a recognized two-set decoder name.
    """
    if decoder_name == "GlobalDecoder":
        try:
            corr = decoder.decode_to_code(received_word)
        except (DecodingError, ValueError):
            corr = copy(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=1, remaining_erasures=0)

    if decoder_name == "GlobalErasureErrorDecoder":
        era_vec = zero_vector(GF(2), len(received_word))
        try:
            corr = decoder.decode_to_code((received_word, era_vec))
        except (DecodingError, ValueError):
            corr = copy(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=1, remaining_erasures=0)

    if decoder_name in ("IterativeDecoder", "TwoStepsDecoder"):
        corr, num_itr = decoder.decode_to_code(received_word)
        return DecodeOutcome(codeword=corr, num_iterations=int(num_itr), remaining_erasures=0)

    if decoder_name == "IterativeErasureErrorDecoder":
        (corr, era_vec), num_itr = decoder.decode_to_code(received_word)
        return DecodeOutcome(
            codeword=corr,
            num_iterations=int(num_itr),
            remaining_erasures=int(era_vec.hamming_weight()),
        )

    if decoder_name == "TwoStepsErasureErrorDecoder":
        corr, num_itr, rem_era = decoder.decode_to_code(received_word)
        return DecodeOutcome(
            codeword=corr,
            num_iterations=int(num_itr),
            remaining_erasures=int(rem_era),
        )

    raise ValueError(
        f"Unknown decoder_name {decoder_name!r}; expected one of {VALID_DECODER_NAMES}"
    )


def seed_simulation_rngs(seed: int | None) -> None:
    """Seed both Python and SageMath pseudo-random number generators if ``seed`` is set.

    Args:
        seed: Optional integer seed; if ``None``, RNG state is left unchanged.
    """
    if seed is not None:
        random.seed(int(seed))
        set_random_seed(int(seed))


def compute_print_frequency(sim_itr: int, print_freq_factor: int = 5) -> int:
    """Return a positive progress-log interval that never divides by zero.

    Args:
        sim_itr: Total number of Monte Carlo trials per error weight.
        print_freq_factor: Target number of progress messages per error weight.

    Returns:
        Positive integer step size ``>= 1``.
    """
    return max(1, int(sim_itr) // max(1, int(print_freq_factor)))


def build_candidate_error_positions(
    code: TamoBargCodeTwoSets,
    base_positions: Sequence[int] | None = None,
    shorten: bool | None = None,
) -> list[int]:
    """Build valid codeword error-injection indices, excluding ``0`` when shortened.

    In mixed additive/multiplicative codes (``sub_group_type[0] != sub_group_type[1]``)
    of length ``q``, the evaluation point ``0`` belongs only to the additive partition.
    Excluding its coordinate upfront ensures every sampled error pattern has exact
    requested Hamming weight without post-hoc overwriting.

    Args:
        code: Two-set Tamo-Barg code instance.
        base_positions: Optional candidate coordinate subset; defaults to ``0 .. n - 1``.
        shorten: Whether to exclude the coordinate of ``0``; defaults to
            ``sub_group_type[0] != sub_group_type[1]``.

    Returns:
        Ordered list of distinct candidate coordinate indices in ``0 .. n - 1``.
    """
    length = int(code.length())
    _, sg_types = code.sub_group()
    if shorten is None:
        shorten = sg_types[0] != sg_types[1]

    raw_positions = range(length) if base_positions is None else base_positions
    candidates = list(dict.fromkeys(int(pos) for pos in raw_positions if 0 <= int(pos) < length))

    if shorten:
        zero_elem = code.base_field().zero()
        eval_pts = code.evaluation_points()
        if zero_elem in eval_pts:
            zero_idx = eval_pts.index(zero_elem)
            candidates = [pos for pos in candidates if pos != zero_idx]

    return candidates


def sample_fixed_weight_error(
    field: Any,
    length: int,
    weight: int,
    candidate_positions: Sequence[int],
) -> Any:
    """Sample a random error vector of exact Hamming ``weight`` on ``candidate_positions``.

    Args:
        field: Finite field ``GF(q)`` over which the error vector is defined.
        length: Ambient vector length ``n``.
        weight: Exact Hamming weight (number of non-zero coordinates) to inject.
        candidate_positions: Sequence of distinct valid coordinate indices in ``0 .. n - 1``.

    Returns:
        Vector in ``GF(q)^n`` with uniform random non-zero elements at ``weight``
        uniformly sampled positions from ``candidate_positions``.

    Raises:
        ValueError: If ``candidate_positions`` contains out-of-bounds indices or
            ``weight`` is outside ``0 .. len(unique_positions)``.
    """
    unique_positions = list(dict.fromkeys(int(pos) for pos in candidate_positions))
    if any(pos < 0 or pos >= length for pos in unique_positions):
        raise ValueError(f"candidate_positions must lie in 0 .. {length - 1}")
    if weight < 0 or weight > len(unique_positions):
        raise ValueError(
            f"Requested error weight {weight} outside valid range 0 .. {len(unique_positions)}"
        )

    chosen_positions = random.sample(unique_positions, weight)
    return random_error_vector(length, field, chosen_positions)


def _code_metadata_rows(
    code: TamoBargCodeTwoSets,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
) -> list[list[Any]]:
    """Build standardized metadata key-value rows for CSV and console reporting."""
    subgroups, sg_types = code.sub_group()
    localities = code.locality()
    local_dists = code.local_minimum_distance()
    q = int(code.base_field().cardinality())

    rows: list[list[Any]] = [
        ["GF: ", q],
        ["Code dim n: ", int(code.length())],
        ["Message dim k: ", int(code.dimension())],
        ["k_max: ", code.max_dimension()],
        ["max rate: ", code.max_rate()],
        ["Locality r1: ", localities[0]],
        ["Locality r2: ", localities[1]],
        ["Local Minimum distance d1: ", local_dists[0]],
        ["Local Minimum distance d2: ", local_dists[1]],
        ["Design Distance: ", code.design_distance()],
        ["First Subgroup: ", subgroups[0]],
        ["First Subgroup Type: ", sg_types[0]],
        ["First Subgroup Size: ", len(subgroups[0])],
        ["Second Subgroup: ", subgroups[1]],
        ["Second Subgroup Type: ", sg_types[1]],
        ["Second Subgroup Size: ", len(subgroups[1])],
        ["Evaluation Points: ", code.evaluation_points()],
        ["Partitions Graph is Connected:", graph_metrics["is_connected"]],
        ["Number of disjoint subgraphs:", graph_metrics["num_connected_components"]],
        ["Partitions Graph Edge Expansion:", graph_metrics["edge_expansion"]],
    ]
    if codeword is not None:
        rows.append(["Codeword : ", codeword])
    if extra_rows:
        rows.extend(list(row) for row in extra_rows)
    return rows


def write_code_metadata_csv(
    writer: Any,
    code: TamoBargCodeTwoSets,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
) -> None:
    """Write standardized code and coset-graph metadata rows to a CSV writer.

    Args:
        writer: Active ``csv.writer`` object.
        code: Two-set Tamo-Barg code instance.
        graph_metrics: Expansion/connectivity metrics dictionary from
            :func:`compute_expansion_metrics`.
        codeword: Optional reference codeword vector included in the header.
        extra_rows: Optional additional ``[label, value]`` rows appended after
            standard metadata.
    """
    for row in _code_metadata_rows(code, graph_metrics, codeword, extra_rows):
        writer.writerow(row)


def _print_code_summary(
    code: TamoBargCodeTwoSets,
    sim_name: str,
    graph_metrics: Mapping[str, Any],
    codeword: Any | None = None,
    extra_rows: Sequence[Sequence[Any]] | None = None,
    stream: Any = None,
    include_banner: bool = True,
) -> None:
    """Print human-readable simulation setup metadata to stdout or a log stream."""
    if include_banner:
        print("## Setup Done ##", file=stream)
        print(f"## Start Simulation {sim_name} ##", file=stream)
    print(code, file=stream)
    for label, value in _code_metadata_rows(code, graph_metrics, codeword, extra_rows):
        print(f"{label}{value}", file=stream)


def _build_sim_name(code: TamoBargCodeTwoSets, suffix: str = "") -> str:
    """Construct canonical simulation identifier ``GF_<q>_<t0>_<s0>_<t1>_<s1>[suffix]``."""
    subgroups, sg_types = code.sub_group()
    q = int(code.base_field().cardinality())
    base = f"GF_{q}_{sg_types[0]}_{len(subgroups[0])}_{sg_types[1]}_{len(subgroups[1])}"
    return f"{base}{suffix}"


def _prepare_code_and_codeword(
    q: int,
    n: int,
    k: int | None,
    locality: Sequence[int],
    local_minimum_distance: Sequence[int],
    sub_group_type: Sequence[Literal["add", "mult"]],
    use_enc: bool,
    subgroup: Sequence[Sequence[Any] | None] | None = None,
) -> tuple[Any, TamoBargCodeTwoSets, Any]:
    """Construct ``GF(q)``, ``TamoBargCodeTwoSets``, and initial transmitted codeword."""
    field = GF(int(q), repr="int")
    sgt_tuple: list[SubgroupType] = [
        "add" if sub_group_type[0] == "add" else "mult",
        "add" if sub_group_type[1] == "add" else "mult",
    ]
    if k is None:
        probe = TamoBargCodeTwoSets(
            field,
            int(n),
            1,
            locality,
            local_minimum_distance,
            sgt_tuple,
            shift_add=False,
            subgroup=subgroup,
        )
        resolved_k = probe.max_dimension()
    else:
        resolved_k = int(k)

    code = TamoBargCodeTwoSets(
        field,
        int(n),
        resolved_k,
        locality,
        local_minimum_distance,
        sgt_tuple,
        shift_add=False,
        subgroup=subgroup,
    )
    if use_enc:
        msg_space = VectorSpace(field, resolved_k)
        message = msg_space.random_element()
        codeword = code.encoder("VectorEncoder").encode(message)
    else:
        codeword = zero_vector(field, int(n))

    return field, code, codeword


def compute_code_weight_distribution(code: TamoBargCodeTwoSets) -> list[int]:
    """Compute exact Hamming weight distribution ``[A_0, A_1, ..., A_n]`` of ``code``.

    Uses SageMath's native ``code.weight_distribution()`` when GAP/libgap is
    available, and falls back seamlessly to projective (monic) message-space
    enumeration achieving a ``(q - 1)x`` speedup over naive enumeration.

    Args:
        code: Two-set Tamo-Barg code instance.

    Returns:
        List of length ``n + 1`` where entry ``w`` is the number of codewords of
        Hamming weight ``w``.
    """
    try:
        return [int(val) for val in code.weight_distribution()]
    except (ImportError, ModuleNotFoundError, NotImplementedError):
        pass

    length = int(code.length())
    field = code.base_field()
    q = int(field.cardinality())
    counts = [0] * (length + 1)
    counts[0] = 1

    echelon_mat = code.generator_matrix().echelon_form()
    non_zero_row_indices = [idx for idx, row in enumerate(echelon_mat.rows()) if not row.is_zero()]
    rank = len(non_zero_row_indices)
    if rank == 0:
        return counts

    basis_mat = echelon_mat.matrix_from_rows(non_zero_row_indices)
    scalar_mult = q - 1

    for lead in range(rank):
        lead_row = basis_mat.row(lead)
        tail_dim = rank - lead - 1
        if tail_dim == 0:
            wt = int(lead_row.hamming_weight())
            counts[wt] += scalar_mult
            continue
        sub_gen = basis_mat.matrix_from_rows(list(range(lead + 1, rank)))
        for tail_vec in VectorSpace(field, tail_dim):
            cw = lead_row + tail_vec * sub_gen
            wt = int(cw.hamming_weight())
            counts[wt] += scalar_mult

    return counts


def run_error_weight_simulation(
    *,
    q: int = 64,
    n: int = 64,
    k: int = 6,
    locality: Sequence[int] = (2, 2),
    local_minimum_distance: Sequence[int] = (3, 3),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
    decoder_name: str = "IterativeDecoder",
    sim_itr: int = 10,
    max_err: int = 16,
    start_err: int = 1,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
    biased: bool | None = None,
    error_boundary: int | None = None,
    candidate_positions: Sequence[int] | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    """Run residual error-weight transition simulation across injected error weights.

    Shared simulation engine for ``simulation/error_weight.py`` and
    ``simulation/expander_sim.py``.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        start_err: Starting error weight (default ``1``).
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.
        biased: If ``True``, restrict errors to ``0 .. error_boundary - 1``. If
            ``None``, standard ``error_weight`` naming is used; if ``bool``,
            expander naming (``_expander_biased_`` / ``_expander_uniform_``) is used.
        error_boundary: Upper bound ``t`` for biased error support ``[0, t)``.
        candidate_positions: Optional explicit coordinate subset for error injection.
        run_label: Optional custom subdirectory suffix override.

    Returns:
        Dictionary with paths (``csv_path``, ``graph_path``) and recorded ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = compute_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    if run_label is not None:
        dir_name = run_label
    elif biased is None:
        dir_name = f"{sim_name}_{decoder_name}"
    elif biased:
        dir_name = f"{sim_name}_expander_biased_{decoder_name}"
    else:
        dir_name = f"{sim_name}_expander_uniform_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    graph_path = res_dir / f"{sim_name}_graph.png"

    save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)

    if candidate_positions is not None:
        base_pos: Sequence[int] | None = candidate_positions
    elif biased:
        bound = min(int(n), int(error_boundary) if error_boundary is not None else int(n))
        base_pos = range(max(0, bound))
    else:
        base_pos = None

    error_positions = build_candidate_error_positions(code, base_pos)
    effective_max_err = min(int(n) - int(code.dimension()), int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)
    design_radius = (code.design_distance() - 1) // 2

    _print_code_summary(code, sim_name, graph_metrics, codeword)
    print(f"error_positions: {error_positions}")

    recorded_rows: list[list[Any]] = []
    with open(csv_path, "w", newline="") as res_file:
        writer = csv.writer(res_file)
        write_code_metadata_csv(writer, code, graph_metrics, codeword)
        writer.writerow(
            [
                "Num_of_error_symbols",
                "probability_of_error_increase",
                "num_of_error_increase",
                "probability_of_error_no_change",
                "num_of_error_no_change",
                "probability_of_error_decrease",
                "num_of_error_decrease",
                "probability_of_failure",
                "num_of_failure",
                "avg_remain_err_weight",
                "min_remain_err_weight",
                "max_remain_err_weight",
            ]
        )

        for n_err in range(int(start_err), effective_max_err + 1):
            print(f"Error Weight: {n_err}")
            twostep_fail_itr = 0
            iter_error_incr_itr = 0
            iter_error_decr_itr = 0
            iter_error_no_change = 0
            overall_remain_err_weight = 0
            max_remain_err_weight = 0
            min_remain_err_weight = int(n)

            for i in range(sim_itr):
                err_vec = sample_fixed_weight_error(field, int(n), n_err, error_positions)
                received = codeword + err_vec
                outcome = invoke_decoder(decoder, decoder_name, received)
                remain_err_weight = int((codeword - outcome.codeword).hamming_weight())

                if remain_err_weight > max_remain_err_weight:
                    max_remain_err_weight = remain_err_weight
                if remain_err_weight < min_remain_err_weight:
                    min_remain_err_weight = remain_err_weight

                overall_remain_err_weight += remain_err_weight

                if remain_err_weight > n_err:
                    iter_error_incr_itr += 1
                elif remain_err_weight < n_err:
                    iter_error_decr_itr += 1
                else:
                    iter_error_no_change += 1

                if n_err <= design_radius and remain_err_weight > design_radius:
                    twostep_fail_itr += 1

                if (i + 1) % print_freq == 0:
                    print(f"Iteration {i + 1} / {sim_itr} Done.")

            row = [
                n_err,
                100.0 * (iter_error_incr_itr / sim_itr),
                iter_error_incr_itr,
                100.0 * (iter_error_no_change / sim_itr),
                iter_error_no_change,
                100.0 * (iter_error_decr_itr / sim_itr),
                iter_error_decr_itr,
                100.0 * (twostep_fail_itr / sim_itr),
                twostep_fail_itr,
                overall_remain_err_weight / sim_itr,
                min_remain_err_weight,
                max_remain_err_weight,
            ]
            writer.writerow(row)
            recorded_rows.append(row)

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "rows": recorded_rows,
    }


def run_global_threshold_simulation(
    *,
    q: int = 256,
    n: int = 256,
    k: int = 6,
    locality: Sequence[int] = (2, 4),
    local_minimum_distance: Sequence[int] = (3, 5),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "add"),
    decoder_name: str = "IterativeDecoder",
    sim_itr: int = 5000,
    max_err: int = 250,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    exact_distance: int | None = None,
    print_log: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
) -> dict[str, Any]:
    """Run global-cleanup threshold simulation and record residual weight distributions.

    Counts a trial as successful when residual error weight after decoding does
    not exceed ``(target_distance - 1) // 2``, writing both summary success rates
    and ``*_err.csv`` residual error-weight histograms consumed by
    :func:`optimal_lrc.simulation.postprocess.process_error_distribution_csv`.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        exact_distance: Optional override distance replacing ``code.design_distance()``.
        print_log: If ``True``, write detailed trial logs to ``*_log.txt``.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.

    Returns:
        Dictionary with paths (``csv_path``, ``err_csv_path``, ``graph_path``,
        ``log_path``) and recorded ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = compute_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    uses_erasure = "Erasure" in decoder_name
    if decoder_name == "IterativeDecoder":
        dir_name = sim_name
    elif decoder_name == "IterativeErasureErrorDecoder":
        dir_name = f"{sim_name}_erasure"
    else:
        dir_name = f"{sim_name}_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    err_csv_path = res_dir / f"{sim_name}_err.csv"
    graph_path = res_dir / f"{sim_name}_graph.png"
    log_path: Path | None = (res_dir / f"{sim_name}_log.txt") if print_log else None

    save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)

    extra_rows: list[list[Any]] = [["Erasure Decoder: ", uses_erasure]]
    if exact_distance is not None:
        extra_rows.append(["Code Distance: ", int(exact_distance)])

    _print_code_summary(code, sim_name, graph_metrics, codeword, extra_rows=extra_rows)

    error_positions = build_candidate_error_positions(code)
    effective_max_err = min(int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)
    target_distance = int(exact_distance) if exact_distance is not None else code.design_distance()
    dd_th = (target_distance - 1) // 2

    recorded_rows: list[list[Any]] = []
    with ExitStack() as stack:
        res_file = stack.enter_context(open(csv_path, "w", newline=""))
        err_file = stack.enter_context(open(err_csv_path, "w", newline=""))
        log_handle = (
            stack.enter_context(open(log_path, "w", encoding="utf-8"))
            if log_path is not None
            else None
        )

        writer = csv.writer(res_file)
        err_writer = csv.writer(err_file)

        write_code_metadata_csv(writer, code, graph_metrics, codeword, extra_rows=extra_rows)
        writer.writerow(["Num_of_error_symbols", "probability_of_success"])
        err_writer.writerow(["Num_of_error_symbols", "remain_err_weight", "count", "dist"])

        if log_handle is not None:
            _print_code_summary(
                code,
                sim_name,
                graph_metrics,
                codeword,
                extra_rows=extra_rows,
                stream=log_handle,
                include_banner=False,
            )

        for n_err in range(1, effective_max_err + 1):
            print(f"Error Weight: {n_err}")
            success_itr = 0
            remain_err_dist = [0] * (int(n) + 1)

            for i in range(sim_itr):
                err_vec = sample_fixed_weight_error(field, int(n), n_err, error_positions)
                received = codeword + err_vec
                outcome = invoke_decoder(decoder, decoder_name, received)
                correct_c = outcome.codeword

                if log_handle is not None:
                    print(f"Error              : {err_vec}", file=log_handle)
                    print(f"Corrected Codeword : {correct_c}", file=log_handle)
                    print(f"Correction Successful: {correct_c == codeword}", file=log_handle)

                if correct_c == codeword:
                    remain_wt = 0
                    success_itr += 1
                else:
                    remain_wt = int((codeword - correct_c).hamming_weight())
                    if remain_wt <= dd_th:
                        success_itr += 1

                remain_err_dist[remain_wt] += 1

                if (i + 1) % print_freq == 0:
                    print(f"Iteration {i + 1} / {sim_itr} Done.")

            succ_prob = 100.0 * (success_itr / sim_itr)
            row = [n_err, succ_prob]
            writer.writerow(row)
            recorded_rows.append(row)

            for wt, count in enumerate(remain_err_dist):
                if count != 0:
                    err_writer.writerow([n_err, wt, count, count / sim_itr])

            if log_handle is not None:
                print(f"Num of error symbols: {n_err}", file=log_handle)
                print(f"Probability of Success: {succ_prob}", file=log_handle)
                print("##########################", file=log_handle)

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "err_csv_path": err_csv_path,
        "graph_path": graph_path,
        "log_path": log_path,
        "rows": recorded_rows,
    }


def run_decoding_success_simulation(
    *,
    q: int = 64,
    n: int = 64,
    k: int | None = 6,
    locality: Sequence[int] = (2, 4),
    local_minimum_distance: Sequence[int] = (3, 4),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "mult"),
    decoder_name: str = "TwoStepsDecoder",
    sim_itr: int = 5000,
    max_err: int = 125,
    max_num_of_itr: int = 10,
    erasure_start_itr: int | None = None,
    use_enc: bool = False,
    print_log: bool = False,
    output_dir: str | Path = "./results",
    seed: int | None = None,
    print_freq_factor: int = 5,
    legacy_iter_dir_naming: bool = False,
) -> dict[str, Any]:
    """Run end-to-end decoding success rate and iteration-count simulation.

    Shared simulation engine for ``simulation/iterative_decoding_sim.py`` and
    ``simulation/two_step_decoding_sim.py``.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension, or ``None`` to use ``code.max_dimension()``.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        decoder_name: Registered two-set decoder name.
        sim_itr: Number of Monte Carlo iterations per error weight (``>= 1``).
        max_err: Maximum error weight to simulate.
        max_num_of_itr: Maximum iterations for iterative/two-step decoders.
        erasure_start_itr: Optional iteration threshold for DUE erasure marking.
        use_enc: Encode a random message if ``True``, else transmit the zero codeword.
        print_log: If ``True``, write detailed trial logs to ``*_log.txt``.
        output_dir: Parent directory under which run artifacts are written.
        seed: Optional integer RNG seed.
        print_freq_factor: Progress print frequency divisor.
        legacy_iter_dir_naming: Use ``<sim_name>`` / ``<sim_name>_erasure`` naming
            when ``decoder_name`` is ``IterativeDecoder`` / ``IterativeErasureErrorDecoder``.

    Returns:
        Dictionary with paths (``csv_path``, ``graph_path``, ``log_path``) and ``rows``.
    """
    if sim_itr < 1:
        raise ValueError(f"sim_itr must be >= 1, got {sim_itr}")

    seed_simulation_rngs(seed)
    field, code, codeword = _prepare_code_and_codeword(
        q, n, k, locality, local_minimum_distance, sub_group_type, use_enc
    )
    decoder = build_decoder(code, decoder_name, max_num_of_itr, erasure_start_itr)

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = compute_expansion_metrics(graph, right_nodes)
    sim_name = _build_sim_name(code)

    if legacy_iter_dir_naming and decoder_name == "IterativeDecoder":
        dir_name = sim_name
    elif legacy_iter_dir_naming and decoder_name == "IterativeErasureErrorDecoder":
        dir_name = f"{sim_name}_erasure"
    else:
        dir_name = f"{sim_name}_{decoder_name}"

    res_dir = Path(output_dir) / dir_name
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}.csv"
    graph_path = res_dir / f"{sim_name}_graph.png"
    log_path: Path | None = (res_dir / f"{sim_name}_log.txt") if print_log else None

    save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)
    _print_code_summary(code, sim_name, graph_metrics, codeword)

    error_positions = build_candidate_error_positions(code)
    effective_max_err = min(int(n) - int(code.dimension()), int(max_err), len(error_positions))
    print_freq = compute_print_frequency(sim_itr, print_freq_factor)

    recorded_rows: list[list[Any]] = []
    with ExitStack() as stack:
        res_file = stack.enter_context(open(csv_path, "w", newline=""))
        log_handle = (
            stack.enter_context(open(log_path, "w", encoding="utf-8"))
            if log_path is not None
            else None
        )

        writer = csv.writer(res_file)
        write_code_metadata_csv(writer, code, graph_metrics, codeword)
        writer.writerow(
            ["Num_of_error_symbols", "average_num_of_iteration", "probability_of_success"]
        )

        if log_handle is not None:
            _print_code_summary(
                code,
                sim_name,
                graph_metrics,
                codeword,
                stream=log_handle,
                include_banner=False,
            )

        for n_err in range(1, effective_max_err + 1):
            print(f"Error Weight: {n_err}")
            success_itr = 0
            overall_num_itr = 0

            for i in range(sim_itr):
                err_vec = sample_fixed_weight_error(field, int(n), n_err, error_positions)
                received = codeword + err_vec
                outcome = invoke_decoder(decoder, decoder_name, received)
                correct_c = outcome.codeword

                if log_handle is not None:
                    print(f"Error              : {err_vec}", file=log_handle)
                    print(f"Corrected Codeword : {correct_c}", file=log_handle)
                    print(f"Correction Successful: {correct_c == codeword}", file=log_handle)

                if correct_c == codeword:
                    success_itr += 1
                    overall_num_itr += outcome.num_iterations

                if (i + 1) % print_freq == 0:
                    print(f"Iteration {i + 1} / {sim_itr} Done.")

            avg_num_of_itr = (
                overall_num_itr / success_itr if success_itr != 0 else float(max_num_of_itr)
            )
            succ_prob = 100.0 * (success_itr / sim_itr)
            row = [n_err, avg_num_of_itr, succ_prob]
            writer.writerow(row)
            recorded_rows.append(row)

            if log_handle is not None:
                print(f"Num of error symbols: {n_err}", file=log_handle)
                print(f"Average Num of Iteration: {avg_num_of_itr}", file=log_handle)
                print(f"Probability of Success: {succ_prob}", file=log_handle)
                print("##########################", file=log_handle)

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "log_path": log_path,
        "rows": recorded_rows,
    }


def run_weight_distribution_analysis(
    *,
    q: int = 64,
    n: int = 64,
    k: int = 6,
    locality: Sequence[int] = (2, 5),
    local_minimum_distance: Sequence[int] = (3, 3),
    sub_group_type: Sequence[Literal["add", "mult"]] = ("add", "mult"),
    add_idx: int | None = None,
    output_dir: str | Path = "./results",
) -> dict[str, Any]:
    """Compute exact codeword weight distribution and minimum distance of a two-set code.

    Args:
        q: Finite field cardinality.
        n: Codeword length.
        k: Message dimension.
        locality: Two-element locality sequence ``[r_1, r_2]``.
        local_minimum_distance: Two-element local minimum distance sequence ``[d_1, d_2]``.
        sub_group_type: Two-element sequence of ``"add"`` or ``"mult"``.
        add_idx: Optional index into ``find_additive_subgroups`` overriding the
            default canonical additive subgroup.
        output_dir: Parent directory under which run artifacts are written.

    Returns:
        Dictionary with ``csv_path``, ``graph_path``, ``weight_distribution``, and
        ``code_distance``.
    """
    field = GF(int(q), repr="int")
    subgroup_override: list[Sequence[Any] | None] = [None, None]
    if add_idx is not None:
        if sub_group_type[0] == "add":
            sg_size = int(locality[0]) + int(local_minimum_distance[0]) - 1
            additive_subgroups = find_additive_subgroups(field, sg_size)
            subgroup_override = [additive_subgroups[int(add_idx)], None]
        elif sub_group_type[1] == "add":
            sg_size = int(locality[1]) + int(local_minimum_distance[1]) - 1
            additive_subgroups = find_additive_subgroups(field, sg_size)
            subgroup_override = [None, additive_subgroups[int(add_idx)]]
        else:
            raise ValueError("Cannot specify add_idx when neither subgroup_type is 'add'")

    _, code, _ = _prepare_code_and_codeword(
        q,
        n,
        k,
        locality,
        local_minimum_distance,
        sub_group_type,
        use_enc=False,
        subgroup=subgroup_override,
    )

    graph, left_nodes, right_nodes = code.bipartite_graph()
    graph_metrics = compute_expansion_metrics(graph, right_nodes)
    suffix = f"_idx_{add_idx}" if add_idx is not None else ""
    sim_name = _build_sim_name(code, suffix=suffix)

    res_dir = Path(output_dir) / f"{sim_name}_weight_dist"
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"{sim_name}_weight_dist.csv"
    graph_path = res_dir / f"{sim_name}_graph.png"

    save_bipartite_graph_plot(graph, left_nodes, right_nodes, graph_path)
    _print_code_summary(code, sim_name, graph_metrics)

    weight_dist = compute_code_weight_distribution(code)
    code_distance = next((i for i in range(1, len(weight_dist)) if weight_dist[i] != 0), None)

    print(weight_dist)
    if code_distance is not None:
        print(f"Code Distance: {code_distance}")

    with open(csv_path, "w", newline="") as res_file:
        writer = csv.writer(res_file)
        write_code_metadata_csv(writer, code, graph_metrics)
        writer.writerow(["Weight Distribution: ", weight_dist])
        if code_distance is not None:
            writer.writerow(["Code Distance: ", code_distance])

    print("## Simulation Done ##")
    return {
        "sim_name": sim_name,
        "csv_path": csv_path,
        "graph_path": graph_path,
        "weight_distribution": weight_dist,
        "code_distance": code_distance,
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
