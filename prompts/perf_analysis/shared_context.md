# Shared Context — Performance & SageMath-Alternatives Analysis Swarm

Every agent MUST read this file first.

## Mission
Analyse the algorithm and solution used in **each function** of `tamo-barg-lrc`, do a deep performance analysis, and evaluate alternatives to the SageMath dependency. **Analysis only: never edit anything under `src/`, `simulation/`, `tests/`, `pyproject.toml`.**

## Environment
- Repo: `/home/antigravity/amirmulla/optimal_lrc` (package `src/tamo_barg_lrc/`).
- Python: `/home/antigravity/.venv-lrc/bin/python` (3.11, modular `passagemath` 10.8.x). No root/sudo/apt.
- Prototype installs go ONLY in a separate venv: `uv venv /tmp/venv-<agent>` then `uv pip install --python /tmp/venv-<agent>/bin/python <pkg>`. Never alter `.venv-lrc`.
- Prototype/benchmark scripts live in `reports/perf_analysis/scratch/<agent>/`. Only reports + scratch are writable.
- Baseline tests: `/home/antigravity/.venv-lrc/bin/pytest -q` → 34 passed (~7 s). Must still pass at the end (nothing should have changed).

## Function inventory (each owned by exactly ONE analysis agent)
| Agent | File(s) | Functions |
|---|---|---|
| A1 | `subgroups.py` | `_validate_finite_field`, `field_element_to_int`, `sort_field_elements`, `is_closed_under_addition`, `find_subspaces`, `find_additive_subgroups`, `find_additive_subgroups_bruteforce`, `list_multiplicative_subgroups`, `list_canonical_additive_subgroups`, `shift_additive_subgroup`, `list_subgroup_cosets` |
| A1 | `polynomials.py` | `build_annihilator_polynomial`, `build_good_polynomial`, `build_algebra_basis`, `build_encoding_basis`, `intersect_encoding_bases`, `evaluate_message_polynomial`, `interpolate_erasures` |
| A2 | `expander.py` | `create_bipartite_coset_graph`, `_compute_k_edge_components`, `get_error_position_subgraph`, `_collapse_to_weighted_simple_graph`, `_compute_adjacency_eigenvalues`, `_compute_algebraic_connectivity`, `compute_expansion_metrics`, `save_bipartite_graph_plot` |
| A3 | `tamo_barg.py` | `TamoBargCode.*`, `TamoBargVectorEncoder.*`, `TamoBargErasureDecoder.decode_to_code`, `TamoBargErrorDecoder.*` |
| A4 | `tamo_barg_two_sets.py` | `_is_int_like`, `_int_log_base`, `_safe_parity_check_matrix`, `TamoBargCodeTwoSets.*` (`__init__`, `design_distance`, `minimum_distance_py`), `TamoBargTwoSetsVectorEncoder.*` |
| A5 | `tamo_barg_two_sets.py` | `TamoBargIterativeDecoder`, `TamoBargIterativeEEDecoder`, `TamoBargGlobalDecoder`, `TamoBargGlobalEEDecoder`, `TamoBargTwoStepDecoder`, `TamoBargTwoStepEEDecoder` (`_partition_is_syndrome_clean`, `decode_to_code` each) |
| A6 | `simulation/runner.py`, `simulation/postprocess.py`, `simulation/*.py` CLIs | `build_decoder`, `invoke_decoder`, `sample_fixed_weight_error`, `compute_code_weight_distribution`, 4× `run_*_simulation`, CSV writers, `process_error_distribution_csv` |

(Run `grep -n "^def \|^    def \|^class " <file>` to confirm; any function missing from the table → A-agent of that file owns it and notes it.)

## Sage dependency map (starting point; B2 verifies)
GF (`finite_field_constructor`) ×6; `VectorSpace` ×6; `vector` ×5; `matrix` ×3; `Decoder/DecodingError` ×3; `PolynomialRing` ×2; `cached_method` ×2; `AbstractLinearCode` ×2; `GeneralizedReedSolomonCode` + Gao / BerlekampWelch / ErrorErasure decoders ×2; `Encoder` ×2; `cartesian_product` ×2; `randstate`, `misc_c`, `channel` (`StaticErrorRateChannel`, `random_error_vector`), `arith.misc.divisors` ×1 each. Method usage: `right_kernel`, `echelon_form`, `row_space().intersection`, `VectorSpace.subspaces`, `lagrange_polynomial`, `multiplicative_generator`, `primitive_element`, `to_integer`.

## Benchmark baselines (earlier QA, single core)
GF(256) subgroup+cosets 0.40 ms · two-set init + `generator_matrix` GF(256) ≈ 348 ms · `get_error_position_subgraph` 0.034 ms · single-set error decoder [16,5,2] ≈ 1 ms/word · Iterative GF(256) 3 errors ≈ 7.9 ms/word · TwoStep GF(256) 3 errors ≈ 10.5 ms/word (dominated by global length-256 Gao) · `minimum_distance_py` GF(64) k=2 0.83 ms.

## Benchmark methodology (mandatory)
1. Fixed seeds (`random.seed`, `numpy.random.seed`, Sage `set_random_seed`).
2. Warm-up ≥ 3 calls; then ≥ 20 repeats (≥ 5 for >1 s calls); report **median, p10, p90**.
3. State parameters (q, n, k, r, partition sizes, error weight) and machine (`nproc`, `python -V`).
4. Use `time.perf_counter`, `cProfile`/`pstats`, `tracemalloc`; `py-spy` if installable.
5. Separate construction cost from per-call cost; report cached vs uncached.
6. Scaling sweeps: q ∈ {16, 64, 256, 1024, 4096} where tractable; fit empirical exponent (log-log slope) and compare with the theoretical complexity.

## Mandatory per-function row (all A-agents)
| Field | Content |
|---|---|
| Function | `module.Class.method` + signature + 1-line purpose |
| Algorithm | Exact algorithm/approach (e.g. "Zassenhaus via RREF of stacked basis") and why it fits |
| Complexity | Time and space in n, k, q, r, d_local, #partitions, with derivation |
| Sage primitives | Each Sage call + its underlying algorithm/backend (Linbox, FLINT, PARI, NTL, pure Python/Cython) and cost |
| Measured | median/p10/p90 and empirical scaling exponent, peak memory |
| Bottleneck | Dominant term / hotspot (from profile) |
| Correctness risks | Edge cases, numerical/field subtleties |
| Better algorithm | Concrete alternative with expected complexity gain |
| Sage-replacement difficulty | 1 (trivial) – 5 (very hard) + the replacement primitive needed |
| Priority | P0 / P1 / P2 by expected speed-up × usage frequency |

Deliver also: a short narrative (≤ 1 page) of the module's overall solution design and critique.

## Output locations (disjoint)
`reports/perf_analysis/<ID>_<name>.md` plus `reports/perf_analysis/scratch/<ID>/`. Final: `reports/perf_analysis/FINAL_REPORT.md`.

## Rules
- Evidence over opinion: every claim cites a measurement, a profile line, or a derivation.
- Do not guess Sage internals: read the source in the installed passagemath (`site-packages/sage/...`) or docs, and cite the path.
- If a candidate library cannot be installed (proxy/hosts), record the failure and evaluate on documentation + API only, flagged "unverified".
- Report a blocked item and stop; never bypass sandbox/proxy restrictions.
- Be concise: tables over prose.
