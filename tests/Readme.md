# `optimal_lrc` Test Suite

Deterministic, headless-safe `pytest` unit, property, and regression suite for `optimal_lrc` and its CLI simulation scripts.

## Running the Test Suite

Using a SageMath Python environment (`sage -python` or a virtual environment with `passagemath` installed):

```bash
sage -python -m pytest -v
```

To run a single test module:

```bash
sage -python -m pytest tests/test_tamo_barg_two_sets.py -v
```

To lint and check formatting on the test suite:

```bash
ruff check tests/
ruff format --check tests/
```

## Test Modules & Regression Coverage

| Test Module | Scope | Key Regressions & Optimizations Covered |
| :--- | :--- | :--- |
| `tests/conftest.py` | Autouse deterministic seeding (`random.seed(42)`, `set_random_seed(42)`) and headless `Agg` Matplotlib backend | `B18` (no interactive GUI plot window blocking) |
| `tests/test_subgroups.py` | Finite-field additive/multiplicative subgroups, Gaussian binomial counts, coset partitions, and univariate polynomial / Lagrange helpers | `B7` ($p \ge 11$ and $p > 36$ subspaces, non-prime-power validation), `B8` (`{0}` and full-field canonical subgroups, deterministic `{0}` coset), `B13` (exact multiplicative subgroup orders), `P3`, `P5` |
| `tests/test_expander.py` | Bipartite coset multigraphs (`["add", "add"]`, `["add", "mult"]`, `["mult", "mult"]`), component index extraction, spectral expansion metrics, and plot saving | `B12` (polynomial field representation without `repr='int'` and $n = 63$ `["mult", "mult"]` indices in `0 .. 62`), `B18` (figure cleanup), `P4` |
| `tests/test_tamo_barg.py` | Single-set `TamoBargCode` (`[12, 8, 3]` over $GF(13)$, `[16, 5, 2]` over $GF(16)$), `VectorEncoder`, `ErasureDecoder`, and `ErrorDecoder` | `B4` (immutable received vectors not mutated), `B5` (`ErrorDecoder` ambient `input_space` & `_repr_`), `B6` (`enc_basis` sliced to $k < k_{\max}$ and `enc_poly`), `B11` (upfront `ValueError` validation), `B17`, `P1` |
| `tests/test_tamo_barg_two_sets.py` | Two-set `TamoBargCodeTwoSets` across all 4 subgroup pairs, `VectorEncoder`, and all 6 iterative/global/two-step decoders | `B1` (Partition 1 uncorrectable error & multi-iteration recovery), `B2` ($d_{\text{local}} - 1$ DUE Lagrange recovery & erasure clearing), `B3` (`GlobalErasureErrorDecoder` `evaluation_points()` Lagrange branch), `B4` (immutability & canonical return shapes), `B5` (Cartesian `input_space`), `B6`, `B9` (`shift_add=True` disjoint subgroups), `B10` (`["mult", "add"]` & 2-set validation), `B11`, `B13`, `B17`, `P1`, `P2`, `P5`, `P6` (`minimum_distance_py`) |
| `tests/test_simulations.py` | `optimal_lrc.simulation` (`DecodeOutcome`, `build_decoder`, `invoke_decoder`, `process_error_distribution_csv`) and `main(argv)` smoke tests across all 7 `simulation/*.py` scripts | `B14` (`(design_distance - 1) // 2` precedence), `B15` (all 6 decoder constructor/return dispatch paths), `B16` (`sim_itr < print_freq_factor` without `ZeroDivisionError`), `B18` (closed CSV handles, no duplicate metadata rows, zero leaked Matplotlib figures) |