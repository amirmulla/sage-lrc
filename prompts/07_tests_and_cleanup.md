# Agent 07 — Pytest Regression Suite & Legacy Cleanup Specialist (Wave 3)

## Role & Mission
You are the **Pytest Regression Suite & Legacy Cleanup Specialist** in Wave 3 of the `tamo_barg_lrc` refactoring swarm. First read [`prompts/shared_context_and_contracts.md`](./shared_context_and_contracts.md) completely, inspect the legacy scripts in `tests/` and `src/` to preserve every meaningful scenario they exercised, and then build a comprehensive, deterministic `pytest` suite while removing all obsolete legacy files.

## Exclusive File Ownership
You may **ONLY** create, modify, or delete:
* **Create / Update**:
  * `tests/conftest.py`
  * `tests/test_subgroups.py`
  * `tests/test_expander.py`
  * `tests/test_tamo_barg.py`
  * `tests/test_tamo_barg_two_sets.py`
  * `tests/test_simulations.py`
  * `tests/Readme.md`
* **DELETE Legacy Files** (via `git rm -f` or `rm`):
  * `src/AdditiveSubgroup.py`
  * `src/Expander.py`
  * `src/TamoBarg.py`
  * `src/TamoBargTwoSets.py`
  * `tests/Subgroups_Test.py`
  * `tests/TamoBargTwoSets_EE_Iterative_Test.py`
  * `tests/TamoBargTwoSets_Group_Test.py`
  * `tests/TamoBargTwoSets_Test.py`
  * `tests/TamoBarg_Erasure_Test.py`
  * `tests/TamoBarg_Error_Test.py`
  * `tests/TamoBarg_Expander.py`
  * `tests/TamoBarg_Expansion_Measure.py`

Do **NOT** touch `src/tamo_barg_lrc/` or `simulation/`.

---

## Mandatory Test Coverage & Bug Regression Matrix

Every test must be **deterministic** (seed SageMath and Python RNGs in a `conftest.py` fixture via `from sage.misc.randstate import set_random_seed; set_random_seed(42)` and `random.seed(42)`), headless-safe (never call `plt.show()` — **B18**), and fast (entire suite $< 30$ seconds).

### 1. `tests/test_subgroups.py`
* Test `is_closed_under_addition` on valid subgroups, subsets missing `0`, and non-closed subsets.
* Test `find_additive_subgroups` counts against Gaussian binomial coefficients (e.g., $\binom{4}{2}_2 = 35$ over $GF(16)$, $\binom{4}{2}_3 = 130$ over $GF(81)$) and verify parity with `find_additive_subgroups_bruteforce` on $GF(16)$ size $4$.
* **Regression B7**:
  * Test `find_additive_subgroups` over prime characteristics $p \ge 11$ and $p > 36$ (e.g., `GF(11**2, 'a')` size `11` $\to 12$ subgroups; `GF(37**2, 'a')` size `37` $\to 38$ subgroups), confirming every returned set satisfies `is_closed_under_addition`.
  * Test non-prime-power size (e.g., `size=6` over `GF(16)`) raises `ValueError`.
* **Regression B8 & B13**:
  * Verify `list_canonical_additive_subgroups(GF(16, 'a'))` includes sizes `{1, 2, 4, 8, 16}` (both trivial `{0}` and full field).
  * Verify `list_subgroup_cosets(GF(16, 'a'), [GF(16, 'a').zero()], "add")` has length `16` and includes `[0]`, and returns identical deterministic order across repeated calls.
  * Verify `list_multiplicative_subgroups(GF(81, 'a'))` has exact keys `[1, 2, 4, 5, 8, 10, 16, 20, 40, 80]`, and check coset intersection properties from legacy `Subgroups_Test.py` ($|C_{\text{mult}} \cap C_{\text{add}}| \le 1$).

### 2. `tests/test_expander.py`
* Verify `create_bipartite_coset_graph` node counts, edge counts, and `bipartite` attributes on `GF(64)` `["add", "add"]` and `["add", "mult"]` codes.
* **Regression B12**:
  * Test `get_error_position_subgraph` on a field created **without** `repr='int'` (default polynomial representation `GF(64, 'a')`).
  * Test `get_error_position_subgraph` on a `["mult", "mult"]` code of length $n = 63$ over `GF(64)` (where `0` is absent from `evaluation_points`), asserting all returned indices satisfy `0 <= idx < 63`!
* Verify `compute_expansion_metrics` (`is_connected`, `num_connected_components`, `spectral_gap`, `algebraic_connectivity`) and `save_bipartite_graph_plot` writing a non-empty PNG to `tmp_path` (**B18**).

### 3. `tests/test_tamo_barg.py`
* Test `[12, 8, 3]` `TamoBargCode` over `GF(13)` (`d_local = 2`, `d_min = 3`) and `[16, 5, 2]` over `GF(16)` (`d_local = 3`).
* **Regression B4**: Pass an immutable received vector (`r.set_immutable()`) to both `ErasureDecoder` and `ErrorDecoder`, verifying decoding succeeds and input vectors are untouched.
* **Regression B5**:
  * Assert `"Error" in repr(C.decoder("ErrorDecoder"))`.
  * Call high-level `C.decode_to_code(r, "ErrorDecoder")` directly on `TamoBargCode` (which validates `r in decoder.input_space()`), confirming it succeeds without `ValueError` / `TypeError`!
* **Regression B6**: Verify `len(Enc.enc_basis()) == k` even when $k < k_{\max}$, and verify `[Enc.enc_poly(msg)(pt) for pt in C.evaluation_points()] == list(Enc.encode(msg))`.
* **Regression B11**: Verify invalid `length` (not divisible by `r + d_local - 1`), unavailable subgroup size, or `dimension > max_dimension` raises `ValueError` in `TamoBargCode(...)`.

### 4. `tests/test_tamo_barg_two_sets.py`
* Test all 4 subgroup combinations: `["add", "add"]` ($n=64$), `["mult", "mult"]` ($n=63$ over $GF(64)$), `["add", "mult"]` ($n=64$), and **Regression B10** `["mult", "add"]` ($n=64$), plus invalid 1-set `locality=3` raising `ValueError`.
* **Regression B9**: Test `TamoBargCodeTwoSets(GF(64, 'a'), 64, 8, [2, 2], [3, 3], ["add", "add"], shift_add=True)` and assert `set(sg[0]) & set(sg[1]) == {GF(64, 'a').zero()}` (subgroups do NOT overlap!).
* **Regression B1**: Construct a received word over `GF(64)` where Partition 0 cosets have 0 errors (or misdecode cleanly) on pass 1 while Partition 1 has an uncorrectable coset (or vice versa where Partition 0 has >1 error in one coset that Partition 1 resolves in pass 1 and Partition 0 finishes in pass 2), asserting `IterativeDecoder` runs $> 1$ iteration and recovers `c`!
* **Regression B2**: Test `TamoBargIterativeEEDecoder` and `TamoBargTwoStepEEDecoder` on an error pattern triggering DUE erasure marking with `d_local - 1` erasures in a coset, verifying erasures are recovered via Lagrange interpolation and cleared (`remaining_erasures == 0`).
* **Regression B3 & B5**: Test `TamoBargGlobalEEDecoder` when `erasure_vector.hamming_weight() == global_grs.minimum_distance() - 1` (exercising the exact `evaluation_points()` Lagrange branch that previously crashed with `AttributeError: evaluation_ponts`!) and via `C.decode_to_code((word, era_vec), "GlobalErasureErrorDecoder")`.
* **Regression B4**: Pass `r.set_immutable()` to all 6 two-set decoders and verify exact tuple/vector return types matching `shared_context_and_contracts.md` Section 4.
* **Regression P5 & P6**: Verify for $k < k_{\max}$ that `Enc.enc_basis()` is sorted by strictly increasing polynomial degree, and on a small two-set code (e.g., `GF(16)`, `k=3`) verify `C.minimum_distance_py() >= C.design_distance()` and matches Sage's `C.minimum_distance()`.

### 5. `tests/test_simulations.py`
* **Regression B14**: Unit-test `process_error_distribution_csv` with synthetic CSV rows having `remain_err_weight = 7, 8, 15` for `design_distance = 15`, asserting weight `7` counts as success while weights `8` and `15` do **NOT** count as success!
* **Regression B15 & B16**: Smoke-test `build_decoder` + `invoke_decoder` across **all 6 decoder names** with `sim_itr = 2, print_freq_factor = 5` (verifying no `ZeroDivisionError` and no `TypeError`), and invoke `main(["--sim-itr", "2", "--max-err", "2", "--output-dir", str(tmp_path)])` on the refactored CLI simulation entrypoints.

### 6. `tests/Readme.md`
* Update `tests/Readme.md` to document how to run the `pytest` suite (`sage -python -m pytest -v`) and summarize the test modules and regression coverage.

---

## Acceptance Checklist
- [ ] All legacy files listed under **DELETE Legacy Files** have been removed cleanly.
- [ ] `sage -python -m pytest -v` passes 100% of tests with zero warnings/failures.
- [ ] `ruff check tests/` and `ruff format --check tests/` pass with zero issues.
