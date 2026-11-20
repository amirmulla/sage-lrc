# B2 — SageMath Dependency Surface Audit

**Scope:** Exhaustive static (`ast` + regex) and runtime audit of every SageMath import, method, attribute, implicit operator, GRS decoder implementation, field-support requirement, and installed wheel footprint across `src/`, `simulation/`, and `tests/` of `tamo-barg-lrc` (`0.2.0`).  
**Environment:** Python `3.11.2` (`/home/antigravity/.venv-lrc/bin/python`), `passagemath-*` `10.8.14`, Linux (2 vCPU).  
**Scratch scripts & raw data:** [ast_sage_audit.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/ast_sage_audit.py), [complete_symbol_table.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/complete_symbol_table.py), [decoder_semantics_probe.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/decoder_semantics_probe.py), [footprint_audit.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/footprint_audit.py).

---

## Executive Summary

1. **Compact dependency surface:** Across all 22 Python files (`src/`, `simulation/`, `tests/`), the project imports exactly **21 unique `sage.*` symbols** (19 in `src/`, plus `ZZ` and `QQ` in `tests/` solely as negative-validation inputs). All 7 CLI scripts under `simulation/`, plus [expander.py](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py) and [postprocess.py](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/simulation/postprocess.py), have **zero direct Sage imports**.
2. **Corrections to initial dependency map:** Static AST verification disproves several items in the starting map: `StaticErrorRateChannel`, `right_kernel`, `row_space().intersection`, `.polynomial()`, and `.nth_root` are **not used anywhere** in the repo. Conversely, `zero_vector`, `sage.all` / `sage.all__sagemath_modules` (runtime bootstrap), `left_kernel_matrix`, and `ZZ`/`QQ` (tests) were missing from the initial map.
3. **Field requirement:** Operational simulations target $\mathbb{F}_{2^m}$ (`q = 64, 256`, scaling up to `4096+`), plus `GF(2)` for binary erasure masks. However, the test suite and library API explicitly require **general finite fields $\mathbb{F}_{p^m}$** including odd primes (`GF(13)`) and odd prime powers with large characteristic (`GF(9)`, `GF(81)`, `GF(121)`, `GF(1369)`). Arbitrary field embeddings are **not** needed—only the prime subfield $\mathbb{F}_p$, power-basis integer mapping, and a primitive element.
4. **GRS decoders are the hardest layer (L4), but only plain RS is needed:** Every `GeneralizedReedSolomonCode` instantiation in the repo uses trivial column multipliers `[1, ..., 1]` (plain Reed–Solomon on arbitrary evaluation points). Reading `sage/coding/grs_code.py` revealed two critical semantics:
   - Both `GRSGaoDecoder` and `GRSBerlekampWelchDecoder` are **100% behaviour-equivalent** bounded-distance decoders (200/200 exact agreement on random beyond-radius words), and **both leak `ValueError`** instead of `DecodingError` in ~2% of beyond-radius inputs when `deg(h) >= k` reaches `GRSEvaluationPolynomialEncoder.encode(h)`.
   - `GRSErrorErasureDecoder` constructs a brand-new `GeneralizedReedSolomonCode` on **every call** by puncturing erased positions and delegating to Gao.
5. **Severe SageMath footprint & licensing overhead:** The installed `passagemath` stack spans **17 `passagemath-*` wheels (1,688.1 MiB, 23,717 files)** + **9 Sage support wheels (21.2 MiB)** + transitive Jupyter/REPL deps (~50+ MiB), accounting for **>85% of the 2.1 GiB `.venv-lrc`**. All 17 `passagemath-*` wheels are **GPL-2.0-or-later** (conflicting with the project's **MIT** license), require Python `>=3.11,<3.15` (contradicting `pyproject.toml`'s `>=3.10`), and impose a **608.5 ms** base import penalty (**1,141.4 ms** to import `TamoBargCode` + `TamoBargCodeTwoSets`).

---

## 1. Complete Sage Symbol Table (`file:line`)

### 1.1 Explicit `sage.*` Imports (All 21 Unique Symbols)

| # | Imported Symbol | Layer | Replaceability | Import Sites (`file:line`) | Usage Count & Sites | Exercised By Tests |
|---|---|---|---|---|---|---|
| 1 | `sage.all` | L5 | Trivial (remove) | `src/tamo_barg_lrc/__init__.py:9` | Runtime bootstrap (`try` branch, unused in `passagemath`) | All 34 tests (via `tests/conftest.py:11`) |
| 2 | `sage.all__sagemath_modules` | L5 | Trivial (remove) | `src/tamo_barg_lrc/__init__.py:11` | Runtime bootstrap (`except` branch; avoids Sage circular `Category` import) | All 34 tests |
| 3 | `sage.rings.finite_rings.finite_field_constructor.GF` | L1 | Moderate | `src/tamo_barg_lrc/subgroups.py:9`, `tamo_barg.py:15`, `tamo_barg_two_sets.py:18`, `simulation/runner.py:18`; `tests/test_expander.py:8`, `test_simulations.py:10`, `test_subgroups.py:4`, `test_tamo_barg.py:7`, `test_tamo_barg_two_sets.py:9` | 58 calls: `GF(2)` for erasure vectors (`tamo_barg.py:349`, `tamo_barg_two_sets.py:723,766,813,845`, `runner.py:126`), `GF(p)` prime subfield (`subgroups.py:112`), `GF(q, repr="int")` (`runner.py:355,983`), plus 49 calls in `tests/` | All 34 tests |
| 4 | `sage.rings.integer_ring.ZZ` | L5 (test only) | Trivial | `tests/test_subgroups.py:5` | 1 use: `tests/test_subgroups.py:77` (non-finite-field negative test) | `test_find_subspaces_and_validation` |
| 5 | `sage.rings.rational_field.QQ` | L5 (test only) | Trivial | `tests/test_tamo_barg.py:8` | 1 use: `tests/test_tamo_barg.py:218` (non-finite-field negative test) | `test_regression_b11_upfront_parameter_validation` |
| 6 | `sage.modules.free_module.VectorSpace` | L2 | Trivial–Moderate | `src/tamo_barg_lrc/subgroups.py:8`, `tamo_barg.py:13`, `tamo_barg_two_sets.py:16`, `simulation/runner.py:16`; `tests/test_simulations.py:8`, `test_tamo_barg.py:5`, `test_tamo_barg_two_sets.py:7` | 22 calls: `subgroups.py:113` (`.subspaces`), `tamo_barg.py:349`, `tamo_barg_two_sets.py:406,813`, `runner.py:386,437` (iteration & `.random_element()`), plus 16 in `tests/` | All test files except `test_expander.py` |
| 7 | `sage.modules.free_module_element.vector` | L2 | Trivial | `src/tamo_barg_lrc/tamo_barg.py:14`, `tamo_barg_two_sets.py:17`; `tests/test_simulations.py:9`, `test_tamo_barg.py:6`, `test_tamo_barg_two_sets.py:8` | 44 calls: `tamo_barg.py:408,462,468,478`, `tamo_barg_two_sets.py:556,570,580,674,688,709,723,766,801,844,845,860`, plus 27 in `tests/` | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| 8 | `sage.modules.free_module_element.zero_vector` | L2 | Trivial | `src/tamo_barg_lrc/simulation/runner.py:17` | 2 calls: `runner.py:126` (`GF(2)`), `runner.py:390` (`field`) | `test_simulations.py` (all runner tests) |
| 9 | `sage.matrix.constructor.matrix` | L2 | Moderate | `src/tamo_barg_lrc/polynomials.py:6`, `tamo_barg.py:11`, `tamo_barg_two_sets.py:14` | 5 calls: `polynomials.py:196,202` (Zassenhaus matrices), `tamo_barg.py:330` (generator matrix), `tamo_barg_two_sets.py:72` (0-row PCM), `tamo_barg_two_sets.py:501` (generator matrix) | All 5 test files |
| 10 | `sage.rings.polynomial.polynomial_ring_constructor.PolynomialRing` | L3 | Moderate | `src/tamo_barg_lrc/tamo_barg.py:16`, `tamo_barg_two_sets.py:19`; `tests/test_subgroups.py:6` | 6 calls: `tamo_barg.py:256,351`, `tamo_barg_two_sets.py:432,644,817`, `tests/test_subgroups.py:193` | All 5 test files |
| 11 | `sage.coding.linear_code.AbstractLinearCode` | L4 | Moderate | `src/tamo_barg_lrc/tamo_barg.py:10`, `tamo_barg_two_sets.py:13` | 2 subclasses: `TamoBargCode` (`tamo_barg.py:54`), `TamoBargCodeTwoSets` (`tamo_barg_two_sets.py:76`) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_expander.py`, `test_simulations.py` |
| 12 | `sage.coding.encoder.Encoder` | L4 | Trivial | `src/tamo_barg_lrc/tamo_barg.py:8`, `tamo_barg_two_sets.py:11` | 2 subclasses + type hints: `TamoBargVectorEncoder` (`tamo_barg.py:62,239`), `TamoBargTwoSetsVectorEncoder` (`tamo_barg_two_sets.py:84,416`) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| 13 | `sage.coding.decoder.Decoder` | L4 | Trivial | `src/tamo_barg_lrc/tamo_barg.py:7`, `tamo_barg_two_sets.py:10` | 8 subclasses + type hints: `tamo_barg.py:63,340,411`; `tamo_barg_two_sets.py:85,511,611,774,807,864,899` | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| 14 | `sage.coding.decoder.DecodingError` | L4 | Trivial | `src/tamo_barg_lrc/tamo_barg.py:7`, `tamo_barg_two_sets.py:10`, `simulation/runner.py:14`; `tests/test_tamo_barg.py:4`, `test_tamo_barg_two_sets.py:6` | 11 uses (raise/catch/pytest.raises): `tamo_barg.py:401,475`, `tamo_barg_two_sets.py:585,725,852,889,930`, `runner.py:121,129`, `test_tamo_barg.py:82`, `test_tamo_barg_two_sets.py:237` | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| 15 | `sage.coding.grs_code.GeneralizedReedSolomonCode` | L4 | **Hard** | `src/tamo_barg_lrc/tamo_barg.py:9`, `tamo_barg_two_sets.py:12` | 5 instantiation sites + type annotations: `tamo_barg.py:427` (BW), `tamo_barg_two_sets.py:541` (local Gao), `659` (local EE), `781` (global Gao), `818` (global EE) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| 16 | `sage.coding.channel.random_error_vector` | L4 | Trivial | `src/tamo_barg_lrc/simulation/runner.py:13` | 1 call: `runner.py:252` (fills `weight` positions with random non-zero elements of `GF(q)`) | `test_simulations.py` |
| 17 | `sage.categories.cartesian_product.cartesian_product` | L5 | Trivial | `src/tamo_barg_lrc/tamo_barg.py:6`, `tamo_barg_two_sets.py:9` | 2 calls: `tamo_barg.py:349`, `tamo_barg_two_sets.py:812` (builds `input_space` for erasure decoders) | `test_tamo_barg.py:161`, `test_tamo_barg_two_sets.py:223` |
| 18 | `sage.misc.cachefunc.cached_method` | L5 | Trivial (`functools.lru_cache` / `cached_property`) | `src/tamo_barg_lrc/tamo_barg.py:12`, `tamo_barg_two_sets.py:15` | 2 decorators: `TamoBargVectorEncoder.generator_matrix` (`tamo_barg.py:325`), `TamoBargTwoSetsVectorEncoder.generator_matrix` (`tamo_barg_two_sets.py:496`) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py` |
| 19 | `sage.misc.misc_c.prod` | L5 / L3 | Trivial (balanced product tree) | `src/tamo_barg_lrc/polynomials.py:7` | 2 calls: `polynomials.py:34` (annihilator), `polynomials.py:69` (additive good polynomial) | `test_subgroups.py`, `test_tamo_barg.py`, `test_tamo_barg_two_sets.py` |
| 20 | `sage.arith.misc.divisors` | L5 | Trivial (~10 LOC) | `src/tamo_barg_lrc/subgroups.py:7` | 1 call: `subgroups.py:241` (`divisors(q - 1)`) | `test_subgroups.py`, `test_tamo_barg.py`, `test_tamo_barg_two_sets.py` |
| 21 | `sage.misc.randstate.set_random_seed` | L5 | Trivial (`numpy.random.Generator` / `random.Random`) | `src/tamo_barg_lrc/simulation/runner.py:15`; `tests/conftest.py:9` | 2 calls: `runner.py:166`, `tests/conftest.py:24` | All 34 tests (autouse fixture) |

### 1.2 Sage-Originated Methods & Attributes (Classified by Layer L1–L5)

#### L1 — Finite Field Arithmetic & Conversions

| Method / Operation | Sites (`file:line`) | Required Semantics | Replaceability | Exercised By Tests |
|---|---|---|---|---|
| `.is_finite()`, `.is_field()` | `subgroups.py:30`, `tamo_barg.py:89-90`, `tamo_barg_two_sets.py:119` | Distinguish finite fields from `ZZ`, `QQ`, or non-field objects | Trivial | `test_subgroups.py:77`, `test_tamo_barg.py:218` |
| `.cardinality()` | `subgroups.py:135,200,236`, `tamo_barg.py:114,201`, `tamo_barg_two_sets.py:238,331`, `runner.py:265,339,416` | Exact integer $q = p^m$ | Trivial | All 5 test files |
| `.characteristic()` | `subgroups.py:112,133,271`, `tamo_barg_two_sets.py:180` | Exact prime $p$ | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `.degree()` (on field) | `subgroups.py:134,272,302` | Extension degree $m$ ($q = p^m$; `1` for prime fields) | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `.gen()` (on field) | `subgroups.py:158,273`; `tests/test_subgroups.py:219`, `test_simulations.py:91`, `test_tamo_barg.py:104,132,156`, `test_tamo_barg_two_sets.py:55,75,94,132,147,185,231,249,334` | Field generator $\alpha$ whose powers $\{1, \alpha, \dots, \alpha^{m-1}\}$ form an $\mathbb{F}_p$-basis (for prime field, `1` in Sage) | Trivial | All test files |
| `.multiplicative_generator()` | `subgroups.py:237,303` | Generator $g$ of cyclic group $\mathbb{F}_q^\times$ (order $q - 1$) | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `.primitive_element()` | `subgroups.py:301` | Primitive element of extension field $\mathbb{F}_{p^m}$ ($m > 1$); identical semantics to `multiplicative_generator()` | Trivial | `test_subgroups.py:183,196`, `test_tamo_barg_two_sets.py:101,106` |
| `.zero()`, `.one()` (on field) | `subgroups.py:77,159,207,238,275`, `runner.py:212`, plus 34 sites in `tests/` | Additive & multiplicative identity elements | Trivial | All 5 test files |
| `.is_zero()` (on element) | `subgroups.py:43,208,335,340` | Boolean test `elem == 0` | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `.to_integer()` / `int(elem)` | `subgroups.py:45-47` (`field_element_to_int`) | Deterministic bijection $\mathbb{F}_{p^m} \to \{0,\dots,q-1\}$ via base-$p$ polynomial evaluation $\sum_{j=0}^{m-1} c_j p^j$ with $0 \mapsto 0$; prime fields fall back to `int(elem)` | Trivial | `test_subgroups.py:31-41`, `test_expander.py:62-95` |
| `.parent()` (on element) | `subgroups.py:77` | Retrieve owning field to get `.zero()` | Trivial | `test_subgroups.py:43-67` |
| Field coercion `field(x)` | `subgroups.py:166,306,334`, `polynomials.py:34,69,254`, `tamo_barg.py:397`, `tamo_barg_two_sets.py:193`; `tests/test_subgroups.py:66` | Convert `int` (mod $p$) or existing field element to element of `field` | Trivial | All 5 test files |
| Field iteration `for x in field` / `list(field)` | `subgroups.py:208,338`, `tests/test_expander.py:46`, `test_subgroups.py:35,38,55,149,174,198` | Enumerate all $q$ elements of $\mathbb{F}_q$ | Trivial | `test_subgroups.py`, `test_expander.py` |
| Field element ops (`+`, `-`, `*`, `**`, `==`, `!=`, `hash`) | `subgroups.py:86,160,166,243,247,278,279,305,306,348`, `tamo_barg_two_sets.py:194,272,280`, plus `tests/` | Exact finite-field arithmetic, equality, and hashability for `set`/`dict` membership | Moderate (speed critical) | All 5 test files |

#### L2 — Linear Algebra (`VectorSpace`, `vector`, `matrix`)

| Method / Operation | Sites (`file:line`) | Required Semantics | Replaceability | Exercised By Tests |
|---|---|---|---|---|
| `VectorSpace(GF(p), m).subspaces(dim)` | `subgroups.py:113-114`, iterated at `subgroups.py:163,169` | Enumerate all `dim`-dimensional subspaces of $\mathbb{F}_p^m$ (Gaussian binomial $\binom{m}{\text{dim}}_p$) and iterate vectors in each subspace | Trivial (RREF pivot enumeration over $\mathbb{F}_p$, ~25 LOC) | `test_subgroups.py:69-129` |
| `VectorSpace(field, d)` iteration | `tamo_barg_two_sets.py:406`, `runner.py:437`, `tests/test_tamo_barg.py:56`, `test_tamo_barg_two_sets.py:349` | Iterate all $q^d$ vectors in $\mathbb{F}_q^d$ | Trivial (`itertools.product` / integer counter) | `test_tamo_barg.py:56`, `test_tamo_barg_two_sets.py:340-352`, `test_simulations.py:236` |
| `VectorSpace(field, k).random_element()` | `runner.py:387`, plus 13 calls in `tests/` | Uniform random vector in $\mathbb{F}_q^k$ seeded by RNG | Trivial | `test_simulations.py`, `test_tamo_barg.py`, `test_tamo_barg_two_sets.py` |
| `vector(field, seq)` / `zero_vector(field, n)` | `tamo_barg.py:408,462,468,478`, `tamo_barg_two_sets.py:556,570,580,674,688,709,723,766,801,844,845,860`, `runner.py:126,390`, plus 27 in `tests/` | Construct 1D vector over `field` (including `GF(2)`); supports indexing `v[i]`, slicing `v[a:b]`, item assignment `v[i] = x`, `len(v)`, `list(v)`, `copy(v)`, `+`, `-`, `==`, `!=` | Trivial (1D NumPy/int array wrapper) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `v.hamming_weight()` | `tamo_barg_two_sets.py:399,408,846,927`, `runner.py:142,433,439,580,768`, plus 6 in `tests/` | Count of non-zero coordinates (`int(np.count_nonzero(v))`) | Trivial | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `v.is_zero()` (on vector) | `tamo_barg.py:469`, `tamo_barg_two_sets.py:557,581,675,709,802,849`, `runner.py:421`, `tests/test_tamo_barg.py:56`, `test_tamo_barg_two_sets.py:350` | All-zero check (syndrome zero check & non-zero row check) | Trivial (`not np.any(v)`) | All decoder & weight tests |
| `obj.set_immutable()` / `.is_immutable()` | `tamo_barg.py:336`, `tamo_barg_two_sets.py:507`, `tests/test_simulations.py:92`, `test_tamo_barg.py:122,123,127,135,139,176`, `test_tamo_barg_two_sets.py:251,254,299` | Freeze vector/matrix so mutation raises error; `copy(v)` yields mutable clone | Trivial (`arr.flags.writeable = False` + guard) | Regression B4 tests (`test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py`) |
| `v.parent()` (on vector) | `tests/test_tamo_barg_two_sets.py:259,264,279` | Verify decoded vector lies in `code.ambient_space()` and erasure vector lies in `VectorSpace(GF(2), n)` | Trivial | `test_regression_b4_and_canonical_return_contracts_all_six_decoders` |
| `matrix(F, nrows, ncols, rows)` + `.nrows()`, `.ncols()`, `.row(i)`, `.rows()`, iteration | `polynomials.py:196,202,210,213,219`, `tamo_barg.py:330`, `tamo_barg_two_sets.py:72,396,501`, `runner.py:421,430`, `tests/test_tamo_barg.py:174-175` | Dense 2D matrix over $\mathbb{F}_q$ (including 0-row matrix `0 x n` at `tamo_barg_two_sets.py:72`) | Moderate | All 5 test files |
| `mat.stack(-mat_b)` | `polynomials.py:209` | Unary negation & vertical block stacking over $\mathbb{F}_q$ | Trivial | `test_subgroups.py`, `test_tamo_barg_two_sets.py`, `test_expander.py`, `test_simulations.py` |
| `mat.left_kernel_matrix()` & `mat.left_kernel().basis()[0]` | `polynomials.py:209` (Zassenhaus intersection), `tests/test_tamo_barg_two_sets.py:33` (test helper) | Exact basis matrix of $\{x : x A = 0\}$ over $\mathbb{F}_q$ | Moderate (Gauss–Jordan RREF on $A^T$) | `test_subgroups.py`, `test_tamo_barg_two_sets.py`, `test_expander.py`, `test_simulations.py` |
| `mat.echelon_form()` | `polynomials.py:215`, `runner.py:420` | Exact Reduced Row Echelon Form (RREF) over $\mathbb{F}_q$ | Moderate (Gauss–Jordan elimination over $\mathbb{F}_q$) | `test_subgroups.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `mat.matrix_from_columns(cols)` / `matrix_from_rows(rows)` | `polynomials.py:213,215`, `tamo_barg_two_sets.py:405`, `runner.py:426,436`, `tests/test_tamo_barg_two_sets.py:33` | Column/row re-indexing & slicing (including reverse column order `num_cols-1..0`) | Trivial (`arr[:, cols]` / `arr[rows, :]`) | All 5 test files |
| `row.nonzero_positions()` | `polynomials.py:220` | Sorted indices of non-zero entries in a matrix row | Trivial (`np.flatnonzero(row)`) | `test_subgroups.py`, `test_tamo_barg_two_sets.py` |
| Matrix–matrix & matrix–vector products (`A * B`, `M * v`, `v * M`) | `polynomials.py:213`, `tamo_barg.py:469`, `tamo_barg_two_sets.py:407,557,581,675,709,802,849`, `runner.py:438`, `tests/test_tamo_barg.py:56`, `test_tamo_barg_two_sets.py:34,348` | Exact $\mathbb{F}_q$ GEMM/GEMV; hot path for local syndrome checks `H * v` and weight enumeration `m * G` | Moderate (performance-critical) | All decoder & weight tests |

#### L3 — Univariate Polynomials

| Method / Operation | Sites (`file:line`) | Required Semantics | Replaceability | Exercised By Tests |
|---|---|---|---|---|
| `PolynomialRing(F, "x")`, `.gen()`, `.zero()`, `.one()`, `.base_ring()`, `poly.parent()` | `tamo_barg.py:256,351`, `tamo_barg_two_sets.py:432,644,817`, `polynomials.py:32-34,64-65,69,98-99,189,252,255`, `tests/test_subgroups.py:193` | Univariate ring $\mathbb{F}_q[x]$; coefficient-list constructor `poly_ring(coeffs)` at `polynomials.py:228` | Moderate | All 5 test files |
| Polynomial arithmetic (`x - c`, `x ** d`, `p * q`, `p % ann`, `c * p`, `sum(..., poly_ring.zero())`) | `polynomials.py:34,67,69,72,101,105,107,144,254` | Univariate multiplication, scalar multiply, addition, remainder `%`, monomial power; plus `prod(...)` product tree | Moderate (also needed by Gao/RS decoders) | `test_subgroups.py`, `test_tamo_barg*.py` |
| `poly.degree()` | `polynomials.py:71,100,106,143,147,148,151,192,193,230`, `tamo_barg_two_sets.py:369`, `tests/test_subgroups.py:200,215`, `test_tamo_barg_two_sets.py:313` | Integer degree ($\deg(0) = -1$ in Sage; note `max(..., default=0)`) | Trivial | All 5 test files |
| `poly.shift(k)` | `polynomials.py:142` | Multiply by $x^k$ (`k > 0`, i.e., prepend `k` zeros to coefficient list) | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `poly.list()` & `poly.padded_list(n)` | `polynomials.py:152,200,206` | Ascending-degree coefficient list `[c_0, c_1, ..., c_d]`, zero-padded to length `n` | Trivial | `test_subgroups.py`, `test_tamo_barg*.py` |
| `poly(alpha)` (evaluation) | `polynomials.py:297`, `tamo_barg.py:334`, `tamo_barg_two_sets.py:505`, `tests/test_subgroups.py:201,205,222`, `test_tamo_barg.py:181,185`, `test_tamo_barg_two_sets.py:319` | Horner evaluation $p(\alpha) \in \mathbb{F}_q$ | Trivial (Horner's rule; vectorizable) | All 5 test files |
| `poly_ring.lagrange_polynomial(known_pts)` | `polynomials.py:295` | Unique interpolating polynomial of degree `< len(known_pts)` through `(x_i, y_i)` pairs; Sage uses $O(m^2)$ Newton divided differences (`polynomial_ring.py:2524-2538`) | Trivial–Moderate (~25 LOC Newton or barycentric Lagrange) | `test_subgroups.py:224-226`, `test_tamo_barg.py`, `test_tamo_barg_two_sets.py:190-225` |

#### L4 — Coding Theory Framework & GRS Decoders

| Method / Operation | Sites (`file:line`) | Required Semantics | Replaceability | Exercised By Tests |
|---|---|---|---|---|
| `AbstractLinearCode` inheritance & `super().__init__(F, n, default_enc, default_dec)` | `tamo_barg.py:54,169`, `tamo_barg_two_sets.py:76,291` | Provides `.base_field()`, `.length()`, `.dimension()`, `.ambient_space()`, `.encoder()`, `.decoder()`, `.generator_matrix()`, `.encode()`, `.unencode()`, `.decode_to_code()`, `_registered_encoders`/`_registered_decoders` dispatch, `@cached_method` on `encoder`/`decoder`, and SageObject `_repr_` | Moderate (~120 LOC clean pure-Python base class) | All test files except `test_subgroups.py` |
| `Encoder` base class (`encode`, `unencode`, `unencode_nocheck`, `message_space`) | `tamo_barg.py:239,251`, `tamo_barg_two_sets.py:416,428`; called via `code.encode(...)`, `code.unencode(...)`, `enc.encode(...)` (`runner.py:388`, `tamo_barg_two_sets.py:608,771,896,937`, plus 13 in `tests/`) | `encode(m) = m * G`; `unencode(c)` checks membership (`c in code`, raising error if not) then multiplies info-set slice by $G_I^{-1}$ | Trivial–Moderate (info-set RREF + inverse, ~40 LOC) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `Decoder` base class (`code()`, `input_space()`, `decode_to_code`, `decode_to_message`) | `tamo_barg.py:340,411`, `tamo_barg_two_sets.py:511,611,774,807,864,899` | Stores `_code`, `_input_space`, `_connected_encoder_name`; default `decode_to_message` calls `code.unencode(decode_to_code(r))` | Trivial (~25 LOC) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `DecodingError` | `tamo_barg.py:401,475`, `tamo_barg_two_sets.py:585,725,852,889,930`, `runner.py:121,129`, `tests/` | Exception subclass `class DecodingError(Exception): pass` | Trivial (1 LOC) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `GeneralizedReedSolomonCode(eval_pts, k)` + `.parity_check_matrix()` + `.minimum_distance()` + `.generator_matrix()` | `tamo_barg.py:427,430`, `tamo_barg_two_sets.py:71-73,541,659,781,818,847`, `tests/test_tamo_barg_two_sets.py:31` | Plain Reed–Solomon code (all column multipliers $= 1$) on arbitrary distinct `eval_pts`; $d = n - k + 1$; Vandermonde $G_{i,j} = \alpha_j^i$; dual Vandermonde $H_{i,j} = \eta_j \alpha_j^i$ with $\eta_j = \prod_{h \ne j}(\alpha_j - \alpha_h)^{-1}$ | Moderate (~60 LOC for RS code + PCM) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `grs.decoder("BerlekampWelch").decode_to_code(v)` | `tamo_barg.py:429,473` | Bounded-distance RS decoder ($t = \lfloor(n-k)/2\rfloor$) on arbitrary eval points (see Section 3) | **Hard** (must be fast & exact over general $\mathbb{F}_q$; can unify with Gao/Sugiyama) | `test_tamo_barg.py:105-107,137-138,159` |
| `grs.decoder("Gao").decode_to_code(v)` | `tamo_barg_two_sets.py:543,584,785,804` | Bounded-distance RS decoder via Lagrange interpolation + partial XGCD (see Section 3) | **Hard** (performance-critical hot loop for local & length-$n$ global decoding) | `test_tamo_barg_two_sets.py`, `test_simulations.py` |
| `grs.decoder("ErrorErasure").decode_to_code((v, era))` | `tamo_barg_two_sets.py:661,722,822,861` | Joint RS error + erasure decoder ($2\tau + e \le d - 1$) on arbitrary eval points (see Section 3) | **Hard** (puncture+Gao or Forney syndrome-based EE decoder) | `test_tamo_barg_two_sets.py:190-238,262-296`, `test_simulations.py` |
| `code.weight_distribution()` & `code.minimum_distance()` (inherited GAP methods) | `runner.py:410` (with fallback at `414-442`), `tests/test_tamo_barg_two_sets.py:355` (wrapped in `try/except ImportError`) | Exact weight spectrum `[A_0, ..., A_n]` and exact $d_{\min}$. **Already has pure-Python projective fallbacks** in `runner.py:414-442` and `minimum_distance_py()` (`tamo_barg_two_sets.py:380-413`)! | Trivial (fallback already implemented) | `test_simulations.py:236`, `test_tamo_barg_two_sets.py:340-360` |
| `random_error_vector(n, F, positions)` | `runner.py:252` | Vector of length `n` with uniform random non-zero elements of `F` at `positions` and `0` elsewhere (`channel.py:64-97`) | Trivial (~6 LOC) | `test_simulations.py:129` + all simulation runs |

#### L5 — Misc & Runtime Glue

| Method / Operation | Sites (`file:line`) | Required Semantics | Replaceability | Exercised By Tests |
|---|---|---|---|---|
| `sage.all` / `sage.all__sagemath_modules` | `src/tamo_barg_lrc/__init__.py:9,11` | Initialize Sage Cython coercion/category framework | Trivial (eliminated completely) | All 34 tests |
| `cartesian_product([V_q, V_2])` + `input_space()((w, e))` | `tamo_barg.py:349`, `tamo_barg_two_sets.py:812`; `tests/test_tamo_barg.py:161`, `test_tamo_barg_two_sets.py:223` | Thin product-space descriptor accepting `(word, erasure_vec)` tuple validation | Trivial (~15 LOC) | `test_tamo_barg.py:161`, `test_tamo_barg_two_sets.py:223` |
| `@cached_method` | `tamo_barg.py:325`, `tamo_barg_two_sets.py:496` | Per-instance method memoization (zero-arg `generator_matrix`) | Trivial (`functools.lru_cache` / instance cache) | `test_tamo_barg.py`, `test_tamo_barg_two_sets.py` |
| `divisors(q - 1)` | `subgroups.py:241` | Sorted list of positive divisors of integer $q - 1$ | Trivial (~8 LOC trial division up to $\sqrt{q-1}$) | `test_subgroups.py`, `test_tamo_barg*.py` |
| `prod(factors, one)` | `polynomials.py:34,69` | Product of linear factors $(x - \alpha_i)$; balanced binary product tree gives $O(M(n)\log n)$ | Trivial (~10 LOC divide-and-conquer product tree) | `test_subgroups.py`, `test_tamo_barg*.py` |
| `set_random_seed(seed)` | `runner.py:166`, `tests/conftest.py:24` | Deterministic seeding for random field elements/vectors | Trivial (`random.seed` + `np.random.default_rng`) | All 34 tests |
| Sage `Integer` duck-typing | `tamo_barg_two_sets.py:54` (`numbers.Integral`), `int(...)` coercions throughout | Accept both Python `int` and backend integer scalars | Trivial (already Python-native) | `test_tamo_barg_two_sets.py` |

### 1.3 Sage Features Explicitly **NOT** Used

Verified zero occurrences across `src/`, `simulation/`, and `tests/`:
- No multivariate polynomial rings, polynomial factorization, polynomial `gcd` (outside `GRSGaoDecoder._partial_xgcd`), or `.polynomial()` / `.nth_root()` calls.
- No finite-field homomorphisms, embeddings $\mathbb{F}_{p^a} \hookrightarrow \mathbb{F}_{p^{ab}}$, or relative tower extensions.
- No `right_kernel()`, `row_space()`, `row_space().intersection()`, or `StaticErrorRateChannel`.
- No non-trivial GRS `column_multipliers` (every GRS code uses `[1, ..., 1]`).
- No symbolic math, calculus, plotting via Sage, or GAP beyond the optional `weight_distribution()`/`minimum_distance()` try/except hooks.

---

## 2. Field-Support Requirement Statement

Exhaustive scan of all 58 `GF(...)` call sites in `src/` and `tests/`, all 7 CLI defaults in `simulation/`, and `README.md`:

| Field Family | Concrete Instances Observed in Repo | Where Used | Mandatory? |
|---|---|---|---|
| **Binary field $\mathbb{F}_2$** | `GF(2)` | `src/tamo_barg_lrc/tamo_barg.py:349`, `tamo_barg_two_sets.py:723,766,813,845`, `runner.py:126` (erasure masks); `subgroups.py:112` (prime subfield of $\mathbb{F}_{2^m}$) | **Mandatory** (erasure masks can be plain `uint8`/`bool` vectors once decoupled from Sage's `GRSErrorErasureDecoder`, while $\mathbb{F}_2$ is still needed as prime subfield) |
| **Binary extension fields $\mathbb{F}_{2^m}$** | `GF(4)`, `GF(16)`, `GF(64)`, `GF(256)` (defaults across all 7 CLIs); scaling sweeps `GF(1024)`, `GF(4096)` (and potentially `GF(65536)`) | Primary workload: 100% of `simulation/*.py` defaults (`q=64`, `q=256`), `test_expander.py`, `test_simulations.py`, `test_tamo_barg_two_sets.py`, `README.md` | **Mandatory — Primary performance target** |
| **Odd prime fields $\mathbb{F}_p$** | `GF(13)` (and `GF(p)` via `GF(base_field.characteristic())` for $p \in \{3, 11, 37, \dots\}$) | `tests/test_tamo_barg.py:20,63,167,191` (4 of 6 single-set tests: `[12, 8, 3]` multiplicative cosets over `GF(13)`), `tests/test_subgroups.py:33`, `subgroups.py:112` | **Mandatory** |
| **Odd-characteristic extension fields $\mathbb{F}_{p^m}$ ($m \ge 2$)** | `GF(9) = GF(3^2)`, `GF(81) = GF(3^4)`, `GF(121) = GF(11^2)`, `GF(1369) = GF(37^2)` | `tests/test_subgroups.py:64,100,108,114,161` — specifically regression **B7** ($p \ge 11$ and $p > 36$ without base-$p$ digit bugs) and regression **B13** (`GF(81)` additive + multiplicative cosets) | **Mandatory** (for full API & test parity) |
| **Subfield / tower embeddings** | None | Not used anywhere. `find_subspaces` only needs $\mathbb{F}_p^m$ and the power basis $\{1, \alpha, \dots, \alpha^{m-1}\}$; `list_canonical_additive_subgroups` builds $\mathrm{span}_{\mathbb{F}_p}(1, \dots, \alpha^{i-1})$; `shift_additive_subgroup` multiplies by $g^s$ where $g$ is primitive. | **NOT required** |
| **`repr="int"` formatting** | `GF(int(q), repr="int")` | `runner.py:355,983` (changes `str(elem)` to base-$p$ integer string in CSV/logs; tests also assert default polynomial repr works without breaking `field_element_to_int`, e.g., regression **B12**) | **Minor** (just `__str__` formatting) |

### Concrete Sage Backends Currently Selected by Field Size (Verified Empirically)

From [decoder_semantics_probe.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/decoder_semantics_probe.py):

| Field | Field Backend | Polynomial Backend | Matrix Backend | `.to_integer()` |
|---|---|---|---|---|
| `GF(2)` | `FiniteField_prime_modn` | `Polynomial_GF2X` (NTL) | `Matrix_mod2_dense` (M4RI) | `False` (`int()`) |
| `GF(13)` | `FiniteField_prime_modn` | `Polynomial_zmod_flint` (FLINT) | `Matrix_modn_dense_float` (LinBox) | `False` (`int()`) |
| `GF(2^m)`, $2 \le m \le 15$ (`16..4096`) | `FiniteField_givaro` (Givaro Zech LUT) | `Polynomial_ZZ_pEX` (NTL) | `Matrix_gf2e_dense` (M4RIE) | `True` |
| `GF(65536)` ($2^{16}$) | `FiniteField_ntl_gf2e` (NTL) | `Polynomial_ZZ_pEX` (NTL) | `Matrix_gf2e_dense` (M4RIE) | `True` |
| Odd `GF(p^m)` (`9, 81, 121, 1369`) | `FiniteField_givaro` (Givaro LUT) | `Polynomial_ZZ_pEX` (NTL) | `Matrix_generic_dense` (generic Cython!) | `True` |

**Strategic conclusion for backend selection:**
- A **binary-only** ($\mathbb{F}_{2^m}$) library (e.g., `zfec`, `reedsolo`, `gf256`, `pyfinite`) **cannot** replace Sage on its own without dropping `GF(13)`, `GF(9)`, `GF(81)`, `GF(121)`, `GF(1369)` support and failing `test_subgroups.py` + `test_tamo_barg.py`.
- A **general $\mathbb{F}_{p^m}$** backend (e.g., `galois`, `python-flint`, or a hybrid engine with a fast $\mathbb{F}_{2^m}$ LUT/SIMD fast path + general $\mathbb{F}_{p^m}$ fallback) satisfies **100%** of field requirements.
- Determinism requirement: Conway polynomials (or a fixed canonical irreducible polynomial table) are needed if exact element-by-element equality of cosets/polynomials with Sage is desired, though all mathematical invariants (subgroup counts, coset partition sizes, code dimensions, design distances, decoding correctness) hold for *any* primitive irreducible polynomial of degree $m$ over $\mathbb{F}_p$.

---

## 3. Sage GRS Decoder Behaviour Specification

All line numbers refer to `/home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/` (`passagemath-modules 10.8.14`).

### 3.1 `GeneralizedReedSolomonCode` (`grs_code.py:74–480`)

- **Construction (`grs_code.py:145–247`):** Accepts `(evaluation_points, dimension, column_multipliers=None)`. In `tamo-barg-lrc`, `column_multipliers` is **always `None`** (defaults to `[1, ..., 1]`). Validates: finite field, `1 <= dimension <= length`, all evaluation points distinct. Default encoder: `"EvaluationVector"`; default decoder: `"Gao"`.
- **Minimum distance (`grs_code.py:327–342`):** MDS formula $d = n - k + 1$.
- **Generator matrix (`GRSEvaluationVectorEncoder.generator_matrix`, `grs_code.py:760–794`):** Cached immutable Vandermonde matrix $G_{i,j} = \alpha_j^i$ ($0 \le i < k, 0 \le j < n$).
- **Parity-check matrix (`grs_code.py:396–479`):** Computes `multipliers_product` $\eta_j = \prod_{h \ne j} (\alpha_j - \alpha_h)^{-1}$ in $O(n^2)$, constructs `dual_code = GeneralizedReedSolomonCode(alphas, n - k, etas)`, and returns its generator matrix $H_{i,j} = \eta_j \alpha_j^i$ ($0 \le i < n - k$).
  - **Edge-case crash:** When $k = n$, `dual_code()` passes `dimension = 0`, raising `ValueError: The dimension must be a positive integer at most the length of the code`. Our `_safe_parity_check_matrix` (`tamo_barg_two_sets.py:69–73`) guards this and returns a `0 x n` matrix.

### 3.2 `GRSBerlekampWelchDecoder` (`grs_code.py:1089–1391`)

**Used in:** `TamoBargErrorDecoder` (`src/tamo_barg_lrc/tamo_barg.py:429,473`).

- **Algorithm (`_decode_to_code_and_message`, `grs_code.py:1186–1253`):**
  1. Validates `r in C.ambient_space()` $\to$ else raises `ValueError`.
  2. Fast-path: if `n == k` or `r in C` (checked via `syndrome(r) == 0`), returns `(r, unencode_nocheck(r))`.
  3. Sets $t = \lfloor(d-1)/2\rfloor$, $\ell_0 = n - 1 - t$, $\ell_1 = n - t - k$.
  4. Builds the $n \times (\ell_0 + \ell_1 + 2)$ homogeneous linear system $S$ (`grs_code.py:1235–1237`):
     $$S_{i,j} = \begin{cases} \alpha_i^j & 0 \le j \le \ell_0 \\ r_i \alpha_i^{j - (\ell_0 + 1)} & \ell_0 + 1 \le j \le \ell_0 + \ell_1 + 1 \end{cases}$$
  5. Takes `S.right_kernel().basis_matrix().row(0)` ($O(n^3)$!) to split into polynomials $Q_0(x)$ ($\deg \le \ell_0$) and $Q_1(x)$ ($\deg \le \ell_1$).
  6. Exact division `f, rem = (-Q0).quo_rem(Q1)`.
     - If `not rem.is_zero()` $\to$ raises `DecodingError`.
     - If `f not in R` $\to$ raises `DecodingError`.
  7. Calls `c = self.connected_encoder().encode(f)` (`GRSEvaluationPolynomialEncoder.encode`, `grs_code.py:945–1015`):
     - **Upstream quirk (`grs_code.py:1010–1011`):** If `f.degree() >= k`, raises **`ValueError("The polynomial to encode must have degree at most ...")`** instead of `DecodingError`!
  8. Distance verification: if `(c - r).hamming_weight() > t` $\to$ raises `DecodingError`.
  9. Returns `c` (from `decode_to_code(r)`).

### 3.3 `GRSGaoDecoder` (`grs_code.py:1393–1769`)

**Used in:** `TamoBargIterativeDecoder` (`tamo_barg_two_sets.py:543,584`), `TamoBargGlobalDecoder` (`tamo_barg_two_sets.py:785,804`), and internally inside `GRSErrorErasureDecoder`.

- **Algorithm (`_decode_to_code_and_message`, `grs_code.py:1574–1629`):**
  1. Validates `r in C.ambient_space()` $\to$ else raises `ValueError`.
  2. Computes/caches $G(x) = \prod_{i=0}^{n-1}(x - \alpha_i)$ (`@cached_method`, `grs_code.py:1506–1531`, using naive sequential multiplication $O(n^2)$!).
  3. Fast-path: if `n == k` or `r in C`, returns `(r, unencode_nocheck(r))`.
  4. Interpolates $R(x)$ through $\{(\alpha_i, r_i)\}_{i=0}^{n-1}$ via `PolRing.lagrange_polynomial(points)` (Newton divided differences, $O(n^2)$).
  5. Runs `_partial_xgcd(G, R, PolRing)` (`grs_code.py:1533–1572`): Euclidean loop maintaining `(prev_r, r)` and `(prev_s, s)` until `r.degree() < (k + n) // 2`, returning `(Q1, Q0) = (r, s)` with $Q_1 \equiv Q_0 R \pmod G$.
  6. Exact division `h, rem = Q1.quo_rem(Q0)`.
     - If `not rem.is_zero()` $\to$ raises `DecodingError`.
     - If `h not in PolRing` $\to$ raises `DecodingError`.
  7. Calls `c = self.connected_encoder().encode(h)`:
     - **Same upstream quirk (`grs_code.py:1010–1011`):** If `h.degree() >= k`, raises **`ValueError`** instead of `DecodingError`!
  8. Distance verification: if `(c - r).hamming_weight() > t` $\to$ raises `DecodingError`.
  9. Returns `c` (from `decode_to_code(r)`).

### 3.4 `GRSErrorErasureDecoder` (`grs_code.py:1771–2010` + `decoder.py:225–258`)

**Used in:** `TamoBargIterativeEEDecoder` (`tamo_barg_two_sets.py:661,722`), `TamoBargGlobalEEDecoder` (`tamo_barg_two_sets.py:822,861`).

- **Call chain:** `GRSErrorErasureDecoder` does **not** override `decode_to_code`; it inherits `Decoder.decode_to_code` (`decoder.py:256–257`), which executes:
  ```python
  word = self.decode_to_message(r)
  return self.connected_encoder().encode(word)  # GRSEvaluationVectorEncoder.encode -> word * G
  ```
- **Algorithm (`decode_to_message`, `grs_code.py:1880–1975`):**
  1. Validates `word in C.ambient_space()` $\to$ else raises `ValueError`.
  2. Validates `erasure_vector in VectorSpace(GF(2), n)` $\to$ else raises `ValueError`.
  3. Checks `erasure_vector.hamming_weight() >= d` ($= n - k + 1$) $\to$ raises `DecodingError("Too many erasures in the received word")`.
  4. **Punctures erased coordinates on every call:** filters `word`, `evaluation_points`, `column_multipliers` to indices with `erasure_vector[i] == 0`, instantiating a **brand-new `GeneralizedReedSolomonCode` `C1`** of length $n' = n - e$ and dimension $k$ (even when $e = 0$!).
  5. If $n' == k$ (i.e., $e = d - 1$): calls `C1.unencode(punctured_word, nocheck=True)` (unique degree-$<k$ interpolation via Vandermonde matrix inverse; **cannot detect errors**, silently returns a codeword!).
  6. Else ($n' > k$): calls `C1.decode_to_message(punctured_word)` $\to$ `C1.unencode(C1.decoder("Gao").decode_to_code(punctured_word))`. Raises `DecodingError` (or `ValueError` if `deg(h) >= k`!) if residual errors exceed $\lfloor(n' - k)/2\rfloor = \lfloor(d - 1 - e)/2\rfloor$.
  7. `Decoder.decode_to_code` re-encodes the $k$-vector message over the full length-$n$ code `C` via `word * G`.

### 3.5 Exact Failure-Semantics & Compatibility Contract

Empirically verified via [decoder_semantics_probe.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/decoder_semantics_probe.py) on `[8, 4, 5]` over `GF(16)`, `[4, 3, 2]` (`d = 2`), `[4, 4, 1]` (`n = k`), and `[8, 4, 5]` over `GF(13)`:

| Scenario | Sage `GRSBerlekampWelch` | Sage `GRSGao` | Sage `GRSErrorErasure` | Required Replacement Behaviour |
|---|---|---|---|---|
| Clean codeword ($w = 0, e = 0$) | Returns `r` | Returns `r` | Reconstructs `C1`, returns `r` | Return copy of `r` (syndrome fast-path) |
| Within radius ($2w + e \le d - 1$, $e < d - 1$) | Returns unique `c` | Returns unique `c` | Returns unique `c` | **Must** return unique `c` |
| Exact $e = d - 1$, $w = 0$ | N/A | N/A | Returns unique `c` via `unencode(nocheck=True)` | **Must** return unique `c` (note: our EE wrappers already intercept `e == d - 1` with `interpolate_erasures`) |
| Exact $e = d - 1$, $w \ge 1$ | N/A | N/A | **Silent misdecode** (interpolates corrupted points) | Same (mathematically unavoidable: $n - e = k$ points uniquely determine a degree-$<k$ polynomial) |
| Too many erasures ($e \ge d$) | N/A | N/A | Raises `DecodingError` | **Must** raise `DecodingError` |
| Beyond radius ($2w + e > d - 1$) & no codeword within radius $t$ of punctured word | Raises `DecodingError` (180/184) or leaks `ValueError` (4/184) | Raises `DecodingError` (180/184) or leaks `ValueError` (4/184) | Raises `DecodingError` or leaks `ValueError` | Raise `DecodingError` (always check `deg(h) < k` before re-encoding; cleaner than Sage and 100% compatible with our `except (DecodingError, ValueError):` guards!) |
| Beyond radius ($w > t$) but within distance $\le t$ of *another* codeword $c' \ne c$ | **Misdecodes to $c'$** (16/200) | **Misdecodes to $c'$** (16/200, identical to BW!) | **Misdecodes to $c'$** | **Must** return that same unique $c'$ within radius $t$ (standard bounded-distance RS decoding; relied upon by B1 regression test `test_tamo_barg_two_sets.py:137-166`!) |
| Degenerate $d = 2$ ($t = 0$) with 1 error | Raises `DecodingError` | Raises `DecodingError` | Raises `DecodingError` | **Must** raise `DecodingError` |
| Full-rate $n = k$ ($d = 1, t = 0$) | Returns `r` | Returns `r` | Returns `r` (when $e = 0$) | Return copy of `r` |
| Wrong word length or invalid erasure mask | Raises `ValueError` | Raises `ValueError` | Raises `ValueError` | **Must** raise `ValueError` |
| Immutable input vectors (`r.set_immutable()`) | Does not mutate `r` | Does not mutate `r` | Does not mutate `word` or `erasure_vector` | **Must never mutate inputs** (Regression B4) |

**Key design takeaway:** Because Gao and Berlekamp–Welch agreed on **200/200** random beyond-radius trials (`disagree = 0`), and all our codes are plain RS codes (column multipliers $= 1$), a single fast **plain-RS bounded-distance error-and-erasure decoder** (e.g., syndrome-based Berlekamp–Massey/Sugiyama + Forney, or fast Gao on non-erased coordinates with a $e = 0$ cached-annihilator fast path) replaces **all three** Sage GRS decoders with 100% behavioural compatibility and eliminates per-call `GeneralizedReedSolomonCode` instantiation in `GRSErrorErasureDecoder`!

---

## 4. Installed `passagemath` Footprint Audit

Measured directly on `/home/antigravity/.venv-lrc` via [footprint_audit.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B2/footprint_audit.py).

### 4.1 Disk Size Summary (`du -sh`)

| Directory | Disk Size | Share of `.venv-lrc` |
|---|---:|---:|
| `/home/antigravity/.venv-lrc` (total venv) | **2.1 GiB** (`2,084.3 MB`) | 100.0% |
| `site-packages/` | **2.05 GiB** (`2,054.2 MB`) | 98.6% |
| `site-packages/sage/` | **900 MiB** (`933.6 MB`) | 44.8% |
| All 12 `passagemath_*.libs/` shared-library bundles | **461 MiB** (`483.7 MB`) | 23.2% |
| Non-Sage app deps (`numpy`, `matplotlib`, `networkx`, `pillow`, `fonttools`, etc., 11 wheels) | **134.7 MiB** | 6.5% |
| Dev + Jupyter/REPL transitive deps (34 wheels, incl. `ruff`, `pytest`, `ipython`, `ipykernel`, `jedi`, `debugpy`, `widgetsnbextension`) | **102.0 MiB** | 4.9% |
| `tamo-barg-lrc` itself | **0.01 MiB** | <0.001% |

### 4.2 Per-Wheel Breakdown: `passagemath-*` (17 Wheels) & Sage Support Deps (9 Wheels)

| Wheel | Version | Installed Size | Files | `Requires-Python` | License | Why Installed / Used? |
|---|---|---:|---:|---|---|---|
| `passagemath-modules` | `10.8.14` | 300.46 MiB | 802 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Core vectors/matrices/coding (`sage.coding.*`, `VectorSpace`) |
| `passagemath-gap` | `10.8.14` | 299.01 MiB | 14,391 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Pulled by `passagemath-groups`; only optional `weight_distribution`/`minimum_distance` |
| `passagemath-pari` | `10.8.14` | 268.03 MiB | 336 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Finite field / number theory C libs (bundles `cypari2`) |
| `passagemath-categories` | `10.8.14` | 175.00 MiB | 742 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Category & coercion framework, `cartesian_product` |
| `passagemath-flint` | `10.8.14` | 172.69 MiB | 379 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Prime-field polynomials (`Polynomial_zmod_flint`), `ZZ`/`QQ` |
| `passagemath-linbox` | `10.8.14` | 127.57 MiB | 56 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Givaro (`GF(p^m)` LUTs) + FFLAS-FFPACK (`GF(p)` matrices) |
| `passagemath-combinat` | `10.8.14` | 73.58 MiB | 404 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Combinatorics (loaded eagerly on bootstrap; not directly imported) |
| `passagemath-objects` | `10.8.14` | 64.61 MiB | 294 | `>=3.11, <3.15` | **GPL-2.0-or-later** | `SageObject`, `Parent`, `Element`, `cached_method` |
| `passagemath-iml` | `10.8.14` | 63.15 MiB | 21 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Pulled transitively by `passagemath-linbox` |
| `passagemath-ntl` | `10.8.14` | 60.17 MiB | 165 | `>=3.11, <3.15` | **GPL-2.0-or-later** | `Polynomial_ZZ_pEX`, `Polynomial_GF2X`, `GF(2^16+)` |
| `passagemath-groups` | `10.8.14` | 22.76 MiB | 42 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Listed in `README.md`; pulls entire GAP + REPL + Jupyter tree |
| `passagemath-gap-pkg-smallgrp-data` | `10.8.14` | 19.08 MiB | 3,378 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Transitive GAP data package (unused) |
| `passagemath-m4ri-m4rie` | `10.8.14` | 15.59 MiB | 30 | `>=3.11, <3.15` | **GPL-2.0-or-later** | `Matrix_mod2_dense` & `Matrix_gf2e_dense` (M4RI/M4RIE) |
| `passagemath-gap-pkg-primgrp-data` | `10.8.14` | 13.62 MiB | 46 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Transitive GAP data package (unused) |
| `passagemath-gap-pkg-factint-data` | `10.8.14` | 10.19 MiB | 2,287 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Transitive GAP data package (unused) |
| `passagemath-repl` | `10.8.14` | 2.14 MiB | 240 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Transitive via `passagemath-gap` (pulls `ipython`/`ipykernel`/`ipywidgets`) |
| `passagemath-environment` | `10.8.14` | 0.47 MiB | 104 | `>=3.11, <3.15` | **GPL-2.0-or-later** | Environment discovery glue |
| **Subtotal `passagemath-*`** | **17 wheels** | **1,688.13 MiB** | **23,717** | **`>=3.11, <3.15`** | **GPL-2.0-or-later** | — |
| `Cython`, `gmpy2`, `memory_allocator`, `mpmath`, `cysignals`, `ipywidgets`, `conway-polynomials`, `pexpect`, `ptyprocess` | 9 wheels | **21.16 MiB** | 622 | `cysignals`: `<3.14,>=3.9` | Mixed (GPLv3 `memory_allocator`, LGPL-3.0 `gmpy2`/`cysignals`, BSD, Apache-2.0, ISC) | Direct Sage runtime deps |
| **Total Sage ecosystem** | **26 wheels** (+ ~18 transitive Jupyter wheels) | **1,709.29 MiB** (~1.76 GiB w/ Jupyter) | **24,339+** | Effective: **`>=3.11, <3.14`** | **Copyleft (GPL-2.0+/GPLv3)** | — |

**Key Compliance & Packaging Risks:**
1. **License conflict:** `tamo-barg-lrc` is distributed under the permissive **MIT License** (`LICENSE`, `pyproject.toml:10`), yet importing it links 17 **GPL-2.0-or-later** `passagemath` wheels and **GPLv3** `memory_allocator` into the process.
2. **Python version contradiction:** `pyproject.toml:11` declares `requires-python = ">=3.10"`, but `passagemath-* 10.8.14` requires `>=3.11, <3.15`, and `cysignals 1.12.5` caps `<3.14`.
3. **Unnecessary transitive bloat:** Listing `passagemath-groups` in `README.md:24` drags in `passagemath-gap` + 3 GAP data packages + `passagemath-repl` + IPython/Jupyter (~400 MiB, >20,000 files) solely for optional `weight_distribution()` that already has a pure-Python fallback.

### 4.3 Import Time Benchmarks (3 warm-ups, 12 subprocess repeats)

| Import Statement | Median | p10 | p90 | Overhead vs Python (`pass`) |
|---|---:|---:|---:|---:|
| `pass` (baseline interpreter) | **11.7 ms** | 11.6 ms | 11.7 ms | 1.0× |
| `import numpy` | **87.2 ms** | 86.0 ms | 88.9 ms | 7.5× |
| `import networkx` | **162.2 ms** | 159.9 ms | 164.7 ms | 13.9× |
| `import sage.all__sagemath_modules` | **612.9 ms** | 603.8 ms | 618.8 ms | 52.4× |
| `import tamo_barg_lrc` (lazy PEP 562, still runs Sage bootstrap) | **608.5 ms** | 601.7 ms | 628.8 ms | 52.0× |
| `from tamo_barg_lrc import TamoBargCode, TamoBargCodeTwoSets` | **1,141.4 ms** | 1,135.4 ms | 1,148.4 ms | **97.6×** |

Top self-time contributors from `python -X importtime -c "import tamo_barg_lrc"` are modules **completely unused** by `tamo_barg_lrc`: `sage.rings.qqbar` (20.2 ms self / 24.3 ms cum), `sage.rings.rational_field` (17.1 ms), `sage.all__sagemath_flint` (15.9 ms self / 112.4 ms cum), `sage.combinat.integer_vector_weighted` (12.9 ms), `sage.libs.pari` (10.2 ms), `sage.rings.quotient_ring` (9.5 ms), `cypari2.pari_instance` (5.4 ms), `sage.rings.real_double_element_gsl` (4.7 ms).

---

## 5. Draft `ff_backend` Interface Specification

The minimal, backend-agnostic `Protocol` surface below covers **100% of L1–L5** across `src/`, `simulation/`, and `tests/`. A Sage adapter (`SageBackend`), a `galois`/NumPy adapter, a `python-flint` adapter, or a custom Cython/Rust backend can each implement this contract cleanly.

```python
"""Minimal finite-field, linear-algebra, polynomial, and RS-decoding backend protocol."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable


class DecodingError(Exception):
    """Raised when bounded-distance decoding fails or erasure count >= d."""


# ---------------------------------------------------------------------------
# L1: Finite Field & Field Elements
# ---------------------------------------------------------------------------

@runtime_checkable
class FieldElement(Protocol):
    """Immutable, hashable scalar in GF(p^m)."""

    def is_zero(self) -> bool: ...
    def to_int(self) -> int:
        """Deterministic bijection GF(p^m) -> 0..q-1 with 0 |-> 0 (base-p poly eval)."""
        ...
    def __add__(self, other: Any) -> FieldElement: ...
    def __sub__(self, other: Any) -> FieldElement: ...
    def __neg__(self) -> FieldElement: ...
    def __mul__(self, other: Any) -> FieldElement: ...
    def __truediv__(self, other: Any) -> FieldElement: ...
    def __pow__(self, exp: int) -> FieldElement: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...


@runtime_checkable
class FiniteField(Protocol):
    """Finite field GF(q), q = p^m for arbitrary prime p and m >= 1."""

    @property
    def order(self) -> int: ...            # q = p^m
    @property
    def characteristic(self) -> int: ...   # prime p
    @property
    def degree(self) -> int: ...           # m >= 1

    def zero(self) -> FieldElement: ...
    def one(self) -> FieldElement: ...
    def gen(self) -> FieldElement:
        """Power-basis generator alpha (1, alpha, ..., alpha^{m-1} spans GF(p^m)/GF(p))."""
        ...
    def primitive_element(self) -> FieldElement:
        """Generator g of cyclic multiplicative group GF(q)^* (order q - 1)."""
        ...
    def elem(self, value: int | FieldElement) -> FieldElement:
        """Coerce integer (mod p) or FieldElement into this field."""
        ...
    def from_int(self, code: int) -> FieldElement:
        """Inverse of FieldElement.to_int() for 0 <= code < q."""
        ...
    def elements(self) -> Sequence[FieldElement]:
        """All q elements in deterministic ascending to_int() order (0 first)."""
        ...
    def random_nonzero(self, rng: Any) -> FieldElement:
        """Uniform random element of GF(q) \\ {0} using caller-provided RNG."""
        ...


# ---------------------------------------------------------------------------
# L2: Vectors & Dense Matrices over GF(q)
# ---------------------------------------------------------------------------

@runtime_checkable
class FFVector(Protocol):
    """1D vector over GF(q); supports copy-on-demand immutability."""

    @property
    def field(self) -> FiniteField: ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int | slice) -> Any: ...
    def __setitem__(self, idx: int, val: FieldElement) -> None: ...
    def __iter__(self) -> Iterator[FieldElement]: ...
    def __add__(self, other: FFVector) -> FFVector: ...
    def __sub__(self, other: FFVector) -> FFVector: ...
    def __eq__(self, other: object) -> bool: ...

    def copy(self) -> FFVector: ...        # Always returns a mutable clone
    def set_immutable(self) -> None: ...
    def is_immutable(self) -> bool: ...
    def is_zero(self) -> bool: ...
    def hamming_weight(self) -> int: ...
    def nonzero_positions(self) -> list[int]: ...


@runtime_checkable
class FFMatrix(Protocol):
    """Dense 2D matrix over GF(q) (supports 0-row matrices for full-rate PCMs)."""

    @property
    def field(self) -> FiniteField: ...
    @property
    def nrows(self) -> int: ...
    @property
    def ncols(self) -> int: ...

    def set_immutable(self) -> None: ...
    def is_immutable(self) -> bool: ...
    def row(self, i: int) -> FFVector: ...
    def rows(self) -> list[FFVector]: ...
    def submatrix_rows(self, row_indices: Sequence[int]) -> FFMatrix: ...
    def submatrix_cols(self, col_indices: Sequence[int]) -> FFMatrix: ...
    def stack(self, other: FFMatrix) -> FFMatrix: ...
    def __neg__(self) -> FFMatrix: ...
    def matmul(self, other: FFMatrix) -> FFMatrix: ...
    def matvec(self, vec: FFVector) -> FFVector: ...       # M * v (column vector)
    def vecmat(self, vec: FFVector) -> FFVector: ...       # v * M (row vector)
    def rref(self) -> FFMatrix: ...                        # Exact Reduced Row Echelon Form
    def left_kernel_matrix(self) -> FFMatrix: ...          # Basis of {x : x * M = 0}
    def inverse(self) -> FFMatrix: ...                     # Square non-singular inverse (for unencode)


# ---------------------------------------------------------------------------
# L3: Univariate Polynomials over GF(q)[x]
# ---------------------------------------------------------------------------

@runtime_checkable
class FFPoly(Protocol):
    """Immutable univariate polynomial in GF(q)[x], ascending coefficients."""

    @property
    def field(self) -> FiniteField: ...
    @property
    def degree(self) -> int: ...           # -1 for the zero polynomial

    def coeffs(self, length: int | None = None) -> list[FieldElement]:
        """Ascending coeffs [c_0, ..., c_d], zero-padded to `length` if given."""
        ...
    def shift(self, k: int) -> FFPoly: ...                 # x^k * self (k >= 0)
    def eval(self, x: FieldElement) -> FieldElement: ...
    def eval_many(self, xs: Sequence[FieldElement]) -> FFVector: ...
    def __add__(self, other: FFPoly) -> FFPoly: ...
    def __sub__(self, other: FFPoly) -> FFPoly: ...
    def __mul__(self, other: FFPoly) -> FFPoly: ...
    def scale(self, c: FieldElement) -> FFPoly: ...
    def divmod(self, other: FFPoly) -> tuple[FFPoly, FFPoly]: ...
    def __mod__(self, other: FFPoly) -> FFPoly: ...
    def __eq__(self, other: object) -> bool: ...


# ---------------------------------------------------------------------------
# L4: Plain Reed-Solomon Codec (Replaces Sage GRS + BW + Gao + ErrorErasure)
# ---------------------------------------------------------------------------

@runtime_checkable
class RSCodec(Protocol):
    """Plain [n, k, d = n - k + 1] Reed-Solomon code on arbitrary distinct eval points."""

    @property
    def field(self) -> FiniteField: ...
    @property
    def evaluation_points(self) -> Sequence[FieldElement]: ...
    @property
    def length(self) -> int: ...
    @property
    def dimension(self) -> int: ...
    @property
    def minimum_distance(self) -> int: ...                 # n - k + 1

    def parity_check_matrix(self) -> FFMatrix:
        """(n - k) x n PCM (0 x n when k == n; never crashes on full-rate codes)."""
        ...
    def has_zero_syndrome(self, word: FFVector) -> bool:
        """O(n * (n - k)) fast syndrome check; True iff word is a valid RS codeword."""
        ...
    def decode_errors(self, received: FFVector) -> FFVector:
        """Bounded-distance RS error decoder (radius t = (d - 1) // 2).

        Semantics:
        - Never mutates `received`.
        - Raises ValueError if len(received) != n or field mismatch.
        - Returns copy of `received` if k == n or syndrome == 0.
        - Returns unique codeword c with wt(c - received) <= t if one exists.
        - Otherwise raises DecodingError (including when deg(h) >= k; never leaks ValueError).
        """
        ...
    def decode_errors_erasures(
        self,
        received: FFVector,
        erasure_mask: Sequence[int],
    ) -> FFVector:
        """Joint RS error-and-erasure decoder (2*tau + e <= d - 1).

        Semantics:
        - Never mutates `received` or `erasure_mask`.
        - Raises ValueError if len(received) != n or len(erasure_mask) != n.
        - Raises DecodingError if erasure count e >= d.
        - If e == 0 and syndrome == 0: returns copy of `received`.
        - If e == d - 1: unique degree-<k interpolation on the k non-erased positions.
        - Otherwise decodes punctured [n - e, k] RS code (radius (d - 1 - e) // 2) and
          re-evaluates recovered polynomial at all n evaluation points; raises DecodingError
          on failure.
        """
        ...


# ---------------------------------------------------------------------------
# Top-Level Backend Factory (L1–L5)
# ---------------------------------------------------------------------------

@runtime_checkable
class FFBackend(Protocol):
    """Factory & algorithmic primitives replacing all Sage imports."""

    # L1
    def gf(self, q: int, *, repr_mode: str = "poly") -> FiniteField: ...

    # L2
    def vector(self, field: FiniteField, entries: Sequence[Any]) -> FFVector: ...
    def zero_vector(self, field: FiniteField, length: int) -> FFVector: ...
    def matrix(
        self,
        field: FiniteField,
        nrows: int,
        ncols: int,
        rows: Sequence[Sequence[Any]] | None = None,
    ) -> FFMatrix: ...
    def enumerate_subspaces(
        self, prime_p: int, ambient_dim: int, dim: int
    ) -> Iterator[list[tuple[int, ...]]]:
        """Yield each `dim`-dim subspace of GF(p)^ambient_dim as its p^dim coordinate tuples."""
        ...
    def iter_vector_space(self, field: FiniteField, dim: int) -> Iterator[FFVector]: ...
    def random_vector(self, field: FiniteField, length: int, rng: Any) -> FFVector: ...
    def random_error_vector(
        self, field: FiniteField, length: int, positions: Sequence[int], rng: Any
    ) -> FFVector: ...

    # L3
    def poly(self, field: FiniteField, coeffs: Sequence[Any]) -> FFPoly: ...
    def vanishing_poly(self, field: FiniteField, roots: Sequence[FieldElement]) -> FFPoly:
        """Monic prod_{a in roots}(x - a) via balanced product tree."""
        ...
    def interpolate(
        self,
        field: FiniteField,
        xs: Sequence[FieldElement],
        ys: Sequence[FieldElement],
    ) -> FFPoly:
        """Unique degree < len(xs) polynomial p with p(xs[i]) == ys[i]."""
        ...

    # L4
    def rs_codec(self, field: FiniteField, eval_points: Sequence[FieldElement], k: int) -> RSCodec: ...

    # L5 (pure Python utilities; no external dependency needed)
    def divisors(self, n: int) -> list[int]: ...
```

### Design Notes on `ff_backend`

1. **Eliminates `GF(2)` overhead in erasure vectors:** Erasure masks in `RSCodec.decode_errors_erasures` accept any `Sequence[int]` (e.g., `list[int]` or `np.ndarray[uint8]`), eliminating the repeated `vector(GF(2), ...)` allocations inside `TamoBargIterativeEEDecoder` (`tamo_barg_two_sets.py:723,766`) and `TamoBargGlobalEEDecoder` (`tamo_barg_two_sets.py:845`). Public API compatibility wrappers can still return a binary `FFVector` over `GF(2)` where tests check `out_era.parent() == VectorSpace(GF(2), n)` (`test_tamo_barg_two_sets.py:279`).
2. **Batch-friendly by design:** `FFPoly.eval_many` and `RSCodec.has_zero_syndrome` allow vectorizing generator-matrix construction (`[poly(alpha) for alpha in alphas]`) and local syndrome checks without per-element Python loops.
3. **Zero Sage coupling in L4 code hierarchy:** `TamoBargCode`, `TamoBargCodeTwoSets`, their encoders, and all 8 decoders only use ~150 LOC of `AbstractLinearCode`/`Encoder`/`Decoder` plumbing (registry dicts, `encoder()`/`decoder()` caching, `encode()`/`unencode()` via info-set inverse, `decode_to_code()` dispatch). Replacing those 3 Sage base classes with lightweight native classes removes the last structural lock-in to Sage's `Parent`/`Category` hierarchy.
4. **RNG unification:** Passing an explicit `rng` (seeded from `seed_simulation_rngs`) removes global `sage.misc.randstate.set_random_seed` state and guarantees thread/process-safe reproducibility.

---

## 6. Acceptance Checklist Verification

- [x] **Complete symbol table with `file:line`** — Section 1 (all 21 explicit `sage.*` imports + all L1–L5 methods, attributes, and implicit operators, audited via AST across all 22 Python files).
- [x] **Field-support requirement statement** — Section 2 (exact $\mathbb{F}_2$, $\mathbb{F}_{2^m}$, odd $\mathbb{F}_p$, and odd $\mathbb{F}_{p^m}$ requirements + concrete Sage backend matrix).
- [x] **Sage GRS decoder behaviour spec** — Section 3 (verified directly from `grs_code.py`, `decoder.py`, `encoder.py` + empirical edge-case probe covering `ValueError` leak on `deg(h) >= k`, BW/Gao 200/200 equivalence, and EE puncturing).
- [x] **Footprint table** — Section 4 (`du -sh`, all 17 `passagemath-*` + 9 support wheels, licenses, Python-version metadata, and 12-repeat import-time benchmarks).
- [x] **Draft `ff_backend` protocol** — Section 5 (minimal typed `Protocol` specification covering L1–L5).
