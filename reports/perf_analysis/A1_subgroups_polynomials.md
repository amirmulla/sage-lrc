# Agent A1 Report — `subgroups.py` & `polynomials.py` Algorithm, Performance & Sage-Replacement Analysis

## 1. Environment & Benchmark Methodology

- **Machine**: 2 vCPU `Intel(R) Xeon(R) Platinum 8481C` (`x86_64`), Linux (Debian 12), `/home/antigravity/.venv-lrc/bin/python` (`Python 3.11.2`, modular `passagemath` 10.8.x).
- **Methodology**: Fixed seeds (`random.seed(42)`, `np.random.seed(42)`, `set_random_seed(42)`). Warm-up $\ge 3$ calls ($\ge 1$ for $>1\text{ s}$ calls); repeats $\ge 20$ ($5\text{--}15$ for $>0.15\text{ s}$, single-shot for $>20\text{ s}$). Timings via `time.perf_counter` reporting **median (p10–p90)**; peak Python heap via `tracemalloc`; profiles via `cProfile`/`pstats`. Field construction is excluded from per-call timings. Log-log OLS slope fitted across $q \in \{16, 64, 256, 1024, 4096\}$ (where tractable).
- **Two-set LRC parameter sweep**: For $q = 2^m$ ($m \in \{4, 6, 8, 10, 12\}$), canonical additive subgroup size $|H| = 2^{m/2} \in \{4, 8, 16, 32, 64\}$, partition size $s_p = q/|H| = 2^{m/2}$, local distance $d_{\text{local}} = 3$, locality $r = |H| - 2 \in \{2, 6, 14, 30, 62\}$, single-set basis size $k_{\text{single}} = r \cdot s_p \in \{8, 48, 224, 960, 3968\}$, and shifted second subgroup $g^{m/2} H$.
- **Scratch scripts & logs**: [bench_subgroups.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A1/bench_subgroups.py), [bench_polynomials.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A1/bench_polynomials.py), [bench_polynomials_p2.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A1/bench_polynomials_p2.py), [bench_polynomials_p3.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A1/bench_polynomials_p3.py), [bench_prototypes.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A1/bench_prototypes.py).

---

## 2. Executive Design Narrative & Critique ($\le 1$ page)

`subgroups.py` (11 functions) and `polynomials.py` (7 functions) form the algebraic foundation of `tamo-barg-lrc`: constructing additive/multiplicative subgroups and coset partitions of $\mathbb{F}_q$, building Tamo–Barg "good" polynomials $g(x)$ and encoding polynomial bases $\{x^i g(x)^j\}$, intersecting two partition encoding subspaces for two-set LRCs, evaluating message polynomials, and recovering erasures via Lagrange interpolation.

**What works well:**
1. **Algebraic clarity & generality**: Both modules work uniformly across prime fields $\mathbb{F}_p$, binary extension fields $\mathbb{F}_{2^m}$, and odd-characteristic extensions $\mathbb{F}_{p^m}$, with deterministic element/coset ordering anchored by `field_element_to_int`.
2. **Stacked-kernel + reverse-RREF intersection**: `intersect_encoding_bases` cleanly yields a strictly degree-minimal basis of $\text{span}(A) \cap \text{span}(B) \cap \mathbb{F}_q[x]_{<n}$ using half the columns of classical block-Zassenhaus and delegates linear algebra over $\mathbb{F}_{2^m}$ ($m \le 16$) to Sage's M4RIE C backend (`Matrix_gf2e_dense`).

**Three structural performance pathologies uncovered by profiling:**
1. **Cross-backend impedance mismatch (NTL `Polynomial_ZZ_pEX` $\leftrightarrow$ Givaro `FiniteField_givaroElement` $\leftrightarrow$ M4RIE `Matrix_gf2e_dense`)**: Every polynomial ring over $\mathbb{F}_{p^m}$ ($m \ge 2$) uses NTL's `ZZ_pEX` (`sage/rings/polynomial/polynomial_ring.py:2650-2663`), whereas field elements live in Givaro and matrices in M4RIE. In `intersect_encoding_bases` at $q=256$, **89.5% of total time** (`914.8 / 1021.3 ms`) is pure Python/Cython serialization (`padded_list` 23.0% + `poly_ring(coeffs)` 66.5%), while all M4RIE linear algebra combined (`left_ker` + `ker*A` + `rev_rref`) takes only **6.3%** (`64.5 ms`). Worse, `TamoBargCodeTwoSets.__init__` invokes `intersect_encoding_bases` **twice** (once inside `max_dimension()` and again in `code.encoder("VectorEncoder")`), and `generator_matrix()` then spends **13.57 s** evaluating the reconstructed `Polynomial_ZZ_pEX` objects at $n$ points. Bypassing polynomial objects via evaluation-domain dual-GRS parity checks computes the full generator matrix in **26.7 ms** at $q=256$ (**612× faster**).
2. **Unnecessary Sage object materialization in hot paths**:
   - `find_subspaces` / `find_additive_subgroups` materialize full Sage `FreeModule_submodule_field` objects and iterate them via `FreeModule.__iter__` (`linear_combination_of_basis`), costing **3.8 ms per subspace** (857k Python calls for 651 2D subspaces of $\mathbb{F}_{64}$, taking `2.06 s`). A pure bitmask RREF enumerator produces identical output in **3.71 ms** (**556× faster**).
   - `interpolate_erasures` calls `PolynomialRing_field.lagrange_polynomial`, which runs divided differences in Python and then expands the polynomial into NTL `Polynomial_ZZ_pEX` just to evaluate at $e$ erased points. Direct Newton divided-difference evaluation in field arithmetic is **124×–419× faster** at local coset sizes and **23×–118× faster** at global sizes.
   - `field_element_to_int` / `sort_field_elements` account for **89%** of `list_subgroup_cosets` and **74%** of `list_multiplicative_subgroups` at $q=4096$, and are called redundantly in `list_canonical_additive_subgroups` (whose construction is already sorted as `0 .. p^i - 1`).
3. **Dense polynomial algorithms ignoring LRC sparsity & algebraic identities**: `build_annihilator_polynomial` builds $\prod_{\alpha \in \mathbb{F}_q}(x-\alpha)$ via a product tree (`1,423 ms` at $q=4096$) instead of $x^q - x$ ($O(1)$); `build_good_polynomial` multiplies $s$ linear factors (`16.5 ms` at $|H|=64$) instead of using the linearized $p$-polynomial recurrence in $O(\log_p^2 s)$ field ops (`0.84 ms`, **19.8× faster**); and `build_algebra_basis` multiplies densely even for $g(x) = x^{|H|}$ or $(\log_p|H|+1)$-sparse subspace polynomials.

---

## 3. Top-3 Hotspots Ranked with Evidence

| Rank | Function (and caller impact) | Evidence (Measured / Profiled) | Root Cause | Concrete Speed-up |
|---|---|---|---|---|
| **1 (P0)** | [`intersect_encoding_bases`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L158-L231) (called **2×** per `TamoBargCodeTwoSets` + `encoder` construction) | `q=256`: **1,262 ms** (`cProfile`: `poly_ring._element_constructor_` 59.5%, `padded_list` 28.7%, M4RIE lin-alg 6.3%). `q=1024`: **33.1 s** (`poly_build` 48.4%, `left_ker` 28.2%, `padded_list` 18.7%, peak 78.2 MB). Accounts for **97.2%** of `code_init + enc_init` and **72.8%** of full `q=256, k=20` construction. | Serializing dense $k \times n$ matrices back and forth between NTL `Polynomial_ZZ_pEX` and Givaro/M4RIE; plus $O(n^3)$ `left_kernel_matrix` on $(k_a+k_b) \times n$ at $q \ge 1024$. | **47×–612×** via M4RIE/evaluation-domain dual-GRS PCM kernel (`26.7 ms` at $q=256$, `907 ms` at $q=1024$); **1.57×** just switching to dual $(U^\perp + V^\perp)^\perp$ at $q=1024$. |
| **2 (P0)** | [`interpolate_erasures`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L259-L298) (called inside **every** local/global erasure decode loop in `TamoBargErasureDecoder`, `TamoBargIterativeEEDecoder`, `TamoBargGlobalEEDecoder`) | Local cosets ($e=2$): `q=256, s=14` takes **6.16 ms**; `q=4096, s=62` takes **63.86 ms**. Global (`GF(1024), n=512, s=384, e=128`): **2,185 ms**. | `sage/rings/polynomial/polynomial_ring.py:2524-2538` computes divided differences in Python, then expands a monomial `Polynomial_ZZ_pEX` via $s$ NTL polynomial multiplications just to evaluate at $e$ points. | **124×–419×** locally (`0.015 ms` at $s=14$, `0.516 ms` at $s=62$) and **23×–118×** globally (`95.5 ms` at $s=384$) via direct Newton-form evaluation or barycentric Lagrange. |
| **3 (P1)** | [`find_additive_subgroups`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L117-L174) & [`find_subspaces`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L91-L114) (called in `simulation/runner.py` when `--add-idx` overrides canonical subgroup) | `GF(64), s=4` (651 subspaces): **2,063 ms** (`find_subspaces` 795 ms); `GF(64), s=8` (1,395 subspaces): **6,016 ms**; `GF(128), s=4` (2,667 subspaces): **7,848 ms** (12.7 MB). `cProfile` on `GF(64), s=4`: 857k calls — 56% `FreeModule.__iter__` (`linear_combination_of_basis`), 42% `FreeModule_submodule_field.__init__`. | Materializing heavyweight Sage `FreeModule` objects + iterating elements via `linear_combination_of_basis` instead of enumerating raw RREF basis bitmasks. | **365×–938×** via integer bitmask RREF enumerator (`3.71 ms` for `GF(64), s=4`; `7.27 ms` for `GF(64), s=8`; `58.8 ms` for `GF(256), s=4` vs ~31.8 s extrapolated). |

*(Secondary P1 hotspots: `build_encoding_basis` `2,446 ms` and `build_annihilator_polynomial` `1,423 ms` and `build_algebra_basis` `1,282 ms` at $q=4096$; `evaluate_message_polynomial` `480.8 ms` at $q=1024, k=900$; `sort_field_elements` `8.80 ms` at $q=4096$ dominating `list_subgroup_cosets` at 89% and `list_multiplicative_subgroups` at 74%.)*

---

## 4. Sage Internals & Replacement Mapping

Verified directly in `/home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/`:

| Sage Primitive Used | Installed Backend & Source Path | Behaviour Across Field Families | Pure-NumPy / `galois` / FLINT Equivalent |
|---|---|---|---|
| `GF(q)` (`finite_field_constructor.py`) | `FiniteField_prime_modn` ($q=p$); `FiniteField_givaro` (Givaro C++ Zech-log tables, $q=p^m < 2^{16}$); `FiniteField_ntl_gf2e` (NTL `GF2E`, $q=2^m \ge 2^{16}$). | Zech-log $O(1)$ ops for Givaro; NTL carryless arithmetic for $m \ge 16$. | `galois.GF(q)` (Numba/LUT); pure NumPy log/antilog LUT (`uint16` for $q \le 65536$); FLINT `fq_nmod_ctx` (`python-flint`). |
| `elem.to_integer()` (`element_base.pyx:1049`) | `FinitePolyExtElement.to_integer` lifts polynomial rep to $\mathbb{Z}$ and evaluates at $p$; `IntegerMod_int` lacks `to_integer` (falls back to `int(elem)`). | Allocates Sage `Integer` then Python `int`; ~0.27–0.88 µs/elem. | **No-op** in integer-encoded `galois` / NumPy arrays (elements *are* `uint16`/`int` in `0..q-1`); FLINT `fq_nmod` bit-unpack. |
| `field.multiplicative_generator()` / `primitive_element()` (`finite_field_base.pyx:689`) | Both are **aliases of the exact same method** (`FiniteField.multiplicative_generator`); cached on field object. | Returns Conway-polynomial primitive root (usually `field.gen()`). | `galois.GF(q).primitive_element`; precomputed primitive element in NumPy LUT; FLINT `fq_nmod_ctx_gen` (note: check primitivity if non-Conway). |
| `VectorSpace(GF(p), m).subspaces(d)` (`free_module.py:4867`) | Iterates `reduced_echelon_matrix_iterator` (`sage/matrix/echelon_matrix.pyx:17`, Cython), then wraps **each** matrix via `self.subspace((m * b).rows())` (`FreeModule_submodule_field.__init__`). | ~1.2 ms/subspace for `subspace()` + ~2.5 ms/subspace for `__iter__`. | Pure Python/NumPy bitmask RREF generator over `itertools.combinations(range(m), d)` (~5 µs/subspace, **500×+ faster**). |
| `divisors(q - 1)` (`sage/arith/misc.py:1490`) | Converts Python `int` to Sage `Integer`, calls `Integer.divisors()` (`integer.pyx:3137`, PARI/FLINT factorization + divisor product). | Negligible (`< 0.1 ms` for $q \le 2^{20}$). | 10-line pure Python $O(\sqrt{q})$ trial division or `sympy.divisors` / `flint.fmpz.factor`. |
| `PolynomialRing(GF(q), 'x')` (`polynomial_ring.py:2650-2663`) | `GF(2)` $\to$ `Polynomial_GF2X` (NTL); `GF(p)` $\to$ `Polynomial_zmod_flint` (FLINT); **`GF(p^m)` ($m \ge 2$) $\to$ `Polynomial_ZZ_pEX` (NTL `ZZ_pEX`)**. | Dense C++ polynomial ops; **expensive Givaro $\leftrightarrow$ NTL `ZZ_pE` conversion** on construction, `list()`, `padded_list()`, `__call__()`. | `galois.Poly` (NumPy 1D coefficient array); NumPy poly arithmetic with GF LUT; FLINT `fq_nmod_poly` (native C, no cross-library conversion!). |
| `sage.misc.misc_c.prod` (`misc_c.pyx:72`) | Cython balanced binary product tree (`recursion_cutoff=5`). | $O(M(s) \log s)$ poly ops; classical $O(s^2)$ for small $s \le 64$ in NTL. | Pure Python divide-and-conquer product tree (6 lines) or linearized recurrence for subspaces; FLINT `fq_nmod_poly_product_roots`. |
| `matrix(F, ...)` (`matrix_space.py:210-300`), `left_kernel_matrix`, `echelon_form` (`matrix2.pyx:4274,4966,8563`, `matrix_gf2e_dense.pyx:928`) | `GF(2)` $\to$ `Matrix_mod2_dense` (M4RI); **`GF(2^m)` ($m \le 16$) $\to$ `Matrix_gf2e_dense` (M4RIE: Newton-John / PLE / Strassen)**; `GF(p)` $\to$ LinBox; **`GF(p^m)` ($p>2, m>1$) & `GF(2^m)` ($m \ge 17$) $\to$ `Matrix_generic_dense` (Python-object fallback!)**. | M4RIE is fast (`64.5 ms` total lin-alg at $q=256$); odd-char `GF(81)` falls back to slow `Matrix_generic_dense` (meataxe absent) and triggers a modular-passagemath `laurent_polynomial_ring` import bug (`matrix_space.py:290`) if imported standalone. | `galois.GF(q)` matrix RREF / `null_space()` (Numba); FLINT `fq_nmod_mat` (`rref`, `nullspace`); pure NumPy/Cython/Numba Gaussian elimination over LUT. |
| `poly_ring.lagrange_polynomial` (`polynomial_ring.py:2361-2538`) | Pure Python `divided_difference()` ($O(s^2)$ field ops) + Horner loop constructing `Polynomial_ZZ_pEX` ($s$ NTL poly muls). | 95%+ time wasted building the monomial `Polynomial_ZZ_pEX`. | 15-line pure Python/NumPy direct Newton divided-difference evaluator or barycentric Lagrange (zero poly-ring overhead); FLINT `fq_nmod_poly_interpolate`. |

---

## 5. Mandatory Per-Function Analysis Tables (All 18 Functions)

### 5.1 `tamo_barg_lrc.subgroups` (11 Functions)

#### 1. `subgroups._validate_finite_field(field: Any) -> None`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups._validate_finite_field`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L28-L31) — Guard raising `ValueError` if `field` fails `is_finite()`. |
| **Algorithm** | Duck-typed `hasattr(field, "is_finite") and field.is_finite()`. |
| **Complexity** | Time $O(1)$; Space $O(1)$. |
| **Sage primitives** | `FiniteSets.ParentMethods.is_finite()` (`sage/categories/finite_sets.py:59`, pure Python returning `True`); ~0.2 µs. |
| **Measured** | `q=16..4096`: median **0.00020–0.00021 ms** (p10 `0.00019`, p90 `0.00026–0.00036`), slope `0.00`, peak `0.06 KB`. |
| **Bottleneck** | None (negligible ~200 ns overhead). |
| **Correctness risks** | **Confirmed bug**: `Zmod(6)` (finite ring, NOT a field) passes `_validate_finite_field(Zmod(6))` without error because `is_field()` is not checked! Subsequent calls to `field.degree()` / `multiplicative_generator()` then fail with cryptic `AttributeError`/`NotImplementedError`. |
| **Better algorithm** | Check `hasattr(field, "is_finite") and field.is_finite() and hasattr(field, "is_field") and field.is_field()` (or `isinstance` check on custom `FiniteField` class). |
| **Sage-replacement difficulty** | **1 / 5** — `isinstance(field, FieldMeta)` (`galois.FieldArray` subclass / custom field object / `flint.fq_nmod_ctx`). |
| **Priority** | **P2** (correctness fix only; zero perf impact). |

#### 2. `subgroups.field_element_to_int(elem: Any) -> int`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.field_element_to_int`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L34-L47) — Deterministically map a finite-field element to `0 .. q - 1`. |
| **Algorithm** | Branch on `elem.is_zero()` $\to 0$, else `int(elem.to_integer())` (polynomial rep evaluated at $p$), else `int(elem)`. |
| **Complexity** | Time $O(m)$ per element ($m = \log_p q$, polynomial evaluation at $p$); $O(q \cdot m)$ for a full field batch. Space $O(1)$ per element. |
| **Sage primitives** | `Element.is_zero()` (`element.pyx:1044`), `FinitePolyExtElement.to_integer()` (`element_base.pyx:1049`, Cython $\to$ Sage `Integer`), plus 2× `hasattr` per call. Note: `IntegerMod_int` (`GF(p)`) has no `to_integer` and falls through to `int(elem)`. |
| **Measured** | Single element: **0.37–0.40 µs**. Full-field batch ($n=q$): `q=16`: `0.0046 ms` (p10 `0.0045`, p90 `0.0047`, 0.4 KB); `q=64`: `0.0172 ms`; `q=256`: `0.0682 ms`; `q=1024`: `0.2783 ms` (p90 `4.29 ms`, 32.8 KB); `q=4096`: **3.604 ms** (p10 `1.10`, p90 `5.14`, 152.4 KB). Empirical batch slope vs $q$: **1.161** (theory $O(q \log q)$). |
| **Bottleneck** | Python `hasattr` ×2 + Givaro $\to$ Sage `Integer` $\to$ Python `int` double boxing (confirmed in `cProfile` of `list_subgroup_cosets`: 81,915 `hasattr` calls alone = 21% of total runtime!). |
| **Correctness risks** | Ordering depends on `field.modulus()` (Conway polynomial by default). Mixing elements from distinct fields silently compares incompatible encodings. |
| **Better algorithm** | In Sage: precompute/cache a `dict` or vectorize via `elem._int_repr()` / field-level mapping; **without Sage**: use integer-encoded polynomial basis (`uint16` / `galois.GF(q)`) where `field_element_to_int` is an $O(1)$ identity no-op (`int(x)` or array view). |
| **Sage-replacement difficulty** | **1 / 5** — Native integer representation in `galois` (`int(elem)` / `arr.view(np.ndarray)`) or NumPy LUT. |
| **Priority** | **P1** (high call volume inside `sort_field_elements`, `list_subgroup_cosets`, `list_multiplicative_subgroups`, `build_encoding_basis`). |

#### 3. `subgroups.sort_field_elements(elements: Iterable[Any]) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.sort_field_elements`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L50-L59) — Sort field elements deterministically by `field_element_to_int`. |
| **Algorithm** | Python Timsort (`sorted(elements, key=field_element_to_int)`), invoking key function once per element (Schwartzian transform). |
| **Complexity** | Time $O(N \log N + N \cdot m)$ for $N$ elements; Space $O(N)$. |
| **Sage primitives** | Indirectly `field_element_to_int` ($N$ calls to `is_zero` + `to_integer`). |
| **Measured** | ($N=q$ shuffled) `q=16`: `0.0047 ms` (`0.0046–0.0049`, 0.3 KB); `q=64`: `0.0183 ms` (`0.0181–0.0185`, 0.6 KB); `q=256`: `0.0746 ms` (`0.0739–0.168`, 4.1 KB); `q=1024`: `0.3697 ms` (`0.345–6.45`, 48.0 KB); `q=4096`: **8.798 ms** (`1.64–11.71`, 216.0 KB). Empirical slope vs $q$: **1.305** (theory $O(q \log q)$). Comparison: `np.sort(uint16)` takes `0.00082 ms` at $q=16$ (**5.7×**), `0.00101 ms` at $q=256$ (**74×**), `0.00873 ms` at $q=4096$ (**1,008× faster**). |
| **Bottleneck** | `field_element_to_int` key extraction (~80% of runtime per `cProfile`), not comparisons; high p90 variance at $q \ge 1024$ due to Sage object allocation + GC/VM jitter (`GC ON p90=7.08 ms` vs `GC OFF p90=4.01 ms`). |
| **Correctness risks** | No deduplication or parent-field homogeneity check. |
| **Better algorithm** | Integer-encoded NumPy array + `np.sort` (**74×–1,008× faster**); for full-field sorting (`sort_field_elements(field)`), construct directly in integer order `0..q-1` in $O(q)$ without sorting at all! |
| **Sage-replacement difficulty** | **1 / 5** — `np.sort(arr)` on integer-encoded `galois` / NumPy arrays. |
| **Priority** | **P1** (dominates `list_subgroup_cosets` at 89% and `list_multiplicative_subgroups` at 74%). |

#### 4. `subgroups.is_closed_under_addition(subgroup: Iterable[Any]) -> bool`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.is_closed_under_addition`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L62-L88) — Verify non-empty subset contains `0` and is closed under pairwise `+`. |
| **Algorithm** | Materialize `set`, verify `zero in elem_set`, then check `(a + b) in elem_set` for all $0 \le i \le j < s$ with early exit. |
| **Complexity** | Time $O(s^2)$ worst case (valid subgroup of size $s$), $O(s)$ best case (early exit on non-subgroup); Space $O(s)$. |
| **Sage primitives** | `elem.parent().zero()`, Givaro/prime-field `+` (`FiniteField_givaroElement._add_`), `__hash__` / `__eq__` for set lookup. |
| **Measured** | Valid subgroups ($s = \sqrt{q}$): `s=4`: `0.0025 ms` (`0.0023–0.0031`, 0.5 KB); `s=8`: `0.0055 ms`; `s=16`: `0.0174 ms`; `s=32`: `0.1040 ms` (`0.103–0.423`, 2.8 KB); `s=64`: **0.4037 ms** (`0.394–6.43`, 3.0 KB); `s=256 (GF(256))`: **18.37 ms** (`9.33–33.1`, 10.4 KB). Empirical slope vs $s$: **1.895** (theory $O(s^2)$). |
| **Bottleneck** | $s(s+1)/2$ Python-loop field additions and hash lookups. |
| **Correctness risks** | **Confirmed edge case**: silently deduplicates via `set(subgroup)` — `is_closed_under_addition([0, 0, 1, 1])` returns `True`! In `tamo_barg_two_sets.py:193-197`, passing a custom subgroup with duplicates of length `target_size` passes both length check and closure check unless uniqueness is validated. Also lacks cardinality check ($s$ must be a power of $p$). |
| **Better algorithm** | First check `len(elem_set) == len(input)` (if sequence) and `s` is a power $p^t$ ($O(\log s)$ reject!); then extract a $\mathbb{F}_p$-basis by Gaussian elimination over $\mathbb{F}_p^m$ (for $p=2$, XOR linear basis in $O(s \cdot m)$ time!) and check `len(elem_set) == p ** rank` — **reduces $O(s^2)$ to $O(s \log q)$**! |
| **Sage-replacement difficulty** | **1 / 5** — Pure Python/NumPy XOR basis (for $p=2$) or $\mathbb{F}_p$ RREF. |
| **Priority** | **P2** (called only during custom-subgroup validation or bruteforce search). |

#### 5. `subgroups.find_subspaces(dim: int, ambient_dim: int, base_field: Any) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.find_subspaces`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L91-L114) — Enumerate all `dim`-dimensional subspaces of $\mathbb{F}_p^{\text{ambient\_dim}}$. |
| **Algorithm** | Delegates to `VectorSpace(GF(p), m).subspaces(d)` (`reduced_echelon_matrix_iterator` wrapped in `FreeModule.subspace`), materializing all $\binom{m}{d}_p$ subspaces into a `list`. |
| **Complexity** | Count is Gaussian binomial $\binom{m}{d}_p = \prod_{i=0}^{d-1} \frac{p^{m-i}-1}{p^{d-i}-1} = \Theta(p^{d(m-d)})$. Time $O(\binom{m}{d}_p \cdot d \cdot m)$ (with huge Sage object constant ~1.2 ms/subspace); Space $O(\binom{m}{d}_p \cdot d \cdot m)$ (eager list). |
| **Sage primitives** | `GF(p)`, `VectorSpace` (`free_module.py`), `FreeModule_ambient_field.subspaces` (`free_module.py:4867`) $\to$ `reduced_echelon_matrix_iterator` (`echelon_matrix.pyx:17`, Cython) + `self.subspace()` (`free_module.py:4822` $\to$ `submodule` $\to$ `span` $\to$ `FreeModule_submodule_field.__init__` $\to$ `Sequence`). |
| **Measured** | Fixed $d=2$ over $\mathbb{F}_{2^m}$: `m=4` (35): **42.3 ms** (`38.7–48.9`, 103 KB); `m=5` (155): **196.4 ms** (`182–224`, 407 KB); `m=6` (651): **794.9 ms** (`790–843`, 1.73 MB); `m=7` (2,667): **3,025 ms** (`2,472–3,066`, 7.04 MB). Mid-dim `m=6, d=3` (1,395): **2,298 ms** (`1,845–2,327`, 3.82 MB). Empirical slope vs $\binom{m}{2}_2$: **0.984** (theory `1.000`). |
| **Bottleneck** | `cProfile` (`m=6, d=2`): **95.6%** (`2.245 / 2.347 s`) is `FreeModule_submodule_field.__init__` + `Sequence` + `isinstance` (398k calls for 651 matrices); `reduced_echelon_matrix_iterator` itself is `< 2%`! |
| **Correctness risks** | Eager `list(...)` causes **OOM / multi-hour hang** with no guard: for `m=8, d=4` ($\binom{8}{4}_2 = 200,787$), runtime exceeds ~5 min and memory > 500 MB; for `m=12, d=6`, $\binom{12}{6}_2 = 2.31 \times 10^{11}$ is physically impossible. |
| **Better algorithm** | Yield raw RREF basis matrices (or $d$-tuples of integer bitmasks for $p=2$) lazily via generator without `FreeModule` wrapper; support `index` / `unrank` for $O(d \cdot m)$ direct lookup of the $k$-th subspace (used by `simulation/runner.py --add-idx`)! |
| **Sage-replacement difficulty** | **1 / 5** — 20-line pure Python/NumPy RREF generator over `itertools.combinations(range(m), d)`. |
| **Priority** | **P1** (severe cliff at $m \ge 6$; only used by `find_additive_subgroups` & tests). |

#### 6. `subgroups.find_additive_subgroups(field: Any, size: int) -> list[list[Any]]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.find_additive_subgroups`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L117-L174) — Enumerate all sorted additive subgroups of `field` of cardinality `size` $= p^d$. |
| **Algorithm** | Validates `size` $= p^d$, calls `find_subspaces(d, m, field)`, iterates each `vec in subspace` mapping coordinates via $\sum_j c_j \alpha^j$, sorts each subgroup and sorts outer list. |
| **Complexity** | Time $O(\binom{m}{d}_p \cdot (s \cdot m + s \log s) + \binom{m}{d}_p \log \binom{m}{d}_p \cdot s)$ where $s = p^d$; Space $O(\binom{m}{d}_p \cdot s)$. |
| **Sage primitives** | `find_subspaces` + `FreeModule.__iter__` (`free_module.py:2412`) $\to$ `linear_combination_of_basis` (`free_module.py:7715`), `field(int(coord))`, `sort_field_elements`. |
| **Measured** | Fixed $s=4$ ($d=2$) over $\mathbb{F}_{2^m}$: `m=4` (35): **95.9 ms** (`90.7–105`, 214 KB); `m=5` (155): **526.4 ms** (`266–675`, 799 KB); `m=6` (651): **2,063 ms** (`1,654–2,314`, 3.12 MB); `m=7` (2,667): **7,848 ms** (`6,488–8,911`, 12.7 MB); `m=6, d=3` (1,395): **6,016 ms** (`5,718–6,294`, 6.87 MB). Slopes: **1.011** vs $\binom{m}{2}_2$ (theory `1.0`), **2.103** vs $q$ (theory $O(q^2)$ since $\binom{m}{2}_2 \approx q^2/3$). |
| **Bottleneck** | `cProfile` (`GF(64), s=4`, 857k calls, 3.90 s): **56.4%** in `FreeModule.__iter__` $\to$ `linear_combination_of_basis` (`2.198 s`) + **42.2%** in `find_subspaces` (`1.643 s`). |
| **Correctness risks** | Same combinatorial blowup/OOM as `find_subspaces` for $m \ge 7$; `runner.py:988` enumerates *all* subgroups just to index `[add_idx]`, which hangs on `GF(256), s=16` (`200,787` subspaces). |
| **Better algorithm** | **Bitmask RREF generator + incremental XOR span** (`bench_prototypes.py` verified identical output): `GF(16), s=4`: `0.064 ms` (**938×**); `GF(64), s=4`: `3.71 ms` (**365×–556×**); `GF(64), s=8`: `7.27 ms` (**827×**); `GF(256), s=4`: `58.8 ms` (**~540×**). Plus lazy `islice` / combinatorial unranking for single-index lookup! |
| **Sage-replacement difficulty** | **1 / 5** — Pure Python/NumPy bitmask RREF (for $p=2$) or base-$p$ RREF over `0..q-1`. |
| **Priority** | **P1** (massive 500×+ win; blocks `--add-idx` sweeps for $q \ge 256$). |

#### 7. `subgroups.find_additive_subgroups_bruteforce(field: Any, size: int, max_subgroups: int = 1000) -> list[list[Any]]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.find_additive_subgroups_bruteforce`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L177-L219) — Enumerate up to `max_subgroups` additive subgroups via `(size-1)`-combinations. |
| **Algorithm** | Lazy `itertools.combinations(non_zero_elements, size - 1)`, prepend `0`, test `is_closed_under_addition`, break at `max_subgroups`. |
| **Complexity** | Time $O(\binom{q-1}{s-1} \cdot s^2)$ worst-case (average check exits in $O(s)$ on first failed sum); Space $O(\min(\binom{m}{d}_p, \text{max\_subgroups}) \cdot s)$. |
| **Sage primitives** | `sort_field_elements`, `is_closed_under_addition` (Givaro `+` & hashing). |
| **Measured** | `q=16, s=2`: `0.054 ms` (`0.051–0.062`, 2.2 KB); `q=256, s=2`: `0.844 ms` (**783× faster than `find_additive_subgroups`!**); `q=16, s=4`: `5.03 ms` (`1.03–7.08`, 5.2 KB); `q=32, s=4`: `32.5 ms`; `q=64, s=4`: **337.3 ms** (`299–362`, 119 KB, still **7.3× faster** than Sage `find_additive_subgroups`!); `q=16, s=8`: **65.7 ms** (`63.6–66.2`, 3.6 KB, crossover = `0.95×` of `find_additive_subgroups`). |
| **Bottleneck** | Exponential $\binom{q-1}{s-1}$ search space (`39,711` combos at $q=64, s=4$; $5.5 \times 10^8$ at $q=64, s=8$). |
| **Correctness risks** | Silent truncation at `max_subgroups=1000` with no warning; intractable for $s \ge 4, q \ge 128$ or $s \ge 8, q \ge 32$. |
| **Better algorithm** | Only useful as a test oracle against `find_additive_subgroups`; replace production uses with fast RREF enumerator (which beats bruteforce by **91×** even at $q=64, s=4$: `3.71 ms` vs `337 ms`). |
| **Sage-replacement difficulty** | **1 / 5** — Pure Python/NumPy XOR closure check on `combinations(range(1, q), s - 1)`. |
| **Priority** | **P2** (test/reference oracle only; not on production hot path). |

#### 8. `subgroups.list_multiplicative_subgroups(field: Any) -> dict[int, list[Any]]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.list_multiplicative_subgroups`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L222-L251) — Build all multiplicative subgroups of $\mathbb{F}_q^*$ keyed by order $d \mid (q-1)$. |
| **Algorithm** | Enumerates `divisors(q - 1)`, computes `step = gen ** ((q - 1) // d)`, multiplies `curr *= step` $d-1$ times, and sorts each subgroup via `sort_field_elements`. |
| **Complexity** | Time $O(\sum_{d \mid (q-1)} d \log d) = O(\sigma_1(q-1) \log q)$ where $\sigma_1(q-1) = O(q \log\log q)$; Space $O(\sigma_1(q-1))$. |
| **Sage primitives** | `divisors` (`sage/arith/misc.py:1490` $\to$ `Integer.divisors`, PARI/FLINT), `field.multiplicative_generator()` (`finite_field_base.pyx:689`), Givaro `*` / `**`, `sort_field_elements`. |
| **Measured** | `q=16` (4 divs, $\sigma_1=24$): `0.0127 ms` (`0.0121–0.0143`, 0.7 KB); `q=64` (6 divs, $\sigma_1=104$): `0.0386 ms`; `q=256` (8 divs, $\sigma_1=432$): `0.1535 ms` (`0.144–0.190`, 8.0 KB); `q=1024` (8 divs, $\sigma_1=1536$): `0.8138 ms` (`0.734–6.86`, 169 KB); `q=4096` (24 divs, $\sigma_1=8736$): **17.65 ms** (`13.3–23.4`, 899 KB). Empirical slope vs $q$: **1.264**. |
| **Bottleneck** | `cProfile` (`q=4096`): **73.7%** (`0.137 / 0.186 s`) in `sort_field_elements` $\to$ `field_element_to_int` (87,365 `hasattr` calls); `divisors(4095)` is `< 0.3%`! |
| **Correctness risks** | `TamoBargCodeTwoSets.__init__` (`tamo_barg_two_sets.py:178`) calls `list_multiplicative_subgroups` unconditionally even when both partitions are `"add"`, wasting all work! |
| **Better algorithm** | (1) Add `get_multiplicative_subgroup(field, d)` returning a single order-$d$ subgroup directly in $O(d)$ — measured **23.5× faster** at $q=4096, d=1365$ (`0.752 ms` vs `17.65 ms`); (2) For all subgroups, precompute `antilog` table once in $O(q)$ (`powers[0:q-1:(q-1)//d]` gives $O(\#\text{divisors})$ zero-copy views in NumPy, or $O(\sigma_1)$ with `np.sort`). |
| **Sage-replacement difficulty** | **1 / 5** — `antilog[::(q-1)//d]` in NumPy/`galois` + 10-line Python divisor loop. |
| **Priority** | **P2** (one-time init cost; `< 1 ms` for $q \le 1024$, `17.6 ms` at $q=4096$). |

#### 9. `subgroups.list_canonical_additive_subgroups(field: Any) -> dict[int, list[Any]]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.list_canonical_additive_subgroups`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L254-L282) — Build nested canonical subspaces $\text{span}(1, \alpha, \dots, \alpha^{i-1})$ for $i = 0 \dots m$. |
| **Algorithm** | Iterative doubling/expansion `current = [h + c * alpha**i for c in range(p) for h in current]`, sorting at every level via `sort_field_elements`. |
| **Complexity** | Time $O(\sum_{i=0}^m p^i \log(p^i)) = O(q \log q)$ (due to redundant sorting; construction itself is $O(q)$); Space $O(q)$. |
| **Sage primitives** | `field.characteristic()`, `field.degree()`, `field.gen()`, Givaro `+` / `*`, `sort_field_elements`. |
| **Measured** | `q=16`: `0.0192 ms` (`0.0189–0.0199`, 0.9 KB); `q=64`: `0.0661 ms` (`0.0656–0.0763`, 2.1 KB); `q=256`: `0.2785 ms` (`0.250–2.30`, 8.7 KB); `q=1024`: `5.299 ms` (`1.34–7.71`, 201 KB); `q=4096`: **21.47 ms** (`13.5–33.3`, 825 KB). Empirical slope vs $q$: **1.329** (theory $O(q \log q)$). |
| **Bottleneck** | Calling `sort_field_elements(current)` on $2q-1$ elements total (`~80%` of time). |
| **Correctness risks** | Verified on `GF(16)`, `GF(81)`, `GF(256)`, `GF(1024)`: `current` is **already** in exact `0 .. p^i - 1` `field_element_to_int` order without sorting! Also `TamoBargCodeTwoSets.__init__` builds ALL $m+1$ levels (including the full field of size $q$) even when only one size $p^t \ll q$ is needed! |
| **Better algorithm** | Drop redundant `sort_field_elements`, and build only up to target dimension $t$ ($O(p^t)$ instead of $O(q)$); in integer encoding, canonical subgroup of size $p^t$ is literally `np.arange(p**t, dtype=np.uint16)` in $O(1)$ / `< 1 µs`! |
| **Sage-replacement difficulty** | **1 / 5** — `np.arange(p**i)` in polynomial-basis integer encoding. |
| **Priority** | **P2** (init-only; easy quick win). |

#### 10. `subgroups.shift_additive_subgroup(field: Any, subgroup: Sequence[Any], shift_exponent: int) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.shift_additive_subgroup`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L285-L306) — Scale an additive subgroup $H$ by $g^{\text{shift\_exponent}}$. |
| **Algorithm** | Fetch primitive element $g$, compute $m = g^e$, multiply every $h \in H$ by $m$, sort result. |
| **Complexity** | Time $O(\log|e| + s \log s)$ for $|H| = s$; Space $O(s)$. |
| **Sage primitives** | `field.primitive_element()` / `multiplicative_generator()` (identical alias in `finite_field_base.pyx:689`), Givaro `**` / `*`, `field(h)` coercion, `sort_field_elements`. |
| **Measured** | ($s = \sqrt{q}$, `shift_exponent=3`): `q=16, s=4`: `0.0134 ms` (`0.0127–0.0146`, 2.0 KB); `q=64, s=8`: `0.0144 ms`; `q=256, s=16`: `0.0182 ms`; `q=1024, s=32`: `0.0255 ms`; `q=4096, s=64`: **0.0387 ms** (`0.0378–0.0416`, 7.0 KB). Empirical slope vs $s$: **0.390** (constant ~12 µs `primitive_element()` + exponentiation overhead dominates at $s \le 64$). |
| **Bottleneck** | Fixed ~12 µs overhead of `primitive_element()` lookup + `field(h)` coercion + `sort_field_elements`. |
| **Correctness risks** | Does not validate `subgroup` is additive; if `shift_exponent % (q - 1) == 0`, returns unshifted subgroup (caller must ensure disjointness). |
| **Better algorithm** | Vectorized `np.sort(gf_mul(sg_arr, g_pow))` in NumPy/`galois` (`< 2 µs`); remove redundant `field(h)` coercion when elements already belong to `field`. |
| **Sage-replacement difficulty** | **1 / 5** — `np.sort(sg * (g ** e))` in `galois` / NumPy LUT. |
| **Priority** | **P2** (called 1–2× at init; `< 0.04 ms`). |

#### 11. `subgroups.list_subgroup_cosets(field: Any, subgroup: Sequence[Any], subgroup_type: SubgroupType) -> list[list[Any]]`
| Field | Analysis |
|---|---|
| **Function** | [`subgroups.list_subgroup_cosets`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/subgroups.py#L309-L352) — Partition $\mathbb{F}_q$ (or $\mathbb{F}_q^*$) into deterministic sorted cosets of $H$. |
| **Algorithm** | Sort all $q$ field elements; scan in order maintaining a `visited: set`; for each unvisited representative $g$, form $g + H$ (or $g \cdot H$), sort it, add to `visited`. |
| **Complexity** | Time $O(q \log q)$ (dominated by sorting all $q$ elements + sorting $q/s$ cosets of size $s$ + $O(q)$ field ops and set lookups); Space $O(q)$. |
| **Sage primitives** | `iter(field)` (`finite_field_givaro.py:509`), `sort_field_elements` ($1 + q/s$ calls!), Givaro `+` / `*`, Python `set` hashing of `FiniteField_givaroElement`. |
| **Measured** | Additive ($s = \sqrt{q}$): `q=16`: `0.0176 ms` (`0.0169–0.0185`, 1.7 KB); `q=64`: `0.0570 ms`; `q=256`: `0.2110 ms` (`0.202–0.235`, 13.2 KB); `q=1024`: `5.196 ms` (`1.17–8.13`, 194 KB); `q=4096`: **22.94 ms** (`14.9–26.6`, 772 KB). Empirical slope vs $q$: **1.361**. Multiplicative (largest proper divisor $d$): `q=256, d=85`: `0.263 ms`; `q=4096, d=1365`: **27.61 ms** (`20.2–38.2`, 874 KB). |
| **Bottleneck** | `cProfile` (`q=4096`, 5 calls): **89.1%** (`0.172 / 0.193 s`) in `sort_field_elements` $\to$ `field_element_to_int` (40,960 calls, 81,915 `hasattr` calls). Actual coset arithmetic is only 4.7%! |
| **Correctness risks** | Does **not** verify `subgroup` is closed under `+`/`*`: a non-subgroup produces overlapping sets silently dropped by `visited` (breaking partition invariant). Also computes *all* $q/|H|$ cosets even when `TamoBargCodeTwoSets` slices `[:p_size]`! |
| **Better algorithm** | Integer-indexed boolean `visited` array (`bytearray(q)`) + integer-encoded `gf_add` (XOR for $p=2$!) / `gf_mul` LUT + `np.sort`; stop once `p_size` cosets are built. For canonical additive subgroups of $\mathbb{F}_{2^m}$, cosets are trivially `range(k*s, (k+1)*s)` in $O(q)$ with zero sorting! |
| **Sage-replacement difficulty** | **1 / 5** — `np.bitwise_xor` (additive $p=2$) or `antilog` stride (multiplicative) + boolean mask. |
| **Priority** | **P2** (init-only; `0.21 ms` at $q=256$, `22.9 ms` at $q=4096$). |

---

### 5.2 `tamo_barg_lrc.polynomials` (7 Functions)

#### 12. `polynomials.build_annihilator_polynomial(poly_ring: Any, evaluation_points: Sequence[Any]) -> Any`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.build_annihilator_polynomial`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L22-L34) — Construct monic vanishing polynomial $\prod_{\alpha \in A}(x - \alpha)$. |
| **Algorithm** | Balanced binary product tree via `sage.misc.misc_c.prod([x - base_ring(pt) for pt in evaluation_points], one)`. |
| **Complexity** | Time $O(M(n) \log n)$ where $M(n)$ is degree-$n$ polynomial multiplication ($O(n \log^2 n)$ with FFT/Karatsuba in NTL, $O(n^2)$ for small $n$); Space $O(n)$. |
| **Sage primitives** | `sage.misc.misc_c.prod` (`misc_c.pyx:72`, Cython balanced product tree), `Polynomial_ZZ_pEX._mul_` (`polynomial_template.pxi:327`, NTL `ZZ_pEX` multiplication), `base_ring(pt)` coercion. |
| **Measured** | ($n = q$, full field): `q=16`: `0.378 ms` (`0.351–4.55`, 3.4 KB); `q=64`: `10.59 ms` (`6.97–13.1`, 8.1 KB); `q=256`: `56.90 ms` (`42.9–61.7`, 26.3 KB); `q=1024`: `294.9 ms` (`257–326`, 99 KB); `q=4096`: **1,422.6 ms** (`1,311–1,442`, 387 KB). Empirical slope vs $q$: **1.428** (reflects NTL subquadratic Karatsuba/FFT transition above $n \approx 64$). |
| **Bottleneck** | Dense NTL `ZZ_pEX` multiplications in top tree levels + $n$ linear factor constructions. |
| **Correctness risks** | Duplicate evaluation points silently yield higher-degree polynomial with repeated roots. Only called by single-set `TamoBargVectorEncoder.__init__` (`tamo_barg.py:257`) — not called in two-set codes! |
| **Better algorithm** | Fast-path exact algebraic identities: if $A = \mathbb{F}_q$, return $x^q - x$ in **$O(1)$** (`< 0.01 ms`, **>100,000× faster** at $q=4096$!); if $A = \mathbb{F}_q^*$, return $x^{q-1} - 1$ in $O(1)$; if $A$ is a union of $s_p$ cosets of $H$ with good polynomial $g$, return $\prod_{c \in \text{cosets}} (g(x) - g(c))$ in $O(s_p \log^2 s_p \cdot |H|)$ or via linearized poly for additive subspaces! |
| **Sage-replacement difficulty** | **2 / 5** — 6-line Python product tree over `galois.Poly` / NumPy FFT or FLINT `fq_nmod_poly_product_roots`. |
| **Priority** | **P1** (init-only for single-set `TamoBargCode`, but `1.42 s` at $q=4096$ is trivially avoidable). |

#### 13. `polynomials.build_good_polynomial(poly_ring, subgroup, subgroup_type, annihilator=None) -> Any`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.build_good_polynomial`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L37-L73) — Build Tamo–Barg good polynomial $g(x)$ constant on cosets of $H$. |
| **Algorithm** | Multiplicative: $x^{|H|}$. Additive: `prod([x - h for h in H])` via balanced product tree. Optional `% annihilator` if $\deg(g) \ge \deg(\text{ann})$. |
| **Complexity** | Mult: Time $O(1)$ (sparse shift), Space $O(|H|)$ (dense NTL rep). Add: Time $O(M(|H|) \log |H|)$ ($O(|H|^2)$ for $|H| \le 64$ in NTL classical regime), Space $O(|H|)$. |
| **Sage primitives** | `x ** len(subgroup)` (`Polynomial_ZZ_pEX.__pow__`), `misc_c.prod`, `Polynomial_ZZ_pEX.quo_rem` (`polynomial_template.pxi:503`). |
| **Measured** | Additive ($|H|=\sqrt{q}$): `q=16, |H|=4`: `0.053 ms` (`0.049–0.071`, 2.2 KB); `q=64, |H|=8`: `0.173 ms`; `q=256, |H|=16`: `0.773 ms` (`0.676–4.85`, 3.4 KB); `q=1024, |H|=32`: `5.928 ms` (`2.78–7.95`, 5.0 KB); `q=4096, |H|=64`: **16.52 ms** (`12.5–22.5`, 8.1 KB). Slope vs $|H|$: **2.165** (NTL classical $O(|H|^2)$ regime!). Multiplicative: **0.00092–0.00125 ms** across all $q$. |
| **Bottleneck** | Dense NTL product tree on $|H|$ linear factors even though an additive subspace polynomial has at most $t+1 = \log_p|H|+1$ nonzero coefficients! |
| **Correctness risks** | Does not verify `subgroup` is a subgroup; if `annihilator` has $\deg \le |H|$ (e.g. $H = \mathbb{F}_q$ and $\text{ann} = x^q - x$), `good % annihilator` silently returns `0`! |
| **Better algorithm** | **Linearized $p$-polynomial recurrence** $L_{i+1}(x) = L_i(x)^p - L_i(b_i)^{p-1} L_i(x)$ maintaining only the $t+1$ coefficients of $x^{p^k}$ in **$O(t^2) = O(\log_p^2 |H|)$** field ops! Verified identical output in `bench_polynomials.py`: `|H|=16`: `0.219 ms` (**3.5×**); `|H|=32`: `0.408 ms` (**14.5×**); `|H|=64`: **`0.836 ms` (19.8× faster)**, with slope `0.995` (dominated only by final dense `poly_ring(dense)` call!). Keeping $g(x)$ in sparse linearized form avoids even the $O(|H|)$ dense allocation. |
| **Sage-replacement difficulty** | **1 / 5** — 15-line pure Python/NumPy linearized recurrence over GF LUT (and $x^{|H|}$ for mult). |
| **Priority** | **P2** (init-only; `16.5 ms` at $q=4096$, trivial 20× win). |

#### 14. `polynomials.build_algebra_basis(good_poly: Any, partition_size: int, annihilator: Any | None = None) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.build_algebra_basis`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L76-L109) — Generate $[1, g, g^2, \dots, g^{s_p - 1}]$ (optionally modulo `annihilator`). |
| **Algorithm** | Sequential multiplication `curr = curr * good_poly` for $j = 1 \dots s_p - 1$. |
| **Complexity** | Let $d_g = \deg(g) = |H|$ and $n = s_p \cdot d_g$. Step $j$ multiplies deg $j d_g$ by deg $d_g$: classical $O(j d_g^2)$, summed over $j=1..s_p-1$ gives **$O(s_p^2 d_g^2) = O(n^2)$** (or $O(s_p \cdot M(n))$ with FFT, i.e. $O(s_p n \log n)$); Space $O(s_p \cdot n) = O(s_p^2 d_g)$. |
| **Sage primitives** | `Polynomial_ZZ_pEX._mul_` (NTL `ZZ_pEX`), optional `quo_rem` (`%`). |
| **Measured** | ($s_p = |H| = \sqrt{q}$, additive $g$, no annihilator): `q=16, s_p=4`: `0.028 ms` (`0.028–0.051`, 0.4 KB); `q=64, s_p=8`: `5.86 ms` (`1.82–7.85`, 0.7 KB); `q=256, s_p=16`: `11.54 ms` (`7.35–15.4`, 1.5 KB); `q=1024, s_p=32`: `127.8 ms` (`118–141`, 3.0 KB); `q=4096, s_p=64`: **1,282.2 ms** (`1,269–1,294`, 6.0 KB — C++ NTL heap untracked by `tracemalloc`!). Empirical slope vs $q$: **1.769** (matches NTL $O(s_p \cdot n \log n) = O(q^{1.5} \log q)$ / $O(q^2)$ transition). |
| **Bottleneck** | Dense NTL `ZZ_pEX` multiplication ignoring that `good_poly` has only **1 term** (mult) or **$t+1 \le 7$ terms** (add)! |
| **Correctness risks** | When $s_p \cdot |H| = n = \deg(\text{ann})$, $\deg(g^j) = j|H| < n$ always, so `% annihilator` never triggers; if `partition_size` exceeds $n / |H|$, reduction collapses degrees. Note: C++ NTL memory is invisible to Python `tracemalloc` (`6 KB` reported vs ~20 MB actual NTL arrays at $q=4096$). |
| **Better algorithm** | **Sparse multiplication**: for mult ($g = x^{|H|}$), `curr.shift(|H|)` takes $O(s_p \cdot n)$ total with zero muls; for additive ($g$ has $t+1$ nonzero terms), multiplying by sparse $g$ takes **$O((t+1) \cdot s_p \cdot n) = O(q^{1.5} \log q)$** simple field ops (and in $\mathbb{F}_{2^m}$, $(g(x))^{2^k}$ is just squaring coefficients at $2^k$-shifted positions, so building powers via binary decomposition is even sparser!). Best of all: in evaluation domain, $g(\alpha)^j$ is constant on each coset and needs **zero** polynomial algebra! |
| **Sage-replacement difficulty** | **1 / 5** — Sparse NumPy/Python array shift-and-add (10 lines) or `galois.Poly`. |
| **Priority** | **P1** (init-only; `128 ms` at $q=1024$, `1.28 s` at $q=4096$, called 2× in two-set encoder). |

#### 15. `polynomials.build_encoding_basis(poly_ring, algebra_basis, locality, annihilator=None) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.build_encoding_basis`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L112-L155) — Construct sorted Tamo–Barg basis $\{x^i g(x)^j : 0 \le i < r, 0 \le j < s_p\}$. |
| **Algorithm** | Outer loop `shift_deg in range(r)`, inner loop `p in algebra_basis`: `p.shift(shift_deg)`, optional `% annihilator`, sort by `degree()` (tie-breaking on reversed coeff ints if degrees collide). |
| **Complexity** | Basis size $K = r \cdot s_p \le n$. Each `shift` copies a degree-$O(n)$ NTL polynomial $\implies$ Time **$O(K \cdot n) = O(r \cdot s_p \cdot n)$** + $O(K \log K)$ sort; Space **$O(K \cdot n)$** in NTL C++ heap. |
| **Sage primitives** | `Polynomial_ZZ_pEX.shift()` (`polynomial_zz_pex.pyx:443`), `degree()`, optional `quo_rem`, `list()`, `field_element_to_int`. |
| **Measured** | ($r = \sqrt{q}-2, s_p = \sqrt{q}, K \approx q$): `q=16, K=8`: `0.0074 ms` (`0.0071–0.0083`, 0.9 KB); `q=64, K=48`: `0.0999 ms` (`0.093–0.108`, 4.4 KB); `q=256, K=224`: `6.21 ms` (`6.08–8.14`, 27.5 KB); `q=1024, K=960`: `172.9 ms` (`149–192`, 143 KB); `q=4096, K=3968`: **2,446.4 ms** (`2,428–2,454`, 611 KB Python + ~128 MB NTL C++ heap). Empirical slope vs $q$: **2.371** (theory $O(q^2)$ NTL shifts + cache/malloc pressure at 128 MB). |
| **Bottleneck** | Copying $K = O(n)$ dense degree-$O(n)$ `Polynomial_ZZ_pEX` objects (each `shift` allocates and copies a full NTL `ZZ_pEX` array). |
| **Correctness risks** | When $r \le |H| = \deg(g)$ and $n = s_p |H|$, all $r \cdot s_p$ degrees $i + j|H| < n$ are **strictly distinct** $\implies$ guaranteed linearly independent and $\dim(\text{basis}[:k]) = k$ (verified in benchmark). However, if $r > |H|$ or modular reduction collapses degrees, duplicates/linear dependencies are **not** detected! Also outer-`i`/inner-`j` loop order scrambles degrees needlessly; swapping to outer-`j`/inner-`i` yields already-sorted output whenever $r \le |H|$. |
| **Better algorithm** | Never materialize $r \cdot s_p$ shifted NTL polynomials! Either (a) construct the $K \times n$ coefficient matrix directly in M4RIE / NumPy by shifting the $s_p$ rows of `algebra_basis` in place ($O(K \cdot n)$ contiguous memcpy, `< 5 ms` at $q=4096$), or (b) work directly in evaluation domain (no monomial basis at all). |
| **Sage-replacement difficulty** | **1 / 5** — NumPy 2D coefficient matrix slicing `mat[j*r + i, i : i+deg+1] = alg_row`. |
| **Priority** | **P1** (`173 ms` at $q=1024$, `2.45 s` at $q=4096$, called 2× in two-set encoder). |

#### 16. `polynomials.intersect_encoding_bases(poly_ring, basis_a, basis_b, length) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.intersect_encoding_bases`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L158-L231) — Compute degree-minimizing basis of $\text{span}(A) \cap \text{span}(B)$ in $\mathbb{F}_q[x]_{<n}$. |
| **Algorithm** | **Stacked-matrix left kernel + reverse-column RREF** (Note: docstring says "Zassenhaus", which normally RREFs a $(k_a+k_b) \times 2n$ block matrix; the actual code computes `K = left_kernel([A; -B])` on $(k_a+k_b) \times n$, projects `K[:, :k_a] * A`, reverses columns, computes RREF so pivots clear highest degrees first, and reconstructs `poly_ring(coeffs)`). |
| **Complexity** | Let $k_a = |A|, k_b = |B|, N \ge n, d_\cap = \dim(A \cap B)$. Serialization: $O((k_a + k_b + d_\cap) N)$ NTL $\leftrightarrow$ Givaro conversions. Linear algebra: `left_kernel_matrix` on $(k_a+k_b) \times N$ takes $O((k_a+k_b) N \min(k_a+k_b, N)) = O(k^2 n)$ (up to $O(n^3)$ when $k = \Theta(n)$); `ker*A` takes $O(d_\cap k_a N)$; reverse RREF takes $O(d_\cap^2 N)$. Space $O((k_a+k_b) N)$. |
| **Sage primitives** | `Polynomial_ZZ_pEX.padded_list` (`polynomial_element.pyx:6727`), `matrix()` (`matrix_space.py:210-300` $\to$ `Matrix_gf2e_dense` via M4RIE for $\mathbb{F}_{2^m}, m \le 16$; **`Matrix_generic_dense` fallback for odd-char extensions like $\mathbb{F}_{81}$ & $m \ge 17$**), `stack`, `left_kernel_matrix` (`matrix2.pyx:4966` $\to$ transpose + `right_kernel_matrix`), `matrix_from_columns`, M4RIE `*` & `echelon_form` (`matrix_gf2e_dense.pyx:928`), `poly_ring(coeffs)` (`polynomial_ring.py:336`). |
| **Measured** | `q=16` ($k_a=k_b=8, d_\cap=4$): **0.864 ms** (`0.790–5.14`, 4.0 KB); `q=64` ($k=48, d_\cap=36$): **48.7 ms** (`47.2–52.2`, 33.3 KB); `q=256` ($k=224, d_\cap=196$): **1,262 ms** (`1,133–1,283`, 716 KB); `q=1024` ($k=960, d_\cap=900$): **33,078 ms** (single-shot, 78.2 MB; phase-breakdown run `23,661 ms`). Empirical slope vs $q$: **2.518** (blend of $O(q^2)$ NTL serialization + $O(q^3)$ / Strassen M4RIE kernel). Odd-char `GF(81)` (`Matrix_generic_dense`, $k=63, n=81$): **63.6 ms** vs `GF(64)` (`40.0 ms`, **1.6×** at small $n$, worsening cubically at larger $n$). |
| **Bottleneck** | **Phase breakdown** (`q=256`, `1,021 ms`): `poly_build` **66.5%** (`679 ms`) + `padded_list` **23.0%** (`235 ms`) + `matrix()` **4.1%** (`42 ms`) = **93.6% pure serialization**, while `left_ker` (5.2%) + `ker*A` (0.4%) + `rev_rref` (0.7%) = only **6.3%** (`64.5 ms`)! At `q=1024` (`23,661 ms`): `poly_build` **48.4%** (`11.46 s`), `left_ker` **28.2%** (`6.68 s`), `padded_list` **18.7%** (`4.42 s`), `matrix()` **2.4%**, `rev_rref` **1.7%**, `ker*A` **0.6%** — still **69.5% serialization**! |
| **Correctness risks** | (1) `Matrix_generic_dense` fallback on $\mathbb{F}_{p^m}$ ($p>2, m>1$) and $\mathbb{F}_{2^m}$ ($m \ge 17$) hits an unguarded `AttributeError: module 'sage.rings.polynomial' has no attribute 'laurent_polynomial_ring'` in modular `passagemath` (`matrix_space.py:290`) if `sage.coding` wasn't imported first. (2) Called **twice** during `TamoBargCodeTwoSets(...) + code.encoder("VectorEncoder")` (`max_dimension()` doesn't cache the encoder!). |
| **Better algorithm** | (1) **Keep bases as matrices** (eliminate `padded_list` + `poly_build` = immediate **3×–15×** gain, AND turns `generator_matrix()` into a single `CoeffMat * Vandermonde` M4RIE matmul!); (2) **Dual orthogonal method** $(U^\perp + V^\perp)^\perp$ (when $k_a + k_b > n$, typical for high-rate LRCs): measured **1.57× faster** at $q=1024$ (`21.0 s` vs `33.1 s`); (3) **Direct evaluation-domain local dual-GRS PCM kernel** (when full intersection dimension $k = d_\cap$ is used): measured **26.7 ms at $q=256$** (**47× faster** than `intersect_encoding_bases`, **612× faster** than full two-set init + `generator_matrix`) and **907 ms at $q=1024$** (**36× faster**). |
| **Sage-replacement difficulty** | **3 / 5** — Fast dense $\mathbb{F}_q$ matrix RREF/nullspace (`galois` Numba RREF or FLINT `fq_nmod_mat_nullspace`, ~30 lines once matrix backend exists). |
| **Priority** | **P0** (#1 construction bottleneck; dominates two-set code initialization). |

#### 17. `polynomials.evaluate_message_polynomial(poly_ring, basis, message) -> Any`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.evaluate_message_polynomial`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L234-L256) — Form linear combination $\sum_{i=0}^{k-1} m_i b_i(x)$ in $\mathbb{F}_q[x]$. |
| **Algorithm** | Python `sum((base_ring(message[i]) * basis[i] for i in range(k)), poly_ring.zero())` (sequential left-to-right polynomial accumulation). |
| **Complexity** | Time $O(k \cdot n)$ field ops + $O(k)$ NTL `Polynomial_ZZ_pEX` allocations; Space $O(n)$ (with $k$ transient degree-$n$ allocations). |
| **Sage primitives** | `base_ring(message[i])` coercion, `Polynomial_ZZ_pEX._lmul_` (Givaro $\to$ NTL scalar conversion + array scale), `Polynomial_ZZ_pEX._add_`. |
| **Measured** | ($k = d_\cap$): `q=16, k=4`: `0.232 ms` (`0.076–0.319`, 2.4 KB); `q=64, k=36`: `5.24 ms` (`1.18–5.29`, 2.5 KB); `q=256, k=196`: `42.9 ms` (`28.2–49.2`, 2.6 KB); `q=1024, k=900`: **480.8 ms** (`428–516`, 2.9 KB). Empirical slope vs $q$: **1.804** (theory $O(k \cdot n) = O(q^2)$). |
| **Bottleneck** | $k$ separate Python-loop NTL polynomial scalar multiplications + additions + Givaro-to-NTL scalar coercions (allocating $2k$ temporary `Polynomial_ZZ_pEX` objects). |
| **Correctness risks** | None functional; performance degrades quadratically when called per-message instead of caching a $k \times n$ basis coefficient matrix. |
| **Better algorithm** | Store the encoding basis as a $k \times n$ matrix $B$ over $\mathbb{F}_q$ (M4RIE / NumPy / `galois`) and compute `coeffs = vector(message) * B` in a single BLAS/M4RIE vector-matrix multiply (`< 0.5 ms` at $q=1024$, **~1,000× faster**), constructing `poly_ring(coeffs)` only if a polynomial object is explicitly requested. |
| **Sage-replacement difficulty** | **1 / 5** — `msg @ basis_mat` in `galois` / NumPy LUT. |
| **Priority** | **P2** (encoding normally uses `generator_matrix`; `enc_poly` is used mainly in inspection/tests). |

#### 18. `polynomials.interpolate_erasures(poly_ring, eval_points, values, erasure_mask) -> list[Any]`
| Field | Analysis |
|---|---|
| **Function** | [`polynomials.interpolate_erasures`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L259-L298) — Recover erased symbols via univariate Lagrange interpolation. |
| **Algorithm** | Partition indices into $s$ known points and $e$ erased indices; call `poly_ring.lagrange_polynomial(known_pts)` (divided differences + Horner monomial expansion); evaluate `interp_poly(eval_points[idx])` for each erased index. |
| **Complexity** | Time $O(s^2)$ field ops (`divided_difference`) + $O(s^2)$ NTL polynomial ring ops (Horner monomial expansion) + $O(e \cdot s)$ NTL evaluations; Space $O(s^2)$ (full triangular divided-difference table `F` in Sage `polynomial_ring.py:2420`!). |
| **Sage primitives** | `PolynomialRing_field.lagrange_polynomial` (`polynomial_ring.py:2361-2538`) $\to$ `divided_difference` (pure Python $O(s^2)$) + $s$ NTL `Polynomial_ZZ_pEX` multiplications/additions + `Polynomial_ZZ_pEX.__call__` (`polynomial_zz_pex.pyx:225`). |
| **Measured** | **Local coset ($e=2, s=|H|-2$)**: `q=16, s=2`: `0.0775 ms` (`0.073–0.104`, 2.2 KB); `q=64, s=6`: `0.352 ms`; `q=256, s=14`: **6.16 ms** (`1.12–9.53`, 4.0 KB); `q=1024, s=30`: **15.90 ms** (`12.9–18.8`, 39.7 KB); `q=4096, s=62`: **63.86 ms** (`59.2–67.8`, 162 KB). Empirical local slope vs $s$: **2.032** (theory $O(s^2)$). **Global (`GF(1024)`, $e=n/4, s=3n/4$)**: `n=32, s=24`: `13.3 ms`; `n=64, s=48`: `38.5 ms`; `n=128, s=96`: `137.7 ms`; `n=256, s=192`: `558.4 ms`; `n=512, s=384`: **2,185 ms** (slope **1.858**). |
| **Bottleneck** | Monomial expansion loop (`P *= (var - x_i); P += F[i]`) in `polynomial_ring.py:2535-2538` + allocating $O(s^2)$ full divided-difference table `F` + Givaro $\leftrightarrow$ NTL coercions. |
| **Correctness risks** | (1) **Confirmed bug**: when ALL points are erased (`known_pts == []`), `lagrange_polynomial([])` silently returns the zero polynomial, filling output with `[0, ..., 0]` instead of raising `ValueError`/`DecodingError`! (2) Duplicate `eval_points` raise cryptic `ZeroDivisionError`. (3) Accepts no degree bound $r$: in `TamoBargErasureDecoder` (`tamo_barg.py:403`), when $0 < e < d_{\text{local}}-1$ ($s > r$), it interpolates all $s$ points instead of $r$ points, wasting $(s/r)^2$ work and masking undetected errors. |
| **Better algorithm** | **Direct Newton divided-difference evaluation** (1D in-place $O(s)$ table + Horner in Newton basis, **zero polynomial ring calls**) or **barycentric Lagrange** (even faster if coset weights $w_j$ are precomputed once per coset $\to O(e \cdot s)$ per decode!). Verified identical outputs: **Local $s=14$**: `0.0147 ms` (**419× faster**); **Local $s=62$**: `0.516 ms` (**124× faster**); **Global $s=384$**: `95.5 ms` (**22.9× faster**). |
| **Sage-replacement difficulty** | **1 / 5** — 15-line pure Python/NumPy direct Newton or barycentric interpolation over GF LUT (or FLINT `fq_nmod_poly_interpolate` for $s > 512$ using $O(s \log^2 s)$ subproduct trees). |
| **Priority** | **P0** (on the per-codeword hot path of all erasure / error-erasure decoders; 23×–419× drop-in speedup!). |

---

## 6. Deep-Dive Answers & Scaling Tables

### 6.1 Subspace Enumeration (`find_subspaces` / `find_additive_subgroups` vs Bruteforce)

`VectorSpace(GF(p), m).subspaces(d)` calls `reduced_echelon_matrix_iterator` (`sage/matrix/echelon_matrix.pyx:17`), which enumerates every canonical $d \times m$ RREF matrix over $\mathbb{F}_p$ in lexicographic pivot-column order — exactly $\binom{m}{d}_p = \prod_{i=0}^{d-1}\frac{p^{m-i}-1}{p^{d-i}-1}$ subspaces.

#### Table 6.1a — Scaling of `find_subspaces` and `find_additive_subgroups` over $\mathbb{F}_{2^m}$ ($m = 4 \dots 12$)

| $m$ | $q = 2^m$ | Fixed $d=2$ ($s=4$): $\binom{m}{2}_2$ | `find_subspaces` (ms) | `find_additive_subgroups` (ms) | Peak Mem (KB) | Mid-dim $d=\lfloor m/2 \rfloor$: $\binom{m}{d}_2$ | Mid-dim `find_additive_subgroups` (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 35 | 42.3 (38.7–48.9) | 95.9 (90.7–105) | 214 | 35 ($d=2$) | 98.1 (92.5–118) |
| 5 | 32 | 155 | 196.4 (182–224) | 526.4 (266–675) | 799 | 155 ($d=2$) | 504.5 (446–662) |
| 6 | 64 | 651 | 794.9 (790–843) | 2,063 (1,654–2,314) | 3,120 | 1,395 ($d=3$) | **6,016** (5,718–6,294; 6.87 MB) |
| 7 | 128 | 2,667 | 3,025 (2,472–3,066) | **7,848** (6,488–8,911) | 12,723 | 11,811 ($d=3$) | *Intractable (~51 s)* |
| 8 | 256 | 10,795 | *~12.2 s (extrap.)* | *~31.8 s (extrap.)* | *~51 MB* | 200,787 ($d=4$) | *Intractable (~25 min)* |
| 9 | 512 | 43,435 | *~49.3 s (extrap.)* | *~128 s (extrap.)* | *~207 MB* | 3,309,747 ($d=4$) | *Intractable (~7 hr)* |
| 10 | 1024 | 174,251 | *~198 s (extrap.)* | *~513 s (extrap.)* | *~830 MB* | 109,221,651 ($d=5$) | *Intractable (~2 wk)* |
| 11 | 2048 | 698,027 | *~792 s (extrap.)* | *~2,054 s (extrap.)* | *~3.3 GB* | $3.55 \times 10^9$ ($d=5$) | *Intractable* |
| 12 | 4096 | 2,794,155 | *~3,169 s (extrap.)* | *~8,222 s (extrap.)* | *~13.3 GB* | $2.31 \times 10^{11}$ ($d=6$) | *Impossible* |

- **Empirical slopes ($d=2$)**: `0.984` (`find_subspaces`) and `1.011` (`find_additive_subgroups`) vs $\binom{m}{2}_2$; **`2.103` vs $q$** (matching $\binom{m}{2}_2 \sim q^2/3$).
- **Blowup threshold**: Exceeds 1 second at **$m=6$ ($q=64$)** for both $d=2$ (`2.06 s`) and $d=3$ (`6.02 s`), and becomes practically unusable for mid-dimension $d = \lfloor m/2 \rfloor$ starting at **$m=7$ ($q=128$)** (and mathematically impossible to enumerate in full for $m \ge 10$ regardless of implementation, since $\binom{10}{5}_2 = 1.09 \times 10^8$).
- **Fast bitmask RREF prototype (`bench_prototypes.py`, verified exact equality)**: Reduces `GF(16), d=2` from `59.7 ms` $\to$ **`0.064 ms` (938×)**, `GF(64), d=2` from `1,355 ms` $\to$ **`3.71 ms` (365×)**, `GF(64), d=3` from `6,016 ms` $\to$ **`7.27 ms` (827×)**, and `GF(256), d=2` from ~`31.8 s` $\to$ **`58.8 ms` (~540×)**.

#### Table 6.1b — `find_additive_subgroups` vs `find_additive_subgroups_bruteforce` Crossover

| $(q, s)$ | $\binom{m}{d}_2$ | $\binom{q-1}{s-1}$ | `find_additive_subgroups` (ms) | `bruteforce` (ms) | Ratio `bf / alg` | Winner |
|---|---:|---:|---:|---:|---:|---|
| (16, 2) | 15 | 15 | 37.19 | 0.054 | **0.0015×** | Bruteforce (689× faster) |
| (64, 2) | 63 | 63 | 164.52 | 0.201 | **0.0012×** | Bruteforce (819× faster) |
| (256, 2) | 255 | 255 | 660.90 | 0.844 | **0.0013×** | Bruteforce (783× faster) |
| (8, 4) | 7 | 35 | 18.76 | 0.101 | **0.0054×** | Bruteforce (186× faster) |
| (16, 4) | 35 | 455 | 102.13 | 5.032 | **0.049×** | Bruteforce (20× faster) |
| (32, 4) | 155 | 4,495 | 491.07 | 32.48 | **0.066×** | Bruteforce (15× faster) |
| (64, 4) | 651 | 39,711 | 2,451.17 | 337.26 | **0.138×** | Bruteforce (7.3× faster!) |
| **(16, 8)** | **15** | **6,435** | **69.34** | **65.68** | **0.947×** | **Crossover (~1×)** |
| (32, 8) | 155 | $2.63 \times 10^6$ | ~600 ms | *~22 s (intractable)* | *~37×* | Algebraic (`find_additive_subgroups`) |

**Surprising crossover insight**: Because Sage's `find_additive_subgroups` pays ~**3.8 ms per subspace** in `FreeModule` overhead whereas `is_closed_under_addition` rejects non-subgroups in ~8.5 µs on the first sum, `bruteforce` paradoxically **beats** the algebraic enumerator for all $s=2$ (any $q$), for $s=4$ up to $q \approx 512\text{--}900$ ($\approx 1.4\,\mu\text{s}\cdot q^3$ vs $1.27\,\text{ms}\cdot q^2$), and crosses over right at **$(q=16, s=8)$**! However, the bitmask RREF prototype (`3.71 ms` at $q=64, s=4$) beats *both* by orders of magnitude across all $(q, s)$.

---

### 6.2 `intersect_encoding_bases` Scaling, Phase Breakdown & Two-Set Construction Profile

#### Table 6.2a — `intersect_encoding_bases` Scaling & Algorithm Comparison ($q \in \{16, 64, 256, 1024\}$)

| $q$ ($n$) | $k_a = k_b$ | $d_\cap$ | Current Stacked-Ker (ms) | Peak Mem | Classical Zassenhaus (ms) | Dual $(U^\perp+V^\perp)^\perp$ (ms) | Eval-Domain Local PCM $G$ (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8 | 4 | **0.864** (0.790–5.14) | 4.0 KB | 0.871 | 0.881 (0.98×) | **0.189** (4.6×) |
| 64 | 48 | 36 | **48.71** (47.2–52.2) | 33.3 KB | 74.48 | 50.04 (0.97×) | **2.586** (18.8×) |
| 256 | 224 | 196 | **1,262** (1,133–1,283) | 716 KB | 1,243 | 1,232 (1.02×) | **26.70** (47.3×) |
| 1024 | 960 | 900 | **33,078** (single-shot) | 78.2 MB | *skipped* | **21,039** (1.57×) | **907.4** (36.5×) |
| **Slope vs $q$** | — | — | **2.518** | — | — | **2.413** | **2.02** |

*(Note: `GF(4096)` with $k_a=k_b=3968, n=4096$ extrapolates to ~`1,080 s` ($\sim 18\text{ min}$) and >2.5 GB RAM in current Sage implementation, so skipped.)*

#### Table 6.2b — Internal Phase Breakdown of `intersect_encoding_bases`

| $q$ | Total (ms) | `padded_list` (NTL $\to$ Givaro) | `matrix()` (M4RIE load) | `left_kernel_matrix` (M4RIE) | `ker[:,:ka]*A` (M4RIE) | `rev_cols` RREF (M4RIE) | `poly_ring(coeffs)` (Givaro $\to$ NTL) | **Total Serialization %** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 35.47 | 8.79 ms (24.8%) | 3.33 ms (9.4%) | 3.59 ms (10.1%) | 0.20 ms (0.6%) | 0.10 ms (0.3%) | 19.45 ms (54.8%) | **89.0%** |
| 256 | 1,021.3 | 235.3 ms (23.0%) | 42.06 ms (4.1%) | 53.26 ms (5.2%) | 4.01 ms (0.4%) | 7.21 ms (0.7%) | **679.4 ms (66.5%)** | **93.6%** |
| 1024 | 23,661 | 4,417 ms (18.7%) | 563.2 ms (2.4%) | 6,683 ms (28.2%) | 138.2 ms (0.6%) | 396.2 ms (1.7%) | **11,463 ms (48.4%)** | **69.5%** |

#### Table 6.2c — Is `intersect_encoding_bases` the Dominant Term in Two-Set Construction?

| Config (`["add","add"]`, `shift_add=True`) | `code_init` (`TamoBargCodeTwoSets`) | `enc_init` (`code.encoder`) | `generator_matrix()` | Total (ms) | Dominant Component |
|---|---:|---:|---:|---:|---|
| `q=16, |H|=4, r=2, k=2` | 19.35 ms (75.6%) | 5.38 ms (21.0%) | 0.85 ms (3.3%) | 25.58 | `code_init` (field/subgroup setup + `intersect` #1) |
| `q=64, |H|=8, r=6, k=18` | 72.20 ms (23.5%) | 78.22 ms (25.5%) | 156.9 ms (51.1%) | 307.3 | `generator_matrix` (51%), `intersect`×2 (~47%) |
| `q=256, |H|=8, r=6, k=20` | 1,311 ms (37.0%) | 1,270 ms (35.8%) | 965.4 ms (27.2%) | 3,547 | **`intersect_encoding_bases` ×2 (~71%)** |
| `q=256, |H|=16, r=14, k=98` | 829.7 ms (5.1%) | 1,093 ms (6.7%) | **14,427 ms (88.2%)** | 16,350 | **`generator_matrix` (88.2%)**; `intersect`×2 = 97% of init |

- **Answer**: Inside `code_init + enc_init`, **`intersect_encoding_bases` accounts for 97.2% of runtime** (`3.058 / 3.147 s` in `cProfile`) and is **needlessly invoked twice** because `TamoBargCodeTwoSets.__init__` calls `self.max_dimension()` (creating a throwaway `TamoBargTwoSetsVectorEncoder`) before the user calls `code.encoder("VectorEncoder")`! Including `generator_matrix()`, `intersect_encoding_bases` dominates at low-to-moderate $k$ (`71%` at $q=256, k=20$), while at high $k$ (`k=98, q=256`), `generator_matrix()`'s $k \cdot n$ `Polynomial_ZZ_pEX.__call__` evaluations take over (`13.57 s`, 81–88%). Keeping the intersection basis as a matrix (rather than reconstructing `Polynomial_ZZ_pEX`) eliminates **both** bottlenecks at once (turning `16.35 s` into `< 100 ms`).

---

### 6.3 `interpolate_erasures` Comparison (Lagrange vs Newton vs Barycentric vs Vandermonde)

| Regime | $(n, s_{\text{known}}, e_{\text{erased}})$ | Current Sage `lagrange_polynomial` (ms) | Direct Newton (ms) | Barycentric (ms) | Vandermonde M4RIE (ms) | Best Speed-up |
|---|---|---:|---:|---:|---:|---:|
| Local `GF(16)` | $(4, 2, 2)$ | 0.0775 | **0.0014** | 0.0020 | 0.2157 | **56.2×** |
| Local `GF(64)` | $(8, 6, 2)$ | 0.3521 | **0.0039** | 0.0052 | 0.2332 | **91.4×** |
| Local `GF(256)` | $(16, 14, 2)$ | 6.1594 | **0.0147** | 0.0194 | 0.3200 | **418.7×** |
| Local `GF(1024)` | $(32, 30, 2)$ | 15.901 | **0.1219** | 0.1627 | 6.4109 | **130.5×** |
| Local `GF(4096)` | $(64, 62, 2)$ | 63.858 | **0.5161** | 0.5881 | 18.421 | **123.7×** |
| Global `GF(1024)` | $(32, 24, 8)$ | 13.306 | **0.113** | 0.152 | 0.839 | **117.5×** |
| Global `GF(1024)` | $(64, 48, 16)$ | 38.474 | **0.458** | 0.599 | 6.308 | **84.1×** |
| Global `GF(1024)` | $(128, 96, 32)$ | 137.71 | **5.867** | 6.365 | 18.99 | **23.5×** |
| Global `GF(1024)` | $(256, 192, 64)$ | 558.40 | **23.49** | 29.32 | 79.10 | **23.8×** |
| Global `GF(1024)` | $(512, 384, 128)$ | 2,185.0 | **95.53** | 115.6 | 857.5 | **22.9×** |

- **Takeaway**: Direct Newton divided-difference evaluation (1D in-place table, zero `Polynomial_ZZ_pEX` allocations) wins at **every** tested size ($s = 2 \dots 384$), delivering **56×–419×** speedup at typical local coset sizes ($s \le 62$) and **23×–118×** at global erasure sizes! Barycentric Lagrange is within 1.2× of Newton on one-shot calls and becomes **$O(e \cdot s)$** (eliminating the $O(s^2)$ setup entirely!) whenever coset weights $w_j$ are precomputed. Vandermonde solve is asymptotically worse ($O(s^3)$) and 5×–150× slower than Newton across all tested sizes.

---

### 6.4 Remaining Specific Questions Summary

1. **`list_multiplicative_subgroups` ($O(\#\text{divisors})$ lookup?)**: Precomputing `antilog = [gen^i for i in range(q-1)]` takes $O(q)$ field ops once, after which every order-$d$ subgroup is an **$O(1)$ zero-copy stride view** `antilog[0 : q-1 : (q-1)//d]` in NumPy (so enumerating all subgroups' views is $O(\#\text{divisors})$, though materializing/sorting all elements still costs $O(\sigma_1(q-1) \log q)$). In Sage, a table saved 17% (`14.7 ms` vs `17.6 ms` at $q=4096$) because 74% is `sort_field_elements`; fetching only the single needed divisor $d$ saves **23.5×** (`0.752 ms` vs `17.65 ms`).
2. **`field_element_to_int` / `sort_field_elements` vectorization**: Yes — polynomial-basis integer encoding (`0..q-1`, identical to `to_integer()`) makes `field_element_to_int` a zero-cost identity and turns `sort_field_elements` into `np.sort` (**74× faster at $q=256$, 1,008× faster at $q=4096$**).
3. **`build_good_polynomial` linearized recurrence vs product-of-linear-factors**: For an additive subgroup $H$ of size $s = p^t$, $g_H(x) = \prod_{h \in H}(x - h)$ is a linearized $p$-polynomial $\sum_{k=0}^t c_k x^{p^k}$ with **at most $t+1$ nonzero coefficients** (verified: `nnz = 3, 4, 5, 6, 7` for `|H| = 4, 8, 16, 32, 64`). Computing it via $L_{i+1}(x) = L_i(x)^p - L_i(b_i)^{p-1} L_i(x)$ takes $O(t^2) = O(\log_p^2 s)$ field ops vs $O(s \log^2 s)$ (empirically $s^{2.165}$ in NTL), yielding **3.5× at $|H|=16$, 14.5× at $|H|=32$, and 19.8× at $|H|=64$**.
4. **`build_algebra_basis` / `build_encoding_basis` degree bounds & dimension**: Count is $r \cdot s_p$ with $\deg(x^i g(x)^j) = i + j|H| < s_p |H| = n$ for $0 \le i < r \le |H|$ and $0 \le j < s_p$. Strictly distinct degrees guarantee linear independence and $\dim(\text{basis}[:k]) = k$ for any $1 \le k \le r \cdot s_p$ (verified across all $q \in \{16..4096\}$).
