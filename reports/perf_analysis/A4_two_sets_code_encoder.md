# A4 Performance & SageMath-Alternatives Analysis: `TamoBargCodeTwoSets`, Encoder, and Helpers

**Scope:** `src/tamo_barg_lrc/tamo_barg_two_sets.py` lines 52–509 — module helpers (`_is_int_like`, `_int_log_base`, `_safe_parity_check_matrix`), `TamoBargCodeTwoSets.*` (`__init__`, `__eq__`, `__hash__`, `_repr_`, 8 accessors, `design_distance`, `max_dimension`, `max_rate`, `minimum_distance_py`), and `TamoBargTwoSetsVectorEncoder.*` (`__init__`, `_repr_`, 5 accessors, `enc_poly`, `generator_matrix`, plus inherited `Encoder.encode`/`unencode`).

**Environment & Methodology:**
- Machine: 2-vCPU Intel Xeon Platinum 8481C (`nproc=2`), Linux Debian 12, Python 3.11.2 (`/home/antigravity/.venv-lrc/bin/python`), modular `passagemath` 10.8.x, NumPy.
- Reproducibility: fixed seeds (`random.seed(42)`, `np.random.seed(42)`, `sage.misc.randstate.set_random_seed(42)`), warm-up $\ge 3$ calls, $\ge 20$ repeats ($\ge 5$ for $>1\text{ s}$ calls), reporting `median` (`p10`, `p90`) via `time.perf_counter`, hotspots via `cProfile`/`pstats`, peak allocations via `tracemalloc`.
- Scratch benchmark scripts: `reports/perf_analysis/scratch/A4/` (`bench_init_and_phases.py`, `bench_helpers_encoder_mindist.py`, `bench_mindist_and_extra.py`, `bench_quick_followup.py`).

---

## 1. Module Design Narrative & Critique

`TamoBargCodeTwoSets` and `TamoBargTwoSetsVectorEncoder` implement two-partition Tamo–Barg locally recoverable codes (LRCs) over $\mathbb{F}_q$ supporting all four subgroup-type pairs (`add/add`, `mult/mult`, `add/mult`, `mult/add`). The algebraic construction is mathematically clean: for each partition $i \in \{0,1\}$, a good polynomial $g_i(x)$ constant on cosets of $H_i$ generates an algebra basis $\{g_i^j\}_{j=0}^{s_i-1}$ and an encoding basis $\mathcal{B}_i = \{x^a g_i(x)^j : 0 \le a < r_i\}$; the two-set encoding space $\mathcal{B}_0 \cap \mathcal{B}_1 \cap \mathbb{F}_q[x]_{<n}$ is computed via Zassenhaus left-kernel intersection followed by reverse-column RREF to extract a strictly degree-minimal basis, whose first $k$ polynomials define the code.

However, profiling exposes **four severe performance pathologies** caused almost entirely by SageMath backend mismatches rather than algorithmic necessity:

1. **Catastrophic NTL `Polynomial_ZZ_pEX` default over $\mathbb{F}_{2^m}$ (`P0`):** Over `GF(2^m)` ($m \le 15$), Sage chooses `FiniteField_givaro` (fast Zech-log C++ field) and `Matrix_gf2e_dense` (fast M4RIE C matrices), **but** `PolynomialRing(GF(2^m), 'x')` defaults to `Polynomial_ZZ_pEX` (`sage/rings/polynomial/polynomial_ring.py:2655-2662`) — NTL's generic arbitrary-precision `ZZ_pEX` rather than `GF2EX` or FLINT `fq_nmod_poly` (which is unavailable for Givaro). Every multiplication in `build_algebra_basis`, every coefficient read in `padded_list()`, every polynomial construction `R(coeffs)`, and every evaluation `poly(alpha)` in `generator_matrix()` marshals Givaro $\leftrightarrow \mathbb{F}_2[x] \leftrightarrow$ Python `list` $\leftrightarrow$ NTL `ZZ_pE` (`polynomial_zz_pex.pyx:285-287`). On `GF(256)`, this marshaling + NTL `ZZ_pEX` arithmetic consumes **96–98% of `__init__`** (`953 ms` on `global_sim`, `1,264 ms` on `c=4`) while the actual M4RIE linear algebra (`left_kernel_matrix` + `mul` + `rref`) takes only **24–28 ms (2.2%)**. Merely switching to `PolynomialRing(F, 'x', implementation='generic')` cuts `GF(256)` `__init__` by **15×** (`953 ms → 63.5 ms`) and `generator_matrix()` by **48×** (`241.7 ms → 5.04 ms`).
2. **Cubic blowup in `generator_matrix()` for moderate/high $k$ (`P0`):** Evaluating $k$ degree-$O(n)$ polynomials at $n$ points via `poly(alpha)` costs $O(k n \cdot \text{deg})$ with ~100–650 $\mu\text{s}$ per NTL call: for `GF(256), k=98`, `generator_matrix()` takes **16.29 s** (and `478.3 ms` even at `k=16`). Replacing `poly(alpha)` with a vectorized NumPy $\mathbb{F}_{2^m}$ log/antilog Vandermonde evaluator drops `k=16, n=256` from `478.3 ms` to **`0.159 ms` (3,008× faster; `0.302 ms` / 1,584× including Sage polynomial extraction)**.
3. **Un-overridden vector–matrix multiply on `Matrix_gf2e_dense` (`P1`):** `Matrix_gf2e_dense` does not implement `_vector_times_matrix_`, so `Encoder.encode` (`vector(word) * G`) and `minimum_distance_py` (`tail_vec * sub_gen`) fall back to `Matrix._vector_times_matrix_` (`sage/matrix/matrix0.pyx:5312`), which unpacks every M4RIE row into Python/Givaro lists on every call. Batching 1,000 messages into a single M4RIE matrix product `M * G` runs in **`0.378 ms` vs `393.6 ms` (1,040× speedup)**.
4. **Slow Python enumeration in `minimum_distance_py` (`P1`):** Because modular `passagemath` lacks `libgap` (`code.minimum_distance()` raises `ImportError`), exact distance falls back to Python projective enumeration without `design_distance()` early termination, incremental updates, or vectorization. Combining reverse `lead` order, `design_distance()` early stop, incremental XOR, and NumPy `uint8` / 64-bit bit-sliced evaluation yields **215×–277× worst-case full-scan speedup** (`GF(64) k=4`: `14.92 s → 55.0 ms`) and up to **136,000×** when `d_min == d_design`.

Architectural critique: `TamoBargCodeTwoSets.__init__` eagerly builds both subgroup tables (even unused types), the NetworkX bipartite graph `_G`, and the full encoder intersection basis (just to validate `k <= max_dimension`, a check then duplicated inside `TamoBargTwoSetsVectorEncoder.__init__`). Because `k` is required at construction, callers wanting `k = max_dimension` (such as `simulation/runner.py:361-375`) construct the code **twice**, doubling the ~1 s init cost.

---

## 2. SageMath Backend Map (Verified from Installed `passagemath` 10.8.x Source)

| Field class | Example | Field backend | `PolynomialRing` backend | `matrix` backend | `vector` backend |
|---|---|---|---|---|---|
| $\mathbb{F}_{2^m}, m \le 15$ | `GF(16)..GF(4096)` | `FiniteField_givaro` (Givaro C++ Zech-log) | `Polynomial_ZZ_pEX` (NTL `ZZ_pEX`, generic `ZZ_p`!) | `Matrix_gf2e_dense` (M4RIE C) | `FreeModuleElement_generic_dense` (Cython Python-obj list) |
| $\mathbb{F}_{2^{16}}$ | `GF(65536)` | `FiniteField_ntl_gf2e` (NTL `GF2E`) | `Polynomial_ZZ_pEX` (NTL `ZZ_pEX`) | `Matrix_gf2e_dense` (M4RIE C) | `FreeModuleElement_generic_dense` |
| $\mathbb{F}_p$ prime | `GF(13)`, `GF(257)` | `FiniteField_prime_modn` | `Polynomial_zmod_flint` (FLINT C) | `Matrix_modn_dense_float/double` (LinBox/FFLAS-FFPACK) | `Vector_modn_dense` (C array) |
| $\mathbb{F}_{p^m}$, odd $p, m \ge 2$ | `GF(25)`, `GF(27)`, `GF(125)` | `FiniteField_givaro` | `Polynomial_ZZ_pEX` (NTL) | `Matrix_generic_dense` (generic Cython, **no M4RIE/LinBox**) | `FreeModuleElement_generic_dense` |

---

## 3. Mandatory Per-Function Analysis

### 3.1 Module-Level Helpers (`src/tamo_barg_lrc/tamo_barg_two_sets.py:52-74`)

| Field | `_is_int_like` | `_int_log_base` | `_safe_parity_check_matrix` |
|---|---|---|---|
| **Function** | `_is_int_like(value: Any) -> bool` (L52). True iff `value` is `numbers.Integral` and not `bool` (handles Python `int` and Sage `Integer`). | `_int_log_base(size: int, base: int) -> int` (L57). Exact integer $\log_{\text{base}}(\text{size})$ via repeated division; raises `ValueError` if not an exact power. | `_safe_parity_check_matrix(grs_code: GeneralizedReedSolomonCode) -> Any` (L69). Returns parity-check matrix $H$ of a GRS code, safely returning a $0 \times m$ matrix when $k_{\text{GRS}} = m$. |
| **Algorithm** | ABC `isinstance` check against `numbers.Integral` excluding `bool` subclass. Fits Python + Sage `Integer` interoperability. | Trial division loop `while rem > 1: rem //= base; exp += 1`. Exact without float rounding for prime-power subgroup sizes $p^t$. | Fast-path guard `if dim == len: return matrix(F, 0, len)`, else delegates to `grs_code.parity_check_matrix()`, which computes `self.dual_code().generator_matrix()` via Lagrange/Cramer column multipliers $\eta_i = \prod_{h\ne i}(\alpha_i - \alpha_h)^{-1}$ and a scaled Vandermonde matrix — **not** a matrix kernel. |
| **Complexity** | Time: $O(1)$. Space: $O(1)$. | Time: $O(\log_{\text{base}}(\text{size})) = O(m)$ divisions for $q = p^m$. Space: $O(1)$. | Let $m = \text{len}$, $r_H = m - k_{\text{GRS}}$. Full-rate ($r_H=0$): $O(1)$ time, $O(1)$ space. General (uncached): `multipliers_product()` does $m(m-1)$ field ops ($O(m^2)$) + `GRSEvaluationVectorEncoder.generator_matrix()` builds $r_H \times m$ via `alphas[j]**i` ($O(r_H m \log r_H)$); total $O(m^2 + r_H m \log r_H)$ time, $O(r_H m)$ space. Cached: $O(1)$. |
| **Sage primitives** | None (stdlib `numbers.Integral`; Sage `Integer` registers with it). | None (pure Python integer arithmetic). | `matrix(F, 0, m)` (`sage/matrix/constructor.pyx`); `GeneralizedReedSolomonCode.parity_check_matrix()` (`sage/coding/grs_code.py:439`, `@cached_method`) $\to$ `dual_code()` (L458) $\to$ `parity_column_multipliers()` (L419) $\to$ `multipliers_product()` (L396, pure-Python $O(m^2)$ `prod(...)`) $\to$ `GRSEvaluationVectorEncoder.generator_matrix()` (L760, Python `lambda i,j: col_mults[j]*alphas[j]**i` packed into `Matrix_gf2e_dense` / `Matrix_modn_dense_*` / `Matrix_generic_dense`). **Zero LinBox/M4RIE kernel calls.** |
| **Measured** | `int`: `0.782 µs` (`p10=0.557`, `p90=0.978`), peak `200 B`. `bool`: `0.294 µs`. | `(256,2)`: `0.840 µs` (`0.797`, `1.234`), peak `0 B`. `(65536,2)`: `1.705 µs`. | On `GF(256)` (`d=3`, $r_H=2$): uncached `0.211 ms` ($m=4$), `0.268 ms` ($m=16$), `0.708 ms` ($m=64$), `12.44 ms` ($m=256$, `p10=12.02`, `p90=14.63`); half-rate ($r_H=m/2$): `28.20 ms` ($m=256$, peak `25.8 KiB`). Empirical slope ($m \in [16,256]$, $d=3$): **`1.45`** (transitioning from fixed-overhead to $O(m^2)$: `128→256` is $3.08\times$, half-rate `128→256` is $3.49\times$). Cached: **`0.21 µs`**. Full-rate ($k=m$): **`0.019 ms`**. |
| **Bottleneck** | Python ABC dispatch overhead (negligible). | Negligible ($<2\,\mu\text{s}$, called $\le 2\times$ in `__init__`). | `cProfile` at $m=256, k=128$: `grs_code.py:416(<genexpr>)` (`multipliers_product`, 65,536 calls, `49.4%`), `grs_code.py:792(<lambda>)` (32,768 Vandermonde calls, `28.1%`), `matrix_space.py:1036` (`16.9%`). |
| **Correctness risks** | `numpy.integer` also satisfies `numbers.Integral` (fine, coerced via `int()`). | Returns `0` silently for `size <= 0` and infinite-loops if `base <= 1` and `size > 1` (safe for current callers where `base = p >= 2`, `size >= 1`). | (1) Returns **mutable** `0 x m` matrix when `k==m` vs **immutable** cached matrix when `k<m`. (2) Helper itself is uncached, so full-rate branch re-allocates on every call. (3) Relies on `GeneralizedReedSolomonCode` enforcing `1 <= dim <= len` (otherwise `dim==0` fails in Sage). |
| **Better algorithm** | N/A (replace `isinstance(x, numbers.Integral)` with `(int, Integer)` if ever hot). | For `base == 2`: `if size > 0 and (size & (size - 1)) == 0: return size.bit_length() - 1` ($O(1)$). | (1) Compute $\eta_i = 1 / A'(\alpha_i)$ where $A(x)=\prod(x-\alpha_j)$ via formal derivative + multipoint eval ($O(m \log^2 m)$) or vectorized NumPy log-table ($O(m^2)$ C-speed, $<0.1\text{ ms}$ at $m=256$). (2) Build Vandermonde by row recurrence $H_{i+1,j} = H_{i,j}\cdot \alpha_j$ instead of `alphas[j]**i` (eliminates $\log r_H$ factor). (3) For cosets of a subgroup $H$, cosets are affine/multiplicative shifts so $\eta_i$ is identical up to a scalar across all cosets — compute **once** per partition instead of $p_i$ times! |
| **Sage-replacement difficulty** | **1** (pure Python). | **1** (pure Python). | **2** — Needs $\mathbb{F}_q$ Vandermonde + Lagrange multiplier routine (~25 LOC in NumPy/Galois). |
| **Priority** | **P2** (trivial cost). | **P2** (trivial cost). | **P1** (called once per local coset + global GRS during decoder construction; $28\text{ ms}$ for global $m=256$ half-rate, and $O(m^2)$ at $m=1024$ reaches ~0.5 s). |

---

### 3.2 `TamoBargCodeTwoSets` Core Methods (`src/tamo_barg_lrc/tamo_barg_two_sets.py:76-414`)

| Field | `TamoBargCodeTwoSets.__init__` | `TamoBargCodeTwoSets.design_distance` | `TamoBargCodeTwoSets.minimum_distance_py` |
|---|---|---|---|
| **Function** | `TamoBargCodeTwoSets.__init__(self, base_field, length, dimension, locality, local_minimum_distance=(2,2), sub_group_type=('add','add'), shift_add=False, subgroup=None)` (L87). Validates parameters, selects/validates subgroups $H_0,H_1$, partitions evaluation points, builds bipartite coset graph `_G`, initializes `AbstractLinearCode`, and validates `k <= max_dimension()` (triggering eager encoder construction). | `TamoBargCodeTwoSets.design_distance(self) -> int` (L366). Returns enveloping-GRS polynomial degree bound $d_{\text{design}} = n - \max_{b \in \mathcal{B}_k} \deg(b)$. | `TamoBargCodeTwoSets.minimum_distance_py(self) -> int` (L380). Computes exact minimum Hamming distance by projectively normalized (monic) enumeration over $\mathbb{F}_q^k \setminus \{0\}$. |
| **Algorithm** | Multi-phase setup: (1) scalar/shape validation; (2) enumerate all mult & canonical add subgroups, pick/shift $H_0, H_1$; (3) partition $\mathbb{F}_q$ (or $\mathbb{F}_q^*$) into cosets, union & sort evaluation points, map indices; (4) build `nx.MultiGraph` via `create_bipartite_coset_graph`; (5) `super().__init__` + eager `self.max_dimension()` $\to$ `TamoBargTwoSetsVectorEncoder.__init__`. | Linear scan over `self.encoder().enc_basis()` (`k` polynomials) taking `max(int(poly.degree()))` and subtracting from `n`. Valid because every non-zero codeword evaluates a non-zero polynomial of degree $\le \text{max\_deg} < n$ on $n$ distinct points, yielding $\le \text{max\_deg}$ roots. | For `lead = 0 .. k-1`: fixes $m_{\text{lead}} = 1$, $m_{<\text{lead}} = 0$, iterates all `tail_vec` $\in \mathbb{F}_q^{k-\text{lead}-1}$, computes `cw = lead_row + tail_vec * sub_gen`, updates `min_wt`, exits early only if `min_wt == 1`. Avoids $(q-1)\times$ scalar redundancy over naive $q^k$ enumeration. |
| **Complexity** | Let $n \le q$, $c_i = r_i + d_i - 1$, $p_i = \Theta(n/c_i)$, $B_i = r_i p_i \le n$. Phases 1–6: $O(q \log q)$ time/space. Phase 7 (`encoder.__init__`): `build_algebra_basis` is $O(\sum_i p_i^2 c_i) = O(n^2 / c_i)$ poly ops; `intersect_encoding_bases` is $O((B_0+B_1)n)$ marshaling + $O(n^3)$ RREF (`left_kernel_matrix` + matmul + `echelon_form`) + $O(k_{\max} n)$ poly reconstruction. Total: **$O(n^3)$ asymptotic**, **$O(n^2)$ dominated by NTL marshaling for $n \le 1024$**. Space: $O(n^2)$. | Time: $O(k)$ (actually $O(1)$ possible since `enc_basis()` is sorted ascending by degree so `basis[-1].degree()` is already max!). Space: $O(1)$. | Projective count $P(q,k) = \frac{q^k - 1}{q - 1} = \Theta(q^{k-1})$. Per codeword: `tail_vec * sub_gen` costs $O((k - \text{lead} - 1) n)$ because M4RIE rows are unpacked and summed from scratch, plus $O(n)$ `hamming_weight()`. Total time: **$O\!\left(k n \frac{q^k - 1}{q - 1}\right) = O(k n q^{k-1})$**. Space: $O(k n)$ for `sub_gen` + $O(n)$ per codeword vector. |
| **Sage primitives** | `GF` methods (`is_finite`, `characteristic`, `cardinality`), `list_multiplicative_subgroups` (`arith.misc.divisors`), `list_canonical_additive_subgroups`, `shift_additive_subgroup`, `list_subgroup_cosets`, `AbstractLinearCode.__init__` (`sage/coding/linear_code.py`), `self.encoder()` (`abstract_code.py:906`, `@cached_method`) $\to$ `PolynomialRing(F,'x')` (**NTL `Polynomial_ZZ_pEX`**), `Matrix_gf2e_dense` (M4RIE `mzed_echelonize`, `matrix_gf2e_dense.pyx:1002`), `Matrix.left_kernel_matrix` (`matrix2.pyx:4966` $\to$ `_right_kernel_matrix_over_field` at L4141 + redundant second `echelonize()` at L4945). | `self.encoder()` (cached, `0.10 µs`), `Polynomial_ZZ_pEX.degree()` (`polynomial_template`, Cython $O(1)$). | `self.generator_matrix()` (cached M4RIE `Matrix_gf2e_dense`), `gen_mat.row(lead)`, `gen_mat.matrix_from_rows(...)`, `VectorSpace(field, tail_dim).__iter__()` (`free_module.py:2412`, Python generator calling `linear_combination_of_basis`), `tail_vec * sub_gen` (falls back to `Matrix._vector_times_matrix_` at `matrix0.pyx:5312`: unpacks every row via `row(i, from_list=True)`!), `cw.hamming_weight()` (`free_module_element.pyx:3820`: copies `self.list()` + Python `is_zero()` loop). |
| **Measured** | Fixed $c=4, r=[2,2], d=[3,3], k=4$: `GF(16)` `5.64 ms` (`3.17`, `7.72`, peak `26 KiB`); `GF(64)` `64.26 ms` (`55.39`, `94.95`, `91 KiB`); `GF(256)` `1,264 ms` (`1,114`, `1,316`, `539 KiB`); `GF(1024)` `25,372 ms` (`21,787`, `25,880`, `17.5 MiB`). Empirical slope vs $q$: **`2.04`** (`16..1024`), **`2.16`** (`64..1024`). Realistic `GF(256)`: `global_sim` (`r=[2,4], d=[3,5]`) `953.1 ms`; `iter_sim` (`r=[14,3], add/mult`) `914.6 ms`; `balanced` (`r=[14,14]`) `1,122 ms`; `mult/mult` (`r=[13,15]`) `777.7 ms`. | `GF(256), k=16`: **`3.637 µs`** (`3.494`, `13.74`), peak `488 B` (uncached, $O(k)$). | `GF(16)`: $k=2$ `0.181 ms` (`0.170`, `0.677`, `2.7 KiB`), $k=3$ `6.513 ms`, $k=4$ `137.4 ms` (ratios $36.0\times$, $21.1\times$ vs $16\times$ theory because $(k-1)$ factor grows). `GF(64)`: $k=2$ `2.481 ms`, $k=3$ `213.6 ms` (`195.7`, `238.4`, `6.9 KiB`), $k=4$ `14,924 ms` (1-shot, $69.9\times$). `GF(256)`: $k=2$ `24.87 ms`, $k=3$ `7,644 ms` (1-shot, $307\times$). Empirical exponent vs $q^{k-1}$: matches $\Theta(k n q^{k-1})$. |
| **Bottleneck** | `TamoBargTwoSetsVectorEncoder.__init__` is **98.9%** of `__init__` (`cProfile`): `build_algebra_basis` (`44.4–48.3%`, NTL `ZZ_pEX` mul) + `intersect_encoding_bases` marshaling (`padded_list` `15.7–23.7%` + `R(coeffs)` `22.8–50.9%`). Actual M4RIE RREF/kernel is only **2.2% at `GF(256)`** and **9.6% at `GF(1024)`**. | None (`<4 µs`). | `cProfile` (`GF(64), k=3`, `310 ms`): `minimum_distance_py` body (`57.1%` = Cython `tail_vec * sub_gen` row-unpacking + `hamming_weight`) + `FreeModule._element_constructor_` (`29.7%`, 8,327 calls) + `FreeModule.__iter__` (`25.8%`, 4,162 calls). |
| **Correctness risks** | (1) Unconditionally enumerates BOTH `mult_subgroups` and `add_subgroups` even when unused or overridden. (2) Duplicate `1 <= k <= max_dim` check (already raised by `encoder.__init__`). (3) Mixed `add/mult` with `n < q` excludes `0` differently from `n == q`; relies on `len(all_pts) == length_int` check. (4) No `k=None` option forces callers (`runner.py:361`) to construct the code twice. | Only a lower bound ($d_{\min} \ge d_{\text{design}}$); can underestimate true $d_{\min}$ by $+1$ to $+6$ when basis has degree gaps (see §5). Assumes `enc_basis()` non-empty and `deg < n` (guaranteed by init). | (1) Only exits early at `min_wt == 1`, never at `min_wt == design_distance()`. (2) Iterates `lead = 0..k-1` (largest $q^{k-1}$ block first) instead of reverse `k-1..0`. (3) Infeasible beyond `(GF(64), k=4)` or `(GF(256), k=3)`. |
| **Better algorithm** | (1) Use `PolynomialRing(F, 'x', implementation='generic')` (1-line fix: **15× faster** at `GF(256)`, **9.2×** at `GF(1024)`), or pure NumPy/GF($2^m$) polynomial + RREF (**>50×**). (2) Lazy `_G` bipartite graph (only needed by `expander.py`). (3) Compute only requested subgroup size, not all divisors/subgroups. (4) Pass `basis='computed'` to `left_kernel_matrix()` (saves redundant M4RIE echelonization). (5) Support `dimension=None` $\to$ `max_dimension` to eliminate probe double-init. | Return `int(self.length()) - int(self.encoder().enc_basis()[-1].degree())` ($O(1)$) and cache it. | (1) Reverse `lead` order (`k-1` down to `0`) + early stop at `min_wt <= design_distance()`. (2) Incremental Gray-code/odometer XOR updates ($O(n)$ per codeword instead of $O(kn)$). (3) NumPy `uint8` vectorized batching or `uint64` bit-plane slicing for $\mathbb{F}_{2^m}$ (**215×–277×** worst-case; up to **136,000×** when tight). (4) For larger $k$, Brouwer–Zimmermann information-set enumeration. |
| **Sage-replacement difficulty** | **3** — Needs $\mathbb{F}_q$ arithmetic, univariate poly ops, and finite-field RREF (NumPy/Cython or `galois` / `flint` / `gf256`). | **1** (trivial integer subtraction). | **1–2** — Pure NumPy `uint8` / `uint64` bit-sliced implementation is ~60 LOC, zero Sage needed, and **>200× faster**. |
| **Priority** | **P0** (1–25 s per code construction; called twice per simulation when `k=None`). | **P2** (`3.6 µs`, though critical semantically for global decoder radius). | **P1** (high speedup 200×–136,000×; unlocks `GF(64) k=5` and `GF(256) k=4` exact distance checks). |

---

### 3.3 `TamoBargCodeTwoSets` Dunder, Metadata & Accessor Methods (`tamo_barg_two_sets.py:300-379`)

| Function | Algorithm | Complexity | Sage primitives | Measured (`GF(256), k=16`) | Bottleneck / Risks / Replacement | Priority |
|---|---|---|---|---|---|---|
| `__eq__(self, other) -> bool` (L300) | Short-circuiting equality on type, `base_field`, `length`, `dimension`, `locality`, `local_minimum_distance`, `sub_group`. | $O(c_0 + c_1)$ time, $O(1)$ space. | `base_field()` / element equality. | `1.141 µs` (`1.106`, `5.178`), peak `64 B`. | Consistent with `__hash__`. Risk: internal lists are exposed mutably by accessors; mutating them breaks equality/hashing. Difficulty: **1**. | P2 |
| `__hash__(self) -> int` (L312) | Hashes tuple of immutable parameters + `tuple(tuple(sg) for sg in sgs)` + `tuple(sg_types)`. | $O(c_0 + c_1)$ time/space. | `FiniteField.__hash__`, element `__hash__`. | `7.021 µs` (`2.975`, `7.264`), peak `536 B`. | Allocates nested tuples per call; cache `_hash` since code is logically immutable. Difficulty: **1**. | P2 |
| `_repr_(self) -> str` (L327) | f-string formatting `[n, k, locality] Tamo-Barg Code over GF(q)`. | $O(1)$ time/space. | `base_field().cardinality()`. | `6.516 µs` (`2.442`, `6.601`), peak `1,600 B`. | None. Difficulty: **1**. | P2 |
| `max_dimension(self) -> int` (L372) | Delegates to `int(self.encoder().max_dimension())`. | $O(1)$ cached; triggers full encoder init if uncached. | `self.encoder()` (`@cached_method`). | `0.220 µs` (`0.219`, `0.233`), peak `64 B` (cached). | Already triggered eagerly in `__init__`. Difficulty: **1**. | P2 |
| `max_rate(self) -> Any` (L376) | Returns `self.max_dimension() / self.length()`. | $O(1)$ time/space. | `self.length()` returns Python `int` (or Sage `Integer` if base returns `ZZ`, giving rational `QQ` vs Python `float`!). | `0.349 µs` (`0.344`, `0.367`), peak `64 B`. | Subtle type risk: `max_dimension()` returns `int`, `self.length()` in `AbstractCode` returns `Integer` (`ZZ`) $\to$ division yields a Sage `Rational` (`QQ`), not `float`! Difficulty: **1**. | P2 |
| 8 trivial accessors: `bipartite_graph` (L334), `partition` (L338), `sub_group` (L342), `evaluation_points_idx` (L346), `evaluation_points` (L350), `locality` (L354), `num_of_sets` (L358), `local_minimum_distance` (L362) | Direct attribute / tuple return. | $O(1)$ time, $O(1)$ space. | None (`bipartite_graph` returns `nx.MultiGraph`). | `bipartite_graph()`: `0.143 µs` (`0.140`, `0.259`), peak `64 B` (representative of all 8). | Correctness risk: all return direct references to mutable internal `list`/`nx.MultiGraph` attributes (caller mutation corrupts cached invariants). `_G` in `bipartite_graph()` should be built lazily on first access. Difficulty: **1**. | P2 |

---

### 3.4 `TamoBargTwoSetsVectorEncoder` (`src/tamo_barg_lrc/tamo_barg_two_sets.py:416-509` + inherited `Encoder` methods)

| Field | `TamoBargTwoSetsVectorEncoder.__init__` | `TamoBargTwoSetsVectorEncoder.enc_poly` | `TamoBargTwoSetsVectorEncoder.generator_matrix` | Inherited `Encoder.encode` / `unencode` |
|---|---|---|---|---|
| **Function** | `__init__(self, code: TamoBargCodeTwoSets) -> None` (L419). Builds $g_i(x)$, algebra bases, per-partition encoding bases, and degree-minimal intersection basis sliced to `k`. | `enc_poly(self, message: Sequence[Any]) -> Any` (L481). Forms $f_m(x) = \sum_{i=0}^{k-1} m_i b_i(x) \in \mathbb{F}_q[x]$. | `@cached_method generator_matrix(self) -> Any` (L497). Builds and caches immutable $k \times n$ evaluation matrix $G_{i,j} = b_i(\alpha_j)$. | `encode(word)` (`encoder.py:119`: `vector(word) * G`); `unencode(c, nocheck=False)` (`encoder.py:200`: membership check + `cc * U`). |
| **Algorithm** | Loops `i in (0,1)`: `build_good_polynomial` $\to$ `build_algebra_basis` $\to$ `build_encoding_basis`; then `intersect_encoding_bases(R, B_0, B_1, n)` via Zassenhaus left kernel + reverse-col RREF; slices `[:k]`. | Delegates to `evaluate_message_polynomial(R, comb_enc_basis, message)`: linear combination `sum(base_ring(m[i]) * b[i])`. | Nested list comprehension `[[poly(alpha) for alpha in alphas] for poly in comb_enc_basis]` passed to `matrix(F, k, n, ...)` + `set_immutable()`. | `encode`: membership check `word in M`, then `vector(word) * G`. `unencode`: `c in code` (via `parity_check_matrix` / syndrome check), then extracts `information_set` coords and multiplies by cached inverse $U = G_I^{-1}$. |
| **Complexity** | Time: $O(\sum_i p_i^2 c_i + n^3)$ (see §4 for phase breakdown). Space: $O(n^2)$ (two $B_i \times n$ matrices + stacked $(B_0+B_1) \times n$ matrix + basis lists). | Time: $O(k \cdot \text{max\_deg}) \le O(k n)$ field ops (scalar-poly multiply + add in NTL `ZZ_pEX`). Space: $O(\text{max\_deg})$. | Time: **$O(k n \cdot \text{max\_deg})$** ($k n$ Horner evaluations of degree $\le \text{max\_deg}$, i.e., **$O(n^3)$** when $k, \text{max\_deg} = \Theta(n)$). Space: $O(k n)$ (plus temporary Python list of $k n$ Givaro objects). | `encode`: $O(k n)$ time, $O(n)$ space. `unencode(nocheck=True)`: $O(k^2)$ after $O(k^2 n)$ one-time `_unencoder_matrix()` cache. `unencode(nocheck=False)`: adds $O((n-k)n)$ parity-check syndrome check (plus $O(n^3)$ one-time `code.parity_check_matrix()` via `G.right_kernel()`!). |
| **Sage primitives** | `Encoder.__init__`, `PolynomialRing(F, 'x')` (**NTL `Polynomial_ZZ_pEX`**), `prod`, `Polynomial_ZZ_pEX._mul_`, `.shift()`, `.padded_list()`, `matrix(F, ...)` (M4RIE `Matrix_gf2e_dense`), `stack()`, `left_kernel_matrix()`, `echelon_form()`. | `base_ring(m[i])` (`FiniteField_givaro`), `Polynomial_ZZ_pEX._rmul_` (`polynomial_zz_pex.pyx:221` converts `left.polynomial()` to NTL `ZZ_pE` per term!) + `ZZ_pEX_add`. | `Polynomial_ZZ_pEX.__call__` (`polynomial_zz_pex.pyx:285-287`: calls `a.polynomial()` $\to$ `list` $\to$ `ZZ_pE` $\to$ `ZZ_pEX_eval` $\to$ `ZZ_pE_c_to_list` $\to$ `K(...)` on EVERY $(i,j)$!), `matrix(F, k, n, ...)` (M4RIE), `@cached_method`. | `Encoder.message_space()`, `Matrix._vector_times_matrix_` (`matrix0.pyx:5312`: Python row-unpack loop!), `AbstractLinearCodeNoMetric.__contains__`, `information_set()` (echelon pivots), `matrix_from_columns().inverse()` (M4RIE). |
| **Measured** | Accounts for **98.9%** of `TamoBargCodeTwoSets.__init__`: `GF(16)` `~4 ms`, `GF(64)` `~63 ms`, `GF(256)` `910–1,260 ms`, `GF(1024)` `25,340 ms` (slope `2.16`). Cached `code.encoder()`: `0.100 µs`. | `GF(256), k=16`: **`0.888 ms`** (`0.882`, `1.299`), peak `2.6 KiB`. Evaluating `enc_poly(m)` at all $n=256$ pts: **`381.2 ms`** (`302.3`, `408.5`). | Uncached `GF(256)`: $k=4$ (`c=4`) `154.5 ms`; $k=6$ (`global_sim`) `241.7 ms` (peak `15.2 KiB`); $k=16$ (`balanced`) **`478.3 ms`** (`466.1`, `511.4`, peak `36 KiB`); $k=98$ (`balanced`) **`16,288 ms`**! Fixed $k=4$ sweep `GF(16..1024)`: `5.66 → 28.77 → 154.5 → 770.9 ms` (slope `1.19`). Cached: **`0.279 µs`**. | `GF(256), k=16`: `enc.encode(m)` **`0.354 ms`** (`0.150`, `0.478`, peak `41.2 KiB`); `code.encode(m)` `0.664 ms`; `unencode(nocheck=False)` **`5.730 ms`** (peak `572 KiB`); `unencode(nocheck=True)` **`0.077 ms`** (`0.072`, `0.289`, peak `5.1 KiB`). |
| **Bottleneck** | `build_algebra_basis` (`44–48%`, NTL `ZZ_pEX` mul) + `intersect_encoding_bases` NTL$\leftrightarrow$Givaro marshaling (`42–75%`). | `ZZ_pEX` scalar-mul marshaling (`left.polynomial()` per coefficient). | `Polynomial_ZZ_pEX.__call__` marshaling + NTL `ZZ_pEX_eval` (`>99.8%` of time; M4RIE matrix packing is `<0.5 ms`). | `encode`: un-overridden `Matrix._vector_times_matrix_` unpacking M4RIE rows into Python vectors. `unencode`: `c in self.code()` syndrome check (`74×` overhead). |
| **Correctness risks** | Retains full `_algebra_basis` and `_enc_basis` (all $B_0 + B_1$ polynomials) in memory even when $k \ll k_{\max}$. | Coerces `message[i]` via `base_ring(...)` (accepts ints/list); consistent with `encode(m)` (`test_tamo_barg_two_sets.py`). | None mathematically; severe wall-time cliff for $k = \Theta(n)$ ($>16\text{ s}$ at `GF(256)`, ~17 min at `GF(1024)`). | `code.encode(m)` adds `0.31 ms` dispatch overhead over `enc.encode(m)`. `unencode(c)` with `nocheck=False` silently triggers full `code.parity_check_matrix()` ($O(n^3)$) on first call! |
| **Better algorithm** | `PolynomialRing(F, 'x', implementation='generic')` (**10–15×** speedup, 1-line fix) + `left_kernel_matrix(basis='computed')`, or pure NumPy/GF($2^m$) coefficient-matrix algebra (skip polynomials entirely during intersection!). | Vectorized dot product of message vector with $k \times (\text{max\_deg}+1)$ coefficient matrix over $\mathbb{F}_q$ (**>50×** faster). | (1) `implementation='generic'` (`11.49 ms`, **41.6×**). (2) Pure Givaro Horner (`6.58 ms`, **72.7×**). (3) NumPy $\mathbb{F}_{2^m}$ log/antilog Vandermonde (`0.159 ms` pure / `0.302 ms` w/ extraction = **1,584×–3,008×**!). (4) Additive FFT over $\mathbb{F}_{2^m}$ for $n \ge 4096$: $O(k n \log^2 n)$. | Batch $B$ messages as a single $B \times k$ matrix multiply `M * G` in M4RIE (**1,040×** at $B=1000$) or precomputed NumPy LUT (**481×** at $B=1000$, **465×** at $B=100$). Pass `nocheck=True` in decoders when codeword membership is already verified. |
| **Sage-replacement difficulty** | **3** — $\mathbb{F}_q$ poly ops + RREF. | **1** — Matrix-vector product of `message` with coeff matrix. | **1–2** — NumPy log/antilog Vandermonde is ~20 LOC, zero Sage needed. | **2** — NumPy LUT or `galois` / Cython matmul + $k \times k$ inverse for `unencode`. |
| **Priority** | **P0** (dominates `__init__`). | **P2** (rarely on hot path; tests/inspection only). | **P0** (up to **16.3 s** at `GF(256)` and **3,008×** speedup available). | **P1** (1,040× batch speedup for Monte Carlo simulations / throughput encoding). |

*Note on trivial encoder accessors:* `_repr_` (L461), `enc_basis` (L465), `enc_basis_ext` (L469), `enc_algebra_basis` (L473), `enc_good_poly` (L477), `max_dimension` (L492) are all $O(1)$ attribute/string returns (`<1 µs`, `0–64 B`, Priority **P2**, Difficulty **1**; same mutable-list exposure caveat as code accessors).

---

## 4. Deep Dive: Phase-by-Phase `TamoBargCodeTwoSets.__init__` Profile & Scaling

### 4.1 Scaling Sweep Across $q \in \{16, 64, 256, 1024\}$ (Fixed $c=4, r=[2,2], d_{\text{local}}=[3,3], k=4$, `add/add`)

| Phase | `GF(16)` (ms) | `GF(64)` (ms) | `GF(256)` (ms) | `GF(1024)` (ms) | % of `init` (`GF(256)`) | % of `init` (`GF(1024)`) | Slope `16..1024` | Slope `64..1024` | Theoretical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1. `list_multiplicative_subgroups` | 0.034 | 0.101 | 0.467 | 5.252 | 0.04% | 0.02% | 1.20 | 1.43 | $O(q \log \log q)$ |
| 2. `list_canonical_additive_subgroups` | 0.030 | 0.086 | 0.285 | 5.429 | 0.02% | 0.02% | 1.21 | 1.50 | $O(q \log q)$ |
| 3. Select / shift subgroups | 0.048 | 0.089 | 0.117 | 0.136 | 0.01% | 0.00% | 0.25 | 0.15 | $O(c_i \log c_i)$ |
| 4. `list_subgroup_cosets` (×2) | 0.046 | 0.233 | 0.530 | 6.813 | 0.04% | 0.03% | 1.14 | 1.22 | $O(q \log c_i)$ |
| 5. Eval pts union + sort + index map | 0.018 | 0.057 | 0.205 | 5.020 | 0.02% | 0.02% | 1.33 | 1.62 | $O(n \log n)$ |
| 6. `create_bipartite_coset_graph` | 0.054 | 0.166 | 0.695 | 6.757 | 0.05% | 0.03% | 1.15 | 1.34 | $O(n)$ |
| 7a. `build_good_polynomial` (×2) | 0.187 | 0.279 | 0.323 | 0.421 | 0.03% | 0.00% | 0.19 | 0.15 | $O(c_i \log^2 c_i)$ |
| **7b. `build_algebra_basis` (×2)** | **0.082** | **24.40** | **566.8** | **12,261** | **44.8%** | **48.3%** | **2.81** | **2.24** | $O(p_i^2 c_i) = O(n^2)$ |
| 7c. `build_encoding_basis` (×2) | 0.024 | 0.236 | 7.343 | 78.08 | 0.58% | 0.31% | 2.00 | 2.09 | $O(r_i p_i \cdot n) = O(n^2)$ |
| **7d1. Intersect: `padded_list` + `matrix`** | **0.316** | **15.73** | **198.0** | **4,788** | **15.7%** | **18.9%** | **2.27** | **2.06** | $O(n^2)$ marshaling |
| 7d2. Intersect: `left_kernel_matrix` (M4RIE) | 0.188 | 0.683 | 21.14 | 1,915 | 1.67% | 7.55% | 2.24 | **2.86** | **$O(n^3)$** |
| 7d3. Intersect: `ker * mat_a` + rev RREF | 0.082 | 0.166 | 6.818 | 525.5 | 0.54% | 2.07% | 2.16 | **2.91** | **$O(n^3)$** |
| **7d4. Intersect: `R(coeffs)` poly recon** | **0.300** | **23.44** | **337.7** | **5,789** | **26.7%** | **22.8%** | **2.33** | **1.99** | $O(k_{\max} n) = O(n^2)$ |
| **Actual `TamoBargCodeTwoSets.__init__`** | **5.644** | **64.26** | **1,264** | **25,372** | **100%** | **100%** | **2.04** | **2.16** | $O(n^2)$ → $O(n^3)$ |
| `generator_matrix()` (fixed $k=4$) | 5.660 | 28.77 | 154.5 | 770.9 | 12.2% | 3.04% | 1.18 | 1.19 | $O(k n \cdot \text{deg}_4)$ |
| Peak memory (`__init__` / `G`) | 26 / 2.8 KiB | 91 / 4.5 KiB | 539 / 11 KiB | 17.5 MiB / 325 KiB | — | — | — | — | $O(n^2)$ |

### 4.2 Realistic `GF(256)` Configurations & Impact of 1-Line `implementation="generic"` Fix

| Configuration (`GF(256)`) | $k / k_{\max}$ | `init` (NTL default) | `init` (`impl='generic'`) | Speedup | `G` (NTL default) | `G` (`impl='generic'`) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `global_sim` (`r=[2,4], d=[3,5], add/add`) | 6 / 64 | 953.1 ms (`797, 1207`) | **63.47 ms** (`50.6, 72.9`) | **15.0×** | 241.7 ms (`194, 297`) | **5.04 ms** (`3.91, 6.42`) | **48.0×** |
| `iter_sim` (`r=[14,3], d=[3,3], add/mult`) | 10 / 123 | 914.6 ms (`715, 1030`) | — | — | 360.6 ms (`303, 441`) | — | — |
| `balanced` (`r=[14,14], d=[3,3], add/add`) | 10 / 196 | 1,122 ms | — | — | 248.6 ms (`206, 273`) | — | — |
| `balanced` (`r=[14,14], d=[3,3], add/add`) | 16 / 196 | ~1,122 ms | **157.8 ms** (`143, 223`) | **7.1×** | 478.3 ms (`466, 511`) | **11.49 ms** (`9.56, 17.5`) | **41.6×** |
| `balanced` (`r=[14,14], d=[3,3], add/add`) | 98 / 196 | 1,333 ms (1-shot) | — | — | **16,288 ms** (1-shot) | — | — |
| `mult/mult` (`r=[13,15], d=[3,3], n=255`) | 10 / 195 | 777.7 ms | — | — | 270.1 ms (`253, 297`) | — | — |
| `GF(1024)` (`r=[2,2], d=[3,3], add/add`) | 4 / 256 | 25,372 ms | **2,752 ms** (`2644, 2949`) | **9.2×** | 770.9 ms (`606, 791`) | **18.44 ms** (`17.3, 23.0`) | **41.8×** |

**Why it dominates:** At `GF(256)`, `7b` + `7d1` + `7d4` account for **87–95%** of `__init__` (`cProfile`: `build_algebra_basis` `40.8%`, `polynomial_ring._element_constructor_` `30.1%`, `finite_field_givaro._element_constructor_` `16.2%`). At `GF(1024)` with `impl='generic'`, NTL marshaling drops from `22.8 s` to `0.31 s` (**73×**), leaving M4RIE `left_kernel_matrix` + `rref` (`2.44 s`, slope `2.86–2.91`) as the true $O(n^3)$ core.

**What can be made lazy or skipped in `__init__`:**
1. `_G, _G_L, _G_R = create_bipartite_coset_graph(...)`: unused by encoding or any decoder (only used by `expander.py`); make a `@cached_method` on `bipartite_graph()`. Saves `0.7–6.8 ms` and NetworkX object memory.
2. `list_multiplicative_subgroups` & `list_canonical_additive_subgroups`: currently **both** are computed unconditionally for all divisors / all dimensions. Only compute the exact requested order `target_size` (and only when `custom_sg is None`).
3. `self.max_dimension()` / eager encoder init: if `dimension=None` is accepted (defaulting to `max_dimension`), `runner.py` avoids constructing the entire code twice (saving 50% of setup time!).

---

## 5. `design_distance` vs True `minimum_distance_py`

**Formula & Assumptions:** `design_distance()` returns $d_{\text{design}} = n - \max_{0 \le i < k} \deg(b_i)$. Since $b_0, \dots, b_{k-1} \in \mathbb{F}_q[x]_{<n}$ are linearly independent polynomials of strictly ascending degree, every non-zero codeword is the evaluation of a non-zero polynomial of degree $\le \text{max\_deg}$ at $n$ distinct points in $\mathbb{F}_q$, so it has at most $\text{max\_deg}$ zeros and weight $\ge n - \text{max\_deg} = d_{\text{design}}$. Equivalently, $d_{\text{design}}$ is the minimum distance of the **enveloping GRS code** $\text{GRS}(A, \text{max\_deg}+1)$, which is used by `TamoBargGlobalDecoder`.

**Tightness Study (19 exact verifications):**

| Code Configuration | Basis Degrees | $d_{\text{design}}$ | Exact $d_{\min}$ | Gap ($d_{\min} - d_{\text{design}}$) | Regime |
|---|---|---:|---:|---:|---|
| `GF(16) add/add c=4,4 k=2` | `[0, 1]` | 15 | 15 | **0** | $k \le \min(r_0,r_1)$ (RS) |
| `GF(16) add/add c=4,4 k=3` | `[0, 1, 8]` | 8 | 12 | **+4** | Degree gap (2..7 absent) |
| `GF(16) add/add c=4,4 k=4` (shifted & unshifted) | `[0, 1, 8, 9]` | 7 | 9 | **+2** | Degree gap (2..7 absent) |
| `GF(16) add/mult c=4,5 k=2` | `[0, 1]` | 15 | 15 | **0** | RS |
| `GF(16) add/mult c=4,5 k=3` | `[0, 1, 12]` | 4 | 10 | **+6** | Huge degree gap (2..11 absent) |
| `GF(16) mult/mult c=3,5 k=2` | `[0, 1]` | 14 | 14 | **0** | RS |
| `GF(16) mult/mult c=3,5 k=3` | `[0, 1, 6]` | 9 | 10 | **+1** | Degree gap (2..5 absent) |
| `GF(64) add/add c=8,8 k=2,3,4` | `[0..k-1]` | 63, 62, 61 | 63, 62, 61 | **0** | $k \le \min(r_0,r_1)$ (RS) |
| `GF(64) add/add c=4,4 k=2,3` | `[0,1]`, `[0,1,8]` | 63, 56 | 63, 56 | **0** | Tight (deg-8 splits in $\mathbb{F}_{64}$) |
| `GF(64) add/add c=4,4 k=4` | `[0, 1, 8, 9]` | 55 | 56 | **+1** | Loose by +1 |
| `GF(64) add/mult c=4,7 k=2,3` & `mult/add c=7,4 k=3` | `[0,1]`, `[0,1,4]` | 63, 60 | 63, 60 | **0** | Tight (deg-4 splits in $\mathbb{F}_{64}$) |
| `GF(64) mult/mult c=7,9 k=2,3` | `[0,1]`, `[0,1,2]` | 62, 61 | 62, 61 | **0** | RS |

**Key takeaway:** $d_{\text{design}}$ is **always tight** when $k \le \min(r_0, r_1)$ (the code contains $\{1, x, \dots, x^{k-1}\}$ and is an MDS RS code), and often tight when the highest basis polynomial splits completely over $A$. When $k > \min(r_0, r_1)$ introduces large degree gaps (missing intermediate monomials), $d_{\text{design}}$ can be substantially pessimistic (up to $+6$, i.e. $2.5\times$ below $d_{\min}$ on `GF(16) add/mult`). This makes $d_{\text{design}}$ a **sound lower bound** for early termination in `minimum_distance_py`, while also explaining why `global_decoding_sim.py` exposes `--exact-distance`.

---

## 6. `minimum_distance_py` Speed-Up Analysis & Max Feasible $(q, k)$

### 6.1 Measured Baseline vs Optimized Prototypes (Exact Distance Verified Against Sage)

Optimizations prototyped in `reports/perf_analysis/scratch/A4/bench_quick_followup.py`:
1. **Early termination at `design_distance()`** (safe because $d_{\min} \ge d_{\text{design}}$ always; even better when iterating `lead` from `k-1` down to `0` so highest-degree basis polynomials are checked first).
2. **Precomputed scalar-multiple table `mul_rows[r, s]`** + **incremental odometer/Gray-code XOR updates** ($O(n)$ per codeword instead of $O(kn)$).
3. **NumPy `uint8` vectorized batching (`U8`)** over the innermost coordinate (`q` codewords per C call).
4. **64-bit bit-plane slicing (`BitSlice2D`, for $\mathbb{F}_{2^m}, n \le 64$):** packs all $n$ coordinates into $m$ `uint64` bit-planes, batches **two** innermost coordinates ($q^2$ codewords at once), computes non-zero mask via `m-1` `uint64` bitwise-ORs and byte-LUT popcount! (Extends directly to $n \le 256$ via 4 `uint64` lanes.)

| Case (`add/add`) | Projective Count | Sage Baseline | `U8` Full (worst-case) | Speedup | `U8` Early (`d_design`) | Speedup | `BitSlice2D` Full | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `GF(16) k=4` ($d=9 > d_{\text{des}}=7$) | 4,369 | 134.0 ms | 5.865 ms | **22.8×** | 5.868 ms | 22.8× | **0.623 ms** | **215.1×** |
| `GF(64) k=3` ($d=62 = d_{\text{des}}$) | 4,161 | 195.7 ms | 0.732 ms | **267.4×** | **0.096 ms** | **2,035×** | 0.708 ms | **276.6×** |
| `GF(64) k=4` ($d=61 = d_{\text{des}}$) | 266,305 | 14,924 ms | 108.0 ms | **138.2×** | **0.109 ms** | **136,479×** | **55.03 ms** | **271.2×** |
| `GF(256) k=3` ($d=254 = d_{\text{des}}$) | 65,793 | 7,644 ms | **37.21 ms** | **205.4×** | **3.196 ms** | **2,392×** | *(needs 4×u64)* | — |

### 6.2 Max Feasible $(q, k)$ Within a 60-Second Budget (Worst-Case Full Scan, Single Core)

| Field $q$ | Current Sage (`~8–20k cw/s`) | NumPy `U8` / `BitSlice2D` (`~1.8–4.8M cw/s`) | Native C/Rust AVX2 Bit-Sliced + Gray (`~2×10^8 cw/s`) |
|---|---|---|---|
| `GF(16)` | $k \le 6$ (`1.12M` $\approx 40\text{ s}$) | $k \le 8$ (`286M` $\approx 60\text{ s}$ bit-sliced) | $k \le 9$ (`4.58B` $\approx 23\text{ s}$) |
| `GF(64)` | $k \le 4$ (`266k` $\approx 15\text{ s}$) | $k \le 5$ (`17.0M` $\approx 3.5\text{ s}$ bit-sliced) | $k \le 6$ (`1.09B` $\approx 5.5\text{ s}$) |
| `GF(256)` | $k \le 3$ (`65.8k` $\approx 7.6\text{ s}$) | $k \le 4$ (`16.8M` $\approx 9.5\text{ s}$ `uint8`) | $k \le 5$ (`4.31B` $\approx 22\text{ s}$) |
| `GF(1024)` | $k \le 2$ (`1,025` $\approx 0.4\text{ s}$) | $k \le 3$ (`1.05M` $\approx 2\text{ s}$ `uint16`) | $k \le 4$ (`1.07B` $\approx 10\text{ s}$) |

*(For $k > 5$ over `GF(256)`, projective enumeration is fundamentally $O(q^{k-1})$ and intractable; exact distance requires Brouwer–Zimmermann information-set enumeration or minimum-weight codeword ILP/meet-in-the-middle.)*

---

## 7. Encoder Paths, Batching, and Fast Generator-Matrix Construction

### 7.1 Single-Message & Batched Encoding (`GF(256), n=256, k=16`)

- **Message $\to$ polynomial $\to$ codeword (`enc_poly(m)` + evaluate at $n$ points):** `0.888 ms + 380.3 ms = 381.2 ms/msg` (**1,077× slower** than `enc.encode(m)` due to NTL `ZZ_pEX.__call__`).
- **Inherited `enc.encode(m)` (`vector(m) * G`):** `0.354 ms/msg` (`code.encode(m)`: `0.664 ms/msg`), bottlenecked by `Matrix._vector_times_matrix_` unpacking M4RIE rows.
- **Batched $B$ messages as a single matrix product:**

| Batch Size $B$ | Loop `enc.encode(m)` | M4RIE `M * G` (compute only) | Speedup vs Loop | M4RIE (pack + `M * G`) | NumPy Precomputed LUT | Speedup vs Loop |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.142 ms (`142 µs/m`) | 0.038 ms (`38.2 µs/m`) | 3.7× | 0.062 ms | **0.027 ms** (`26.7 µs/m`) | **5.3×** |
| 10 | 4.513 ms (`451 µs/m`) | 0.129 ms (`12.9 µs/m`) | 35.1× | 0.184 ms | **0.032 ms** (`3.20 µs/m`) | **141×** |
| 100 | 44.05 ms (`441 µs/m`) | 0.149 ms (`1.49 µs/m`) | 296× | 0.491 ms | **0.095 ms** (`0.95 µs/m`) | **466×** |
| 1,000 | 393.6 ms (`394 µs/m`) | **0.378 ms** (`0.38 µs/m`) | **1,040×** | 7.036 ms | 0.818 ms (`0.82 µs/m`) | **481×** |

### 7.2 Generator-Matrix Construction (`GF(256), n=256, k=16`, Uncached)

| Implementation | Median (ms) | `p10` / `p90` (ms) | Peak Memory | Speedup vs Baseline | Notes |
|---|---:|---:|---:|---:|---|
| Baseline Sage `[[poly(a)]]` (NTL `ZZ_pEX`) | 478.34 | 466.1 / 511.4 | 36.0 KiB | 1.0× | Marshals Givaro $\leftrightarrow$ NTL per $(i,j)$ |
| Sage `PolynomialRing(..., impl='generic')` | 11.49 | 9.56 / 17.48 | — | **41.6×** | 1-line change, zero new deps |
| Pure Python Horner over Givaro `poly.list()` | 6.582 | 4.27 / 10.30 | 35.6 KiB | **72.7×** | Extracts coeffs once per basis poly |
| NumPy $\mathbb{F}_{2^8}$ vectorized Horner | 4.122 | 1.05 / 6.57 | 17.8 KiB | **116×** | Pure NumPy log/antilog Horner |
| NumPy $\mathbb{F}_{2^8}$ log-table Vandermonde (+Sage coeff extract) | 0.302 | 0.292 / 4.36 | 58.7 KiB | **1,584×** | Precomputes $V_{d,j} = (d \log \alpha_j) \bmod (q-1)$ + XOR reduce |
| **NumPy $\mathbb{F}_{2^8}$ log-table Vandermonde (pure NumPy)** | **0.159** | **0.156 / 0.214** | **57.8 KiB** | **3,008×** | **Sub-millisecond even at $k=98$ (~1 ms vs 16.3 s = ~16,000×!)** |

---

## 8. Prioritized Action Plan (Analysis Only — Not Applied)

| Priority | Target | Concrete Change | Expected Gain | Sage-Free Readiness |
|---|---|---|---|---|
| **P0** | `TamoBargTwoSetsVectorEncoder.__init__` | Switch `PolynomialRing(F, 'x')` to `implementation='generic'` (short-term) or NumPy $\mathbb{F}_{2^m}$ coefficient matrices (Sage-free). Pass `basis='computed'` to `left_kernel_matrix()`. | **7×–15×** on `__init__` (`GF(256)`: `953 → 63 ms`; `GF(1024)`: `25.4 → 2.75 s`) | High (medium-effort RREF) |
| **P0** | `TamoBargTwoSetsVectorEncoder.generator_matrix` | Replace `poly(alpha)` with NumPy log-table Vandermonde (or Givaro Horner if staying in Sage). | **73×–3,008×** (`478 → 0.16 ms`; `k=98`: `16.3 s → ~1 ms`) | Trivial (~20 LOC NumPy) |
| **P1** | `TamoBargCodeTwoSets.minimum_distance_py` | Reverse `lead` order + early stop at `design_distance()` + NumPy `uint8` / 64-bit bit-sliced incremental XOR. | **215×–277×** worst-case; up to **136,000×** when tight | Trivial (~60 LOC NumPy) |
| **P1** | `TamoBargTwoSetsVectorEncoder` batch encode | Add `encode_batch(M)` using M4RIE `M * G` (or NumPy LUT) and use `unencode(..., nocheck=True)` after syndrome-verified decoding. | **466×–1,040×** batch encode; **74×** `unencode` | Easy (~15 LOC NumPy) |
| **P1** | `_safe_parity_check_matrix` | Compute coset Lagrange multipliers once per partition (via derivative/log-table) + Vandermonde row recurrence; make $0 \times m$ return immutable. | **~10×–100×** on decoder init for large $m$ | Easy (~25 LOC NumPy) |
| **P2** | `TamoBargCodeTwoSets.__init__` / accessors | Lazy `bipartite_graph()`, on-demand subgroup lookup, optional `dimension=None` $\to$ `max_dimension`, $O(1)$ `design_distance()`, cached `__hash__`. | Eliminates **2×** probe init in simulations; cleaner invariants | Trivial |
