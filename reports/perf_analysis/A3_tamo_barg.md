# A3 — Performance & Algorithm Analysis: `tamo_barg.py` (Single-Set Tamo–Barg LRC)

**Scope:** [`src/tamo_barg_lrc/tamo_barg.py`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg.py) — `_coerce_int`, `TamoBargCode`, `TamoBargVectorEncoder`, `TamoBargErasureDecoder`, `TamoBargErrorDecoder`, and inherited Sage `Encoder`/`Decoder`/`AbstractLinearCode` dispatch paths.

**Environment & Methodology:** Python 3.11.2 (`/home/antigravity/.venv-lrc/bin/python`, modular `passagemath` 10.8.x), Linux x86_64, 2 vCPUs (`os.cpu_count() == 2`). Fixed seeds (`random.seed(42)`, `np.random.seed(42)`, Sage `set_random_seed(42)`). Warm-up $\ge 3$ calls, $\ge 20$ repeats ($\ge 3$–$5$ for calls $> 1\text{ s}$); timings via `time.perf_counter()`, peak memory via `tracemalloc`, profiles via `cProfile`/`pstats` plus targeted sub-step micro-instrumentation (needed because Python `cProfile` attributes Cython C calls to their Python callers). All scratch scripts and raw JSON results are in `reports/perf_analysis/scratch/A3/`.

**Notation:** $q = |\mathbb{F}_q|$, block length $n$, dimension $k$, locality $r$, local minimum distance $\delta = d_{\text{local}} \ge 2$, coset size $m = r + \delta - 1$, partition count $s = n / m$, local error-correcting radius $t = \lfloor (\delta - 1)/2 \rfloor$, and $s' = \lceil k / r \rceil \le s$.

---

## 1. Module Design Narrative & Critique

`tamo_barg.py` implements the classic single-partition optimal $(n, k, r, \delta)$-LRC of Tamo & Barg (2014) on top of SageMath's `AbstractLinearCode` / `Encoder` / `Decoder` framework. Structurally the design is clean and well-factored: `TamoBargCode` owns only the finite field, validated integer parameters, chosen subgroup $H$ (`"mult"` or `"add"`), and coset partition $\{A_0, \dots, A_{s-1}\}$, deferring polynomial algebra (`TamoBargVectorEncoder`), generator matrix construction (`@cached_method`), and local decoders (`TamoBargErasureDecoder`, `TamoBargErrorDecoder`) to lazily constructed cached objects via `AbstractCode.encoder` / `AbstractCode.decoder`.

However, the implementation suffers from **six severe algorithmic and Sage-backend performance traps** that together cause **100×–1,000× slowdowns** at moderate field sizes ($q = 256, 1024$):

1. **Catastrophic $O(k n \cdot \deg)$ generator matrix build via NTL marshaling (`P0`):** `TamoBargVectorEncoder.generator_matrix()` evaluates each basis polynomial at each evaluation point (`[[poly(alpha) for alpha in alphas] for poly in self._enc_basis]`). Over binary extension fields $\mathbb{F}_{2^e}$ (`GF(16)`–`GF(4096)`), Sage represents polynomials with NTL `Polynomial_ZZ_pEX` (`sage/rings/polynomial/polynomial_zz_pex.pyx:285-287`), which converts every Givaro element through `list(a.polynomial()) -> ntl_ZZ_pE -> ZZ_pEX_eval -> ZZ_pE_c_to_list -> K(...)`. On `GF(256)` $[256, 112, 14]$, `generator_matrix()` takes **18.0 s** (vs **0.06 ms** on prime field `GF(13)` where FLINT `nmod_poly_evaluate_nmod` is used); at `GF(1024)` $k=480$ it is intractable ($\sim 1\text{ h}$). Exploiting the Khatri–Rao / Vandermonde identity $(x^i g(x)^j)(\alpha) = \alpha^i g(\alpha)^j$ with $g$ constant per coset reduces this to $O(k n)$ Givaro multiplications and yields **162×–352× measured speedups** (and $>1{,}000\times$ at $q=256, k=112$).
2. **Wasted annihilator construction in `TamoBargVectorEncoder.__init__` (`P1`):** Because $m = r + \delta - 1 \ge r + 1$, every encoding basis polynomial $x^i g(x)^j$ ($0 \le i < r, 0 \le j < s$) has degree $i + j m \le n - \delta < n = \deg(\text{annihilator})$ for all $s \ge 2$ (and for $s=1$, `algebra_basis = [1]`, so $g$ is unused). The degree-$n$ annihilator is **never triggered for modular reduction**, yet consumes **64%–67%** of `TamoBargVectorEncoder.__init__` time (`41.1 ms` of `63.8 ms` at $q=256$, and dominates the `5.56 s` init at $q=4096$). Moreover, `build_encoding_basis` materializes all $k_{\max} = s r$ basis polynomials even when $k \ll k_{\max}$.
3. **134× encoding cliff at $q \ge 512$ (`P0`):** `TamoBargVectorEncoder` inherits `Encoder.encode`, which computes `vector(word) * self.generator_matrix()`. Because `Matrix_gf2e_dense` does not override `_vector_times_matrix_` (`sage/matrix/matrix0.pyx:5312-5313`), Sage unpacks every matrix row into Python-object vectors and sums `v[i] * row_i`. Combined with Sage's Givaro element-cache cutoff `elem_cache = (order < 500)` (`sage/rings/finite_rings/finite_field_constructor.py:755`, `element_givaro.pyx:1760-1767`), `GF(1024)` heap-allocates tens of thousands of `FiniteField_givaroElement` objects per encode (`136.95 ms`, **5.0 MB** peak for `[1024, 30, 30]`), whereas a single M4RIE row-matrix multiply `matrix(F, 1, k, list(msg)) * G` takes **1.02 ms** (**134× faster**).
4. **85%–96% Sage framework overhead in `TamoBargErrorDecoder` (`P0`):** Every coset instantiates its own `GeneralizedReedSolomonCode`, `GRSBerlekampWelchDecoder`, and parity-check matrix ($s$ redundant copies). On error-free cosets, `H * local_vec` hits the same generic `matrix0.pyx:5346` column-unpacking loop (96% overhead vs batched M4RIE). On errored cosets (`GF(256), m=16`), `GRSBerlekampWelchDecoder` spends **71%** of its time in `S.right_kernel()` (`2.02 ms`), of which actual M4RIE `echelonize()` is only **`0.023 ms`** (1.1%) while Sage `VectorSpace` wrapping, `verbose()`/`time.process_time`, and `importlib` account for 98.9%.
5. **Unexploited affine equivalence across cosets (`P0`):** Every coset $A_c$ (additive $\beta_c + H$ or multiplicative $\beta_c H$) induces the **exact same** local Reed–Solomon code on $H$ up to an alignment permutation $\pi_c$ (and in $\mathbb{F}_{2^e}$ canonical additive cosets, $\pi_c$ is the identity!). Sharing **one** canonical local GRS code/decoder/PCM cuts `TamoBargErrorDecoder.__init__` time by **17×–38×** and memory by **8.5×–11.6×**, enables single-call batched M4RIE syndrome gating (**2×–5.8×** faster on 0-error words), and via a nonzero-coset shift unlocks `KeyEquationSyndrome` even for additive cosets (**2.4×–3.4×** faster error decoding).
6. **$O(m^2)$ Lagrange interpolation for local erasures (`P1`):** `TamoBargErasureDecoder` reconstructs a full degree-$(m - e_c - 1)$ polynomial per erased coset via Python `divided_difference` + NTL `Polynomial_ZZ_pEX`. Solving $H_{0, E} x_E = -H_{0, K} x_K$ with the shared local parity-check matrix runs in $O(e_c m + e_c^3)$ and gives **7.3×–476×** measured speedups (`480 ms -> 1.01 ms` at $q=1024$).

---

## 2. Mandatory Per-Function Analysis

All 32 functions/methods defined in `tamo_barg.py` (confirmed via `grep -n "^def \|^    def \|^class " src/tamo_barg_lrc/tamo_barg.py`) plus the hot inherited `Encoder.encode` entry point are analyzed below. Baseline measurements in the per-function table use `GF(256)` (`[256, 28, 14]`, $d_{\text{local}}=3, m=16, s=16$, `add`, and `[256, 112, 14]` for microbenchmarks) unless noted; full scaling sweeps are in Section 4.

### 2.1 Module Helper & `TamoBargCode`

| Function | Algorithm | Complexity (Time / Space) | Sage Primitives & Backend | Measured (median [p10, p90], peak mem, scaling) | Bottleneck | Correctness Risks | Better Algorithm | Sage Repl. (1–5) | Priority |
|---|---|---|---|---|---|---|---|---|---|
| `_coerce_int(value: Any, name: str) -> int` — Strict integer coercion rejecting `bool`/non-integer floats | `isinstance(bool)` guard, `int(value)`, equality check `int_val != value` | $O(1)$ / $O(1)$ | None (pure Python; accepts Sage `Integer` via `__int__`) | `0.12 µs` `[0.12, 0.13]`, $O(1)$ | Python call overhead | None; handles `bool`, `float`, `Sage Integer`, `np.int64` cleanly | Keep as-is | 1 (pure Python) | P2 |
| `TamoBargCode.__init__(self, base_field, length, dimension, locality, local_minimum_distance=2, sub_group_type="any")` — Validate params, pick subgroup, build coset partition, init `AbstractLinearCode` | Calls `list_multiplicative_subgroups` AND `list_canonical_additive_subgroups` unconditionally; selects subgroup matching $m = r+\delta-1$; enumerates all cosets via `list_subgroup_cosets`; slices first $s = n/m$; calls `super().__init__` | Time: $O(\sigma_1(q-1)\log q + q\log q)$ (`mult` builds all divisor subgroups + sorts; `add` builds all $p^i$ subspaces + sorts; `cosets` sorts all $q$ elements). Space: $O(q + \sigma_1(q-1))$ | `base_field.is_finite/is_field/cardinality`, `divisors` (FLINT/PARI), `multiplicative_generator`, `AbstractLinearCode.__init__` (`sage/coding/linear_code.py:346`) -> `VectorSpace` + `Parent.__init__` (pure Python/Cython category setup) | `q=16`: `0.09 ms` (4 KB); `q=256`: `0.75–1.80 ms` `[0.71, 4.79]` (22 KB); `q=1024`: `15.51 ms` (477 KB); `q=4096`: `78.61 ms` (2.1 MB). Slope: $n^{1.25}$–$n^{1.29}$ | Enumerating **both** subgroup families unconditionally + sorting all $q$ field elements (`70%` math, `30%` `super().__init__` at $q=256$) | Only checks *canonical* additive subgroups (`span(1, ..., a^(i-1))`), fine for single-set; rejects `q` prime additive subgroup only if $m \nmid q$ (correctly) | Build **only** the requested subgroup of exact size $m$ directly ($O(m)$ mult via $g^{(q-1)/m}$, $O(m)$ add via bitmask $0..p^t-1$) + generate only $s$ cosets directly ($O(n)$ vs $O(q \log q)$) | 2 (GF arith + `int` bitmask/generator) | P2 |
| `TamoBargCode.__eq__(self, other) -> bool` — Structural equality | Field-by-field comparison ending with `evaluation_points() == other.evaluation_points()` | $O(n)$ worst-case (equal codes) / $O(1)$ | `FiniteField_givaroElement.__eq__` (Cython int compare) | `1.64 µs` `[1.56, 1.82]` ($n=256$); $O(n)$ | Elementwise list compare of $n$ points | Safe; compares `evaluation_points` not `_sub_group_type` (two identical codes with different internal labels compare equal) | Early-out on cached `_hash` or compare `(sub_group, sub_group_type, partition_size)` in $O(m)$ | 1 (pure Python) | P2 |
| `TamoBargCode.__hash__(self) -> int` — Hash consistent with `__eq__` | Converts `self._evaluation_points` to a fresh `tuple` and hashes 6-tuple | $O(n)$ time / $O(n)$ temp tuple per call | `FiniteField_givaroElement.__hash__` × $n$ | `57.70 µs` `[14.32, 81.19]` ($n=256$); $O(n)$ | Re-allocating `tuple` + hashing $n$ Givaro elements every call | Consistent with `__eq__`; slow if code used as dict/cache key | Cache `self._cached_hash` once (code is immutable after `__init__`) -> `~0.1 µs` (500× faster) | 1 (pure Python) | P2 |
| `TamoBargCode._repr_(self) -> str` — Sage string representation | f-string formatting `length`, `dimension`, `locality`, `base_field.cardinality()` | $O(1)$ / $O(1)$ | `base_field.cardinality()` | `2.20 µs` `[2.14, 28.33]` | `cardinality()` Sage `Integer` conversion | Note omits `d_local` in repr (informational only) | Keep as-is | 1 (pure Python) | P2 |
| `TamoBargCode.partition(self)` — Return `(coset_partition, partition_size)` | Attribute lookup | $O(1)$ / $O(1)$ | None | `0.13 µs` `[0.13, 0.13]` | Negligible | Returns mutable internal list `self._partition` (caller could accidentally mutate!) | Return tuple-of-tuples or document read-only | 1 | P2 |
| `TamoBargCode.sub_group(self)` — Return `(subgroup_elements, sub_group_type)` | Attribute lookup | $O(1)$ / $O(1)$ | None | `0.12 µs` `[0.12, 0.15]` | Negligible | Returns mutable internal list `self._sub_group` | Return tuple or document read-only | 1 | P2 |
| `TamoBargCode.evaluation_points(self)` — Return flattened `n` evaluation points | Attribute lookup | $O(1)$ / $O(1)$ | None | `0.10 µs` `[0.10, 0.11]` | Negligible | Returns mutable internal list `self._evaluation_points` | Return tuple or document read-only | 1 | P2 |
| `TamoBargCode.locality(self) -> int` — Return locality `r` | Attribute lookup | $O(1)$ / $O(1)$ | None | `0.10 µs` `[0.10, 0.10]` | Negligible | None | Keep as-is | 1 | P2 |
| `TamoBargCode.local_minimum_distance(self) -> int` — Return $\delta$ | Attribute lookup | $O(1)$ / $O(1)$ | None | `0.10 µs` `[0.10, 0.12]` | Negligible | None | Keep as-is | 1 | P2 |
| `TamoBargCode.max_dimension(self) -> int` — Return $s \cdot r$ | Integer multiply | $O(1)$ / $O(1)$ | None | `0.12 µs` `[0.12, 0.21]` | Negligible | None | Keep as-is | 1 | P2 |
| `TamoBargCode.minimum_distance(self) -> int` — Closed-form Singleton-like optimal LRC distance | $n - k + 1 - (\lceil k/r \rceil - 1)(\delta - 1)$ | $O(1)$ / $O(1)$ | `self.length()` (Sage `Integer` -> `int`) | `0.31 µs` `[0.29, 2.69]` | Negligible | Exact for all valid $(n,k,r,\delta)$ (proven in §3.2); **API risk:** overrides `AbstractLinearCode.minimum_distance(self, algorithm=None)` (`linear_code.py:1336`) without accepting `algorithm=None`, so `code.minimum_distance(algorithm=...)` raises `TypeError` | Accept optional `algorithm=None` (ignore or delegate if explicit algorithm requested) | 1 (pure Python) | P2 |

### 2.2 `TamoBargVectorEncoder` (plus Inherited `Encoder.encode`)

| Function | Algorithm | Complexity (Time / Space) | Sage Primitives & Backend | Measured (median [p10, p90], peak mem, scaling) | Bottleneck | Correctness Risks | Better Algorithm | Sage Repl. (1–5) | Priority |
|---|---|---|---|---|---|---|---|---|---|
| `TamoBargVectorEncoder.__init__(self, code)` — Build annihilator, good poly, algebra basis, full encoding basis, slice `[:k]` | Product-tree annihilator $\prod (x-\alpha)$; good poly $x^m$ or $\prod_{h\in H}(x-h)$; powers $g^0..g^{s-1}$; shifts $x^i g^j$ for all $s r = k_{\max}$; sort by degree; slice `[:k]` | Time: $O(n \log^2 n)$ (annihilator) $+ O(s \cdot n \log n)$ (NTL `ZZ_pEX` mults for $g^j$) $+ O(k_{\max} \log k_{\max})$ sort; space $O(k_{\max} \cdot n)$ poly coeffs | `PolynomialRing(F, 'x')` -> NTL `Polynomial_ZZ_pEX` (`GF(2^e)`) or FLINT `Polynomial_zmod_flint` (`GF(p)`); `prod` (`sage.misc.misc_c`); `p.shift(i)` | `q=16`: `0.55–1.82 ms` (4 KB); `q=64`: `18.6 ms` (8 KB); `q=256`: `47.4–83.9 ms` (29 KB); `q=1024`: `448–580 ms` (146 KB); `q=4096`: `5,561 ms` (617 KB). Slope: $n^{1.57}$ ($k \approx n/2$) | **Unused** `build_annihilator_polynomial` = **64.5%–67.4%** of init (`41.1 ms` of `63.8 ms` at $q=256$); plus computing all $s r$ polys when only $k$ needed | For $s=1$, `good_poly % annihilator` yields `0` (harmless since `algebra_basis=[1]`, though `enc_good_poly()` returns `0` instead of $\prod_{h\in H}(x-h)$!) | Drop annihilator (never needed for $s \ge 2$); compute only $s' = \lceil k/r \rceil$ powers of $g$ lazily (or skip polynomial basis altogether if only `generator_matrix`/`encode` used) -> $O(k_{\text{used}})$ (~5×–10× faster init) | 2 (NumPy/Galois poly array) | P1 |
| `TamoBargVectorEncoder.__eq__(self, other)` | `isinstance` + `self.code() == other.code()` | $O(n)$ / $O(1)$ | Delegates to `TamoBargCode.__eq__` | `1.81 µs` `[1.72, 2.00]` ($n=256$) | Code equality compare | Safe | Benefits automatically from code `__eq__` speedup | 1 | P2 |
| `TamoBargVectorEncoder.__hash__(self)` | `hash((type(self), self.code()))` | $O(n)$ / $O(n)$ | Delegates to `TamoBargCode.__hash__` | `51.56 µs` `[22.23, 63.05]` ($n=256$) | Uncached `TamoBargCode.__hash__` | Safe | Cache `TamoBargCode.__hash__` | 1 | P2 |
| `TamoBargVectorEncoder._repr_(self)` | f-string wrapping `self.code()` | $O(1)$ / $O(1)$ | `TamoBargCode._repr_` | `3.06 µs` `[2.70, 29.52]` | String formatting | Safe | Keep as-is | 1 | P2 |
| `TamoBargVectorEncoder.enc_basis(self)` | Attribute lookup (`_enc_basis`) | $O(1)$ / $O(1)$ | None | `0.08 µs` `[0.07, 0.08]` | Negligible | Returns mutable internal list | Return tuple or copy | 1 | P2 |
| `TamoBargVectorEncoder.enc_algebra_basis(self)` | Attribute lookup (`_algebra_basis`) | $O(1)$ / $O(1)$ | None | `0.08 µs` `[0.07, 0.08]` | Negligible | Returns mutable internal list | Return tuple or copy | 1 | P2 |
| `TamoBargVectorEncoder.enc_good_poly(self)` | Attribute lookup (`_good_poly`) | $O(1)$ / $O(1)$ | None | `0.08 µs` `[0.07, 0.08]` | Negligible | Returns `0` when $s=1$ (`n == m`, `add`) due to `% annihilator` | Omit `% annihilator` in single-set encoder | 1 | P2 |
| `TamoBargVectorEncoder.max_dimension(self)` | Attribute lookup (`_max_dimension`) | $O(1)$ / $O(1)$ | None | `0.08 µs` `[0.08, 0.08]` | Negligible | None | Keep as-is | 1 | P2 |
| `TamoBargVectorEncoder.enc_poly(self, message)` — Construct message polynomial $\sum_{i=0}^{k-1} m_i b_i(x)$ | Linear combination via `evaluate_message_polynomial`: `sum((F(m[i]) * b[i]), 0)` in `GF(q)[x]` | Time: $O(k \cdot D_{\max}) = O(k \cdot \frac{k}{r} m)$ NTL poly scalar-mult + add (with $O(k)$ intermediate poly allocations in Python `sum`); Space: $O(D_{\max})$ | NTL `Polynomial_ZZ_pEX` (`__mul__`, `__add__`) + `base_ring(m[i])` coercion | `q=16,k=4`: `0.08 ms`; `q=64,k=24`: `0.78 ms`; `q=256,k=28`: `4.98 ms` `[1.02, 7.03]`; `q=256,k=112`: `21.77 ms` `[17.47, 29.72]`; `q=1024,k=60`: `9.60 ms` (3 KB). Slope ($k=2r$): $n^{1.24}$ | Repeated NTL `ZZ_pEX` temporary allocations + Givaro->ZZ_pE scalar conversion inside Python `sum` | Validates `len(message) == k`; safe | Accumulate in flat coefficient list/array of length $D_{\max}+1$ using Horner in $g(x)$: $f(x) = \sum_{j=0}^{s'-1} (\sum_{i<r} m_{j,i} x^i) g(x)^j$ ($s'$ poly mults instead of $k$ full-degree adds) | 2 (NumPy poly Horner) | P1 |
| `TamoBargVectorEncoder.generator_matrix(self)` (`@cached_method`) — Build immutable $k \times n$ generator matrix | Evaluates every basis poly at every evaluation point: `[[poly(alpha) for alpha in alphas] for poly in self._enc_basis]` | Time: $O(k \cdot n \cdot D_{\text{avg}}) = O(k^2 n \frac{m}{r})$ ($\Theta(n^3)$ for $k \propto n$!) + $k n$ NTL↔Givaro marshals; Space: $O(k n)$ Python list + M4RIE matrix | `@cached_method` (`sage/misc/cachefunc.pyx`); `Polynomial_ZZ_pEX.__call__` (`polynomial_zz_pex.pyx:285-287`, NTL `ZZ_pEX_eval`); `matrix(...)` -> `Matrix_gf2e_dense` (M4RIE) / `Matrix_modn_dense_float` (LinBox) | **Cached:** `0.06 µs`. **Uncached:** `q=13,k=8`: `0.06 ms` (FLINT); `q=16,k=4`: `5.34 ms`; `q=64,k=24`: `269 ms`; `q=256,k=28`: `1,646 ms` (61 KB); **`q=256,k=112`: `18,042 ms` (18.0 s!)**; **`q=1024,k=60`: `26,646 ms` (26.6 s, 4.8 MB)**. Slope: $n^{2.04}$ ($k=2r$), $n^{3.03}$ ($k \approx n/2$, 64→256) | **99.6%** in `[[poly(alpha)...]]`: `Polynomial_ZZ_pEX.__call__` marshals Givaro→`list(a.polynomial())`→`ntl_ZZ_pE`→`ZZ_pEX_eval`→`ZZ_pE_c_to_list`→Givaro $k \cdot n$ times | Immutable (`set_immutable()`); safe | **Khatri–Rao / Vandermonde build:** $G[jr+i, cm+l] = \alpha_{c,l}^i \cdot g_c^j$ using $s$ evals of $g$, $n r$ powers of $\alpha$, $k n$ Givaro mults ($O(k n)$ total, 0 NTL evals). **Measured 162×–352× faster** ($>1{,}000\times$ at $q=256, k=112$)! | 2 (NumPy GF LUT / Vandermonde) | **P0** |
| *(Inherited)* `Encoder.encode(self, word)` / `TamoBargCode.encode(self, word)` — Encode message vector to codeword | `M = message_space(); if word not in M: raise; return vector(word) * self.generator_matrix()` (`sage/coding/encoder.py:119`) | Time: $O(k n)$ generic row-by-row Python-object vector sum; Space: $O(k n)$ temp objects when $q \ge 512$! | `Matrix._vector_times_matrix_` (`sage/matrix/matrix0.pyx:5312-5313`), NOT M4RIE! `FreeModuleElement_generic_dense` + `Cache_givaro` (`elem_cache = order < 500`) | `q=16,k=4`: `0.011 ms`; `q=64,k=24`: `0.05 ms`; `q=256,k=28`: `0.31 ms` (67 KB); `q=256,k=112`: `3.16 ms` (252 KB); **`q=1024,k=60`: `152.5 ms` (5.0 MB!!)**; `q=1024,k=28`: `77.1 ms` (2.4 MB). Slope: $n^{2.23}$ (cliff at $q \ge 512$!) | `Matrix_gf2e_dense` lacks `_vector_times_matrix_`, hitting Python row-unpacking + Givaro heap allocation cliff for $q \ge 512$ (**79.5%–99.3%** non-kernel overhead!) | Calls `message_space()` & membership check every call; safe | Override `encode` to use M4RIE `(matrix(F, 1, k, list(word)) * G).row(0)` (**2.4×–134× faster** in Sage), or 2-stage factorized $O(k(s+m))$ / additive-FFT $O(n \log n)$ in NumPy/C | 2 (GF matmul / FFT) | **P0** |

### 2.3 `TamoBargErasureDecoder`

| Function | Algorithm | Complexity (Time / Space) | Sage Primitives & Backend | Measured (median [p10, p90], peak mem, scaling) | Bottleneck | Correctness Risks | Better Algorithm | Sage Repl. (1–5) | Priority |
|---|---|---|---|---|---|---|---|---|---|
| `TamoBargErasureDecoder.__init__(self, code)` — Build input space `GF(q)^n x GF(2)^n` and poly ring | Constructs `cartesian_product([ambient_space, GF(2)^n])` and caches `PolynomialRing(F, 'x')`, `m` | $O(1)$ (lazy spaces) / $O(1)$ | `cartesian_product`, `VectorSpace(GF(2), n)`, `Decoder.__init__`, `PolynomialRing` | `0.022–0.027 ms` across $q \in [16, 4096]$ (2 KB); slope $n^{0.0}$ | Constant Sage `Parent` lookup | None | Keep lightweight; optionally precompute shared canonical $H_0$ (`<0.7 ms`) for fast erasure solve | 1 (pure Python) | P2 |
| `TamoBargErasureDecoder.__eq__(self, other)` | `isinstance` + `self.code() == other.code()` | $O(n)$ / $O(1)$ | `TamoBargCode.__eq__` | `1.83 µs` `[1.76, 21.17]` ($n=256$) | Code equality | Safe | Benefits from code `__eq__` | 1 | P2 |
| `TamoBargErasureDecoder.__hash__(self)` | `hash((type(self), self.code()))` | $O(n)$ / $O(n)$ | `TamoBargCode.__hash__` | `55.56 µs` `[38.98, 72.53]` ($n=256$) | Uncached code hash | Safe | Cache code hash | 1 | P2 |
| `TamoBargErasureDecoder._repr_(self)` | f-string wrapping `self.code()` | $O(1)$ / $O(1)$ | `TamoBargCode._repr_` | `3.22 µs` `[2.81, 22.66]` | String formatting | Safe | Keep as-is | 1 | P2 |
| `TamoBargErasureDecoder.decode_to_code(self, word_and_erasure_vector)` — Recover $\le \delta-1$ erasures per coset | Loops over $s$ cosets: coerces `word` slice to `F` & `era` slice to `int`; counts erasures $e_c$; if $e_c \ge \delta$ raises `DecodingError`; if $e_c > 0$ calls `interpolate_erasures` -> `poly_ring.lagrange_polynomial(known_pts)` (`divided_difference` in pure Python + Horner poly build in NTL `ZZ_pEX`), evaluates at $e_c$ erased points | Time: $O(n) + \sum_{c: e_c > 0} O((m - e_c)^2 + e_c(m - e_c)) = O(n + E_{\text{cosets}} \cdot m^2)$ (note: for $e_c \le \delta - 1$, $m - e_c \ge r$, so $\Theta(m^2)$ per erased coset, NOT $\Theta(e_c r^2)$!); Space: $O(n + m^2)$ (divided-difference triangle) | `PolynomialRing.lagrange_polynomial` (`polynomial_ring.py:2361`), `divided_difference` (`polynomial_ring.py:2272`, pure Python $O(m^2)$), NTL `Polynomial_ZZ_pEX` mul/add/eval, `vector(F, ...)` | `q=16`: `0.06 / 0.17 / 0.56 ms` (0 / 1 / all cosets); `q=64`: `0.09 / 0.46 / 11.5 ms`; `q=256,m=16`: `0.18 / 5.48 / 70.7 ms` `[66.5, 91.1]` (9 KB); `q=1024,m=32`: `0.57 / 13.1 / 452.6 ms` `[438, 513]` (55 KB). Slopes ($k=2r$): 0-era $n^{0.53}$, 1-coset $n^{1.12}$ (theor. $m^2 = n^{1.0}$), all-cosets $n^{1.58}$ (theor. $s m^2 = n^{1.5}$) | `lagrange_polynomial` + NTL `_element_constructor_` = **82%–90%** at $q=256$; for 0/1 erased coset on $q=16$, Python slicing & coercion is **60%** | **Subtlety:** accepts *any* $m - e_c \ge r$ points, so if $e_c < \delta - 1$ it interpolates a degree-$(m-e_c-1)$ poly instead of degree-$(r-1)$; if an *unmarked* symbol has an error, it silently interpolates garbage without checking consistency against the $\delta - 1 - e_c$ redundant symbols! Also `erasure_vector` checks `int(b) != 0` (safe for `GF(2)` and ints) | **Shared canonical parity-check erasure solve:** $H_{0, E} x_E = -H_{0, K} x_K$ in $O(e_c m + e_c^3)$ (direct scalar divide for $e_c=1$, $2\times 2$ Cramer for $e_c=2$); or barycentric weights on canonical $H$. **Measured 7.3×–476× faster** (`480 ms -> 1.01 ms` at $q=1024$)! | 2 (GF LUT + small linear solve) | **P0** |

### 2.4 `TamoBargErrorDecoder`

| Function | Algorithm | Complexity (Time / Space) | Sage Primitives & Backend | Measured (median [p10, p90], peak mem, scaling) | Bottleneck | Correctness Risks | Better Algorithm | Sage Repl. (1–5) | Priority |
|---|---|---|---|---|---|---|---|---|---|
| `TamoBargErrorDecoder.__init__(self, code)` — Precompute $s$ local GRS codes, Berlekamp–Welch decoders, and PCMs | For each of $s$ cosets: builds `GeneralizedReedSolomonCode(coset, r)`, `grs.decoder("BerlekampWelch")`, `grs.parity_check_matrix()` (which constructs `dual_code()` via $O(m^2)$ `multipliers_product()` + Vandermonde `generator_matrix()` via Python `lambda`) | Time: $O(s \cdot m^2) = O(n m)$ math + $O(s)$ heavy Sage `Parent`/Category object setups; Space: $O(s \cdot m \delta)$ PCMs + $O(s)$ GRS + dual GRS + decoder objects | `GeneralizedReedSolomonCode` (`grs_code.py:145`), `dual_code` (`:458`), `multipliers_product` (`:396`, $O(m^2)$), `GRSEvaluationVectorEncoder.generator_matrix` (`:760`), `GRSBerlekampWelchDecoder` (`:1117`), `Matrix_gf2e_dense` | `q=16`: `6.09–12.6 ms` (27 KB); `q=64`: `14.2–29.3 ms` (58–106 KB); `q=256,m=16`: `31.8–32.7 ms` `[26.3, 42.3]` (129 KB); `q=1024,m=32`: `97.2–107.5 ms` (415 KB); `q=4096,m=64`: `336.6 ms` (1.21 MB). Slope: $n^{0.52}$–$n^{0.71}$ (dominated by $O(s)=O(n^{0.5})$ Sage object creation up to $m=64$) | Redundantly building $s$ separate affine-equivalent `GRS` + dual `GRS` + `Decoder` objects | **Note:** when $d_{\text{local}} = 2$, $t = \lfloor 1/2 \rfloor = 0$ (can only detect, not correct errors; still builds $s$ Berlekamp–Welch decoders that can never correct a nonzero error!) | **Share ONE canonical local GRS code/decoder/PCM** via affine coset equivalence + $O(m)$ alignment permutation per coset (identity for binary `add`!). **Measured 17×–38× faster init, 8.5×–11.6× less memory!** | 3 (RS syndrome + BM/Euclid/Forney) | **P0** |
| `TamoBargErrorDecoder.__eq__(self, other)` | `isinstance` + `self.code() == other.code()` | $O(n)$ / $O(1)$ | `TamoBargCode.__eq__` | `1.81 µs` `[1.74, 8.83]` ($n=256$) | Code equality | Safe | Benefits from code `__eq__` | 1 | P2 |
| `TamoBargErrorDecoder.__hash__(self)` | `hash((type(self), self.code()))` | $O(n)$ / $O(n)$ | `TamoBargCode.__hash__` | `64.82 µs` `[19.78, 115.6]` ($n=256$) | Uncached code hash | Safe | Cache code hash | 1 | P2 |
| `TamoBargErrorDecoder._repr_(self)` | f-string wrapping `self.code()` | $O(1)$ / $O(1)$ | `TamoBargCode._repr_` | `2.97 µs` `[2.79, 71.55]` | String formatting | Safe | Keep as-is | 1 | P2 |
| `TamoBargErrorDecoder.decode_to_code(self, r)` — Local coset syndrome gate + Berlekamp–Welch error decoding | Copies input `copy(vector(F, r))`; for each coset $i=0..s-1$: slices `local_vec = vector(F, r_vec[...])`, tests `(H_i * local_vec).is_zero()`; if nonzero calls `GRSBerlekampWelchDecoder.decode_to_code(local_vec)` ($m \times (m+1)$ linear system `S.right_kernel()`, polynomial `quo_rem`, NTL re-encode); catches `(DecodingError, ValueError)` and keeps original slice | Time: $O(n \delta)$ syndrome checks $+ E_{\text{cosets}} \cdot O(m^3)$ (`right_kernel` Gaussian elim; M4RIE PLE/Newton–John); Space: $O(n + m^2)$ | `copy`, `vector`, `Matrix._matrix_times_vector_` (`matrix0.pyx:5346`, generic column unpacking!), `GRSBerlekampWelchDecoder._decode_to_code_and_message` (`grs_code.py:1186`): redundant `r in C`, `matrix(..., lambda)`, `S.right_kernel()` (M4RIE + `VectorSpace` + `verbose`/`process_time`), NTL `quo_rem`, `GRSEvaluationPolynomialEncoder.encode` (`grs_code.py:1014`, $m$ NTL evals) | `q=16`: `0.12 / 3.44 / 12.4 ms` (0 / 1 / all cosets); `q=64`: `0.20 / 5.01 / 35.7 ms`; `q=256,m=16`: `0.43 / 7.15 / 129.8 ms` `[126, 163]` (49 KB); `q=1024,m=32`: `3.48 / 19.5 / 827.6 ms` `[811, 1026]` (144 KB). Slopes ($k=2r$): 0-err $n^{0.79}$, 1-err $n^{0.40}$, all-err $n^{1.00}$ (framework-dominated!) | **85%–96% Sage framework overhead!** Per errored coset (`q=256`): `right_kernel()` = `2.02 ms` (**98.9%** `VectorSpace`+`process_time`+`importlib`, only `0.023 ms` M4RIE `echelonize`!); NTL re-encode = `0.43 ms`. On 0-err: `matrix0.pyx:5346` + `vector()` slicing = **96.1%** overhead | (1) Silently swallows `DecodingError` and returns un-decoded slice (best-effort local repair, so returned vector may NOT be in the global code!). (2) Redundant double copy `copy(vector(base_field, r))` (`vector()` already allocates a new vector). (3) `GRSBerlekampWelchDecoder` redundantly checks `r in C` after syndrome already proved `r not in C` | (1) Single batched M4RIE syndrome check `H0 * R_mat` (**2×–5.8×** faster on 0-err); (2) Shared canonical local decoder using **`KeyEquationSyndrome`** (via nonzero-coset shift for `add`, $O(m \delta + \delta^2)$ -> **2.4×–3.4×** faster in Sage) or pure C/NumPy Berlekamp–Massey + Chien + Forney (**50×–200×** faster, zero Sage framework overhead!) | 3 (syndrome + BM/Euclid + Chien/Forney in ~100 LOC NumPy/C) | **P0** |

---

## 3. Deep-Dive Answers to Specific Questions

### 3.1 `TamoBargCode.__init__`: Precomputed vs Lazily Cached (`@cached_method`) Cost Split

In `tamo_barg.py`, **`TamoBargCode.__init__` does NOT precompute any polynomials, matrices, or decoders**:
- **Precomputed eagerly in `TamoBargCode.__init__`:** parameter validation, BOTH `list_multiplicative_subgroups(F)` and `list_canonical_additive_subgroups(F)`, full coset enumeration `list_subgroup_cosets`, slicing `_partition` and `_evaluation_points`, and `AbstractLinearCode.__init__`.
- **Lazily constructed & cached via `@cached_method` on `AbstractCode` (`sage/coding/abstract_code.py:743, 906`):**
  - `code.encoder("VectorEncoder")` → constructs `TamoBargVectorEncoder` (builds annihilator, good poly, algebra basis, full encoding basis).
  - `enc.generator_matrix()` → `@cached_method` on `TamoBargVectorEncoder` (builds immutable $k \times n$ matrix; cache hit takes **`0.06 µs`** vs up to **`26.6 s`** uncached!).
  - `code.decoder("ErasureDecoder")` → constructs `TamoBargErasureDecoder` (`~0.023 ms`).
  - `code.decoder("ErrorDecoder")` → constructs `TamoBargErrorDecoder` (eagerly builds all $s$ local GRS codes + decoders + PCMs).

**Measured cost split across initialization stages (median ms, `add` subgroup, $d_{\text{local}}=3$):**

| Config $(q, n, k, r, m, s)$ | `TamoBargCode.__init__` (eager) | `VectorEncoder.__init__` (lazy) | `generator_matrix()` uncached (lazy) | `ErasureDecoder.__init__` (lazy) | `ErrorDecoder.__init__` (lazy) | Share of full cold setup in `generator_matrix()` |
|---|---:|---:|---:|---:|---:|---:|
| `(16, 16, 4, 2, 4, 4)` | `0.09 ms` | `1.82 ms` | `5.34 ms` | `0.023 ms` | `12.61 ms` | 26.9% |
| `(64, 64, 24, 6, 8, 8)` | `0.22 ms` | `19.62 ms` | `269.09 ms` | `0.023 ms` | `16.00 ms` | 88.2% |
| `(256, 256, 28, 14, 16, 16)` | `1.56 ms` | `47.42 ms` | `1,646.36 ms` | `0.022 ms` | `32.72 ms` | **95.3%** |
| `(256, 256, 112, 14, 16, 16)` | `1.80 ms` | `83.88 ms` | **`18,042.50 ms`** | `0.022 ms` | `31.79 ms` | **99.4%** |
| `(1024, 1024, 60, 30, 32, 32)` | `15.51 ms` | `448.33 ms` | **`26,645.79 ms`** | `0.025 ms` | `107.53 ms` | **97.9%** |
| `(4096, 4096, 2000, 62, 64, 64)` | `78.61 ms` | `5,561.22 ms` | *intractable ($\sim 1\text{ h}+$)* | `0.023 ms` | `336.59 ms` | $>99.9\%$ |

Inside `TamoBargCode.__init__` at `GF(256)` (`0.88 ms` total): `list_multiplicative_subgroups` = `0.15 ms` (17%, wasted when `sub_group_type="add"`!), `list_canonical_additive_subgroups` = `0.25 ms` (29%), `list_subgroup_cosets` = `0.21 ms` (24%), `AbstractLinearCode.__init__` + validation = `0.27 ms` (30%). Inside `TamoBargVectorEncoder.__init__` at `GF(256)` (`63.8 ms`): unused `build_annihilator_polynomial` = **`41.1 ms` (64.5%)**, `good_poly` = `0.76 ms` (1.2%), `algebra_basis` = `6.83 ms` (10.7%), `encoding_basis` = `5.87 ms` (9.2%), Sage wrapper = `9.20 ms` (14.4%).

### 3.2 `minimum_distance`: Closed Form vs Search & Validity Conditions

`TamoBargCode.minimum_distance()` returns the closed-form Singleton-like LRC bound in $O(1)$ (`0.31 µs`) instead of $O(q^k)$ codeword weight search:
$$d = n - k + 1 - \left(\left\lceil \frac{k}{r} \right\rceil - 1\right)(\delta - 1).$$

**Validity proof for all parameter combinations accepted by `__init__`:**
- Let $m = r + \delta - 1$. Since $\delta \ge 2$, $m \ge r + 1 > r$, and $n = s m$ distinct evaluation points exist in $\mathbb{F}_q$.
- For $s \ge 2$, the basis polynomials $x^i g(x)^j$ ($0 \le i < r, 0 \le j < s$) have strictly distinct degrees $i + j m \le (r - 1) + (s - 1) m = n - \delta < n$ (never reduced mod annihilator). Sorting by ascending degree places index $b = j r + i$ at position $b$. For $s = 1$, $k \le r$ and basis is $\{1, x, \dots, x^{k-1}\}$ (standard MDS RS code).
- Taking the first $k$ basis polynomials gives maximum degree $D_{\max} = \deg(p_{k-1}) = q_0 m + r_0$ where $k - 1 = q_0 r + r_0$, $q_0 = \lceil k/r \rceil - 1$, $0 \le r_0 < r$. Expanding $m = r + \delta - 1$ gives $D_{\max} = (k - 1) + (\lceil k/r \rceil - 1)(\delta - 1)$.
- Any nonzero $f \in \text{span}(p_0, \dots, p_{k-1})$ has at most $D_{\max}$ roots among the $n$ distinct points, so $d_{\text{actual}} \ge n - D_{\max} = n - k + 1 - (\lceil k/r \rceil - 1)(\delta - 1)$.
- Conversely, every coordinate lies in a disjoint local $(m, r, \delta)$ RS repair group, so the Prakash–Kamath–Lalitha–Kumar (2012) Singleton-like upper bound holds **even when $r \nmid k$**: $d_{\text{actual}} \le n - k + 1 - (\lceil k/r \rceil - 1)(\delta - 1)$.
- Therefore, equality holds unconditionally for **every** code accepted by `TamoBargCode.__init__`. Verified computationally against exhaustive $q^k - 1$ codeword enumeration on 7 edge-case codes (covering $r \mid k$, $r \nmid k$, `add`, `mult`, $\delta \in \{2, 3\}$, and $s = 1$): **7/7 exact matches** (`construction_results.json`).
- **Minor API caveat:** overrides `AbstractLinearCode.minimum_distance(self, algorithm=None)` without accepting `algorithm=None`.

### 3.3 `enc_poly`, `generator_matrix`, and Vandermonde / Khatri–Rao Structure

In `TamoBargVectorEncoder.generator_matrix()` (`tamo_barg.py:326-337`), each of the $k$ basis polynomials $p_b(x)$ is evaluated independently at all $n$ points via `poly(alpha)`. Over $\mathbb{F}_{2^e}$, `Polynomial_ZZ_pEX.__call__` (`sage/rings/polynomial/polynomial_zz_pex.pyx:285-287`) invokes NTL `ZZ_pEX_eval` (Horner's method in $O(\deg(p_b))$ `ZZ_pE` ops) **plus** a Givaro $\leftrightarrow$ Python `list` $\leftrightarrow$ `ntl_ZZ_pE` conversion on **every single `(poly, alpha)` pair**. Total cost is $O(k \cdot n \cdot D_{\text{avg}}) = O(k^2 n \cdot m/r)$, i.e., $\Theta(n^3)$ when $k = \Theta(n)$ (measured slope **$n^{3.03}$** between $n=64$ and $n=256$!).

**Vandermonde / Khatri–Rao structure:** Because $p_{j r + i}(\alpha) = \alpha^i \cdot g(\alpha)^j$ on all evaluation points and $g(\alpha) \equiv \gamma_c$ is **constant on every coset $A_c$**:
1. Compute $\gamma_c = g(A_c[0])$ once per coset ($s$ evals; for `mult`, just $A_c[0]^m$).
2. Precompute $\gamma_c^j$ for $0 \le j < s'$ ($s' s$ field mults) and $\alpha^i$ for $0 \le i < r$ ($n r$ field mults).
3. Assemble $G[j r + i, c m + l] = \alpha_{c,l}^i \cdot \gamma_c^j$ ($k n$ Givaro mults, **zero NTL evaluations**).

**Prototype benchmark (`bench_optimizations.py`, bit-exact `G_fast == G_ref` verified on all 6 configs):**

| Config | Current `generator_matrix()` | Vandermonde / Khatri–Rao | Speedup |
|---|---:|---:|---:|
| `q16_add_m4_k4` | `3.31 ms` | `0.13 ms` | **25.4×** |
| `q64_add_m8_k24` | `167.38 ms` | `0.75 ms` | **223.6×** |
| `q256_add_m16_k28` | `1,153.93 ms` | `6.73 ms` | **171.5×** |
| `q256_mult_m15_k52` | `2,588.21 ms` | `7.36 ms` | **351.7×** |
| `q256_add_m16_k48_d5` | `3,184.66 ms` | `11.78 ms` | **270.3×** |
| `q1024_add_m32_k60` | `23,712.88 ms` | `146.12 ms` | **162.3×** |

*(Note: on `q256_add_m16_k112`, current takes `18,042 ms` while Vandermonde scales linearly in $k$ to $\sim 16\text{ ms}$ — a **$>1{,}000\times$** speedup!)*

### 3.4 Per-Message Encoding: Dense Matvec vs M4RIE vs 2-Stage Factorized vs Additive FFT

1. **Current `Encoder.encode` ($O(k n)$ generic row-unpacking):** Hits `matrix0.pyx:5312`, which unpacks rows of `Matrix_gf2e_dense` into Python `FreeModuleElement_generic_dense` objects. At $q \ge 512$, `elem_cache = (order < 500)` turns off Givaro element interning, triggering a **134× cliff** (`136.95 ms` and `5.0 MB` peak at `q=1024, k=60`).
2. **In-Sage M4RIE 1-row matmul** `(matrix(F, 1, k, list(msg)) * G).row(0)`: Uses C M4RIE `mzed_mul` directly. Beats current encode by **2.4×–3.7×** at $q=256$ and **134.0×** at $q=1024$ (`136.95 ms -> 1.02 ms`), with zero heap explosion. (Slightly slower at tiny $q=16$ (`0.044 vs 0.010 ms`) due to 1-row `matrix()` construction overhead.)
3. **2-Stage Khatri–Rao factorized encoding ($O(r s' s + s r m) = O(k s + r n)$):** Reshape message into $r \times s'$ matrix $M$, compute $A = M \cdot V_{\text{outer}}$ ($r \times s$ outer coset evaluation), then evaluate each coset's degree-$(r-1)$ polynomial on its $m$ points. Algebraically reduces field MACs by a factor of $\frac{s m}{s + m}$ (e.g., **8× fewer MACs** at $n=256, m=s=16, k=224$). However, in Sage (`bench_optimizations.py`), allocating $s$ tiny `matrix(F, 1, r)` objects inside a Python loop adds $\sim 3$–$9\text{ ms}$ of Sage wrapper overhead (making it **slower** than 1-row M4RIE at $q \le 256$, though still **14.6× faster** than current `encode` at $q=1024$). In a Sage-free NumPy/C implementation with contiguous uint8/uint16 arrays (zero object overhead), the 2-stage factorization wins cleanly for $n \ge 64$.
4. **Additive / Multiplicative FFT ($O(n \log n)$):** For binary additive subgroups $H \subset \mathbb{F}_{2^e}$, the Lin–Chung–Han (2014) novel-basis additive FFT evaluates on all cosets of a subspace in $O(n \log n)$ XORs and $O(n \log n)$ field mults (and similarly radix-2/mixed-radix NTT for multiplicative subgroups when $m$ or $s$ is smooth). Practically worthwhile for $n \ge 1024$ in C/Cython; for $n \le 256$, 2-stage vectorized lookup-table matmul in NumPy/C already runs in microseconds.

### 3.5 `TamoBargErasureDecoder.decode_to_code`: Lagrange vs Parity-Check Recovery & Batching

Current implementation runs `poly_ring.lagrange_polynomial(known_pts)` per coset with $e_c \ge 1$ erasures (`polynomial_ring.py:2361`), calling pure-Python `divided_difference` ($O((m - e_c)^2)$) plus NTL Horner polynomial construction and $e_c$ NTL evaluations. Total complexity is $O(n + E_{\text{cosets}} \cdot m^2)$ (since $m - e_c \ge r \approx m$). Measured scaling on the $m = \sqrt{n}$ family matches theory closely: 1 erased coset scales as **$n^{1.12}$** (theory $m^2 = n^{1.0}$), all $s$ cosets erased scales as **$n^{1.58}$** (theory $s m^2 = n^{1.5}$).

**Better algorithm (shared canonical parity-check / barycentric solve):** Using the **single** canonical $( \delta - 1 ) \times m$ local parity-check matrix $H_0$ shared across all affine-equivalent cosets (§3.7), recover the $e_c < \delta$ erased symbols in coset $c$ by solving $H_{0, E} x_E = -H_{0, K} x_K$:
- For $e_c = 1$: single dot product + 1 division — **$O(m)$** (only $m-1$ mults!).
- For $e_c = 2$: 2 dot products + $2 \times 2$ Cramer's rule — **$O(m)$**!
- General $e_c \le \delta - 1$: $O(e_c m + e_c^3)$ (much smaller than $O(m^2)$ whenever $\delta \ll m$, which is the entire LRC regime!).
- **Batching across cosets:** If multiple cosets share the same erasure pattern (or in a NumPy/C backend), the right-hand side $-H_{0, K} X_K$ batches into a single matrix multiply.

**Prototype benchmark (`bench_optimizations.py`, all $s$ cosets with $\delta - 1$ erasures each, bit-exact verified):**

| Config | Current `lagrange_polynomial` | Shared Canonical PCM Solve | Speedup |
|---|---:|---:|---:|
| `q16_add_m4_k4` ($\delta=3, e_c=2$) | `0.52 ms` | `0.07 ms` | **7.3×** |
| `q64_add_m8_k24` ($\delta=3, e_c=2$) | `3.22 ms` | `0.10 ms` | **32.0×** |
| `q256_add_m16_k28` ($\delta=3, e_c=2$) | `46.55 ms` | `0.21 ms` | **216.7×** |
| `q256_mult_m15_k52` ($\delta=3, e_c=2$) | `44.82 ms` | `0.22 ms` | **205.8×** |
| `q256_add_m16_k48_d5` ($\delta=5, e_c=4$, Sage `solve_right`) | `66.35 ms` | `11.99 ms` | **5.5×** *(inline GE avoids Sage matrix overhead)* |
| `q1024_add_m32_k60` ($\delta=3, e_c=2$) | `480.25 ms` | `1.01 ms` | **476.2×** |

### 3.6 `TamoBargErrorDecoder`: Local GRS Algorithms, Syndrome Gating, and Redundant Work

1. **Syndrome gating effectiveness:** Extremely high! At `GF(256)` ($m=16, s=16$):
   - **0 errors:** `0.43 ms` vs **all 16 cosets errored:** `129.8 ms` (**302×** faster via gating!).
   - **1 errored coset:** `7.15 ms` (**18.1×** faster than decoding all cosets).
   - **Uncorrectable ($t+1$ errors in 1 coset):** `5.95 ms` (fails cleanly at `rem.is_zero()` / weight check without cascading).
   - However, **each** syndrome check currently does `self._local_parity_checks[i] * local_vec`, hitting generic `matrix0.pyx:5346` (`0.43 ms` total at $q=256$ vs **`0.017 ms`** for a single batched M4RIE `H0 * R_mat` — **25×** slower than the actual math!).
2. **Inside `GRSBerlekampWelchDecoder` (`sage/coding/grs_code.py:1186`):** Line-by-line timing on `GF(256), m=16` (`3.26 ms` total per errored coset):
   - `checks_and_inC` (`r in C` **redundantly** recomputes `H * local_vec` even though `TamoBargErrorDecoder` already verified syndrome $\neq 0$!): `0.046 ms` (1.6%)
   - `build_S` (Python `lambda i, j: pts[i]**j` without incremental powers): `0.127 ms` (4.5%)
   - `S.right_kernel()`: **`2.024 ms` (71.4%)** — of which M4RIE `echelonize()` is only **`0.023 ms` (0.8% of BW!)**, while Sage `VectorSpace` construction + `verbose()`/`time.process_time()` + `importlib.find_spec` consume `~2.00 ms`!
   - `Q0, Q1` build + NTL `quo_rem`: `0.196 ms` (6.9%)
   - `GRSEvaluationPolynomialEncoder.encode(f)` (`grs_code.py:1014`, $m$ NTL `ZZ_pEX` evals): `0.428 ms` (15.1%)
3. **Berlekamp–Welch ($O(m^3)$) vs Gao ($O(m^2)$) vs KeyEquationSyndrome ($O(m \delta + \delta^2)$):**
   - **Surprise finding on Sage's `GRSGaoDecoder` (`grs_code.py:1574`):** Although asymptotically $O(m^2)$, Sage's `Gao` calls `PolRing.lagrange_polynomial(points)` on all $m$ points in pure Python + NTL `ZZ_pEX`! For $m \in \{16, 32\}$ with small $\delta = 3$, this makes Sage's `Gao` **comparable to or slower than** `BerlekampWelch` (`859.7 ms` vs `549.4 ms` at `q=1024, m=32`!).
   - **Winner: `GRSKeyEquationSyndromeDecoder` (`grs_code.py:2233`):** Computes only $\delta - 1 = 2t$ syndromes, runs partial XGCD on degree $2t$ (instead of degree $m$!), and evaluates Forney's formula — $O(m \delta + \delta^2)$! Normally Sage forbids `KeyEquationSyndrome` when $0 \in \text{evaluation\_points}$ (`grs_code.py:2065`, because Forney computes $1/\alpha_i$), which blocks additive coset $A_0 = H$. **Our workaround:** using affine coset equivalence (§3.7), pick **any nonzero coset** $A_1 = \beta_1 + H$ ($0 \notin A_1$, always exists whenever $s \ge 2$) as the single shared canonical GRS code! Measured speedups (`bench_optimizations.py`, all $s$ cosets errored, bit-exact verified):
     - `q256_add_m16_k28`: `89.97 ms -> 38.18 ms` (**2.4×**)
     - `q256_mult_m15_k52`: `86.44 ms -> 35.09 ms` (**2.5×**)
     - `q256_add_m16_k48_d5` ($t=2$): `134.05 ms -> 73.59 ms` (**1.8×**)
     - `q1024_add_m32_k60`: `635.35 ms -> 215.43 ms` (**3.0×**; and `1err`: `22.73 ms -> 6.63 ms`, **3.4×**!)
   - *(In a Sage-free backend, Berlekamp–Massey / Peterson–Gorenstein–Zierler + Chien + Forney on uint8/uint16 GF tables eliminates the remaining ~95% Sage wrapper overhead for another ~50×–100× gain.)*

### 3.7 Shared Single Local Decoder via Affine Equivalence (Evaluated with Numbers)

**Mathematical proof (holds for BOTH `"add"` and `"mult"` subgroups):**
Let $H = \{h_0, \dots, h_{m-1}\}$ be the subgroup of size $m$.
- **Additive coset** $A_c = \beta_c + H$: For any polynomial $f$ of degree $< r$, $\tilde{f}(y) = f(\beta_c + y)$ also has degree $< r$, and $(f(\beta_c + h_0), \dots, f(\beta_c + h_{m-1})) = (\tilde{f}(h_0), \dots, \tilde{f}(h_{m-1}))$.
- **Multiplicative coset** $A_c = \beta_c H$ ($\beta_c \neq 0$): $\tilde{f}(y) = f(\beta_c y)$ also has degree $< r$, and $(f(\beta_c h_0), \dots, f(\beta_c h_{m-1})) = (\tilde{f}(h_0), \dots, \tilde{f}(h_{m-1}))$.

Thus **every coset $A_c$ defines the IDENTICAL local RS code $\text{RS}(H, r)$** once its elements are aligned to $H$ by permutation $\pi_c$ (`aligned[l] = partition[c][pi_c[l]]`). Furthermore, for canonical additive subgroups of $\mathbb{F}_{2^e}$ sorted by `field_element_to_int` (binary representation), XOR with a disjoint coset-leader bitmask is strictly order-preserving — so **$\pi_c = \text{id}$ for EVERY coset (`all_identity = True`, zero permutation cost!)**. For multiplicative cosets, $\pi_c$ is a precomputed $m$-element index list (`all_identity = False`).

**Measured savings (`bench_optimizations.py`, bit-exact agreement verified across all 6 configs):**

| Config ($s$ cosets) | Init Time: Orig ($s$ GRS) → Shared (1 GRS) | Init Peak Mem: Orig → Shared | 0-err Decode: Orig → Batched M4RIE `H0*R` | All-coset Decode: Orig BW → Shared `KeyEqSyn` | `all_identity` |
|---|---:|---:|---:|---:|:---:|
| `q16_add_m4_k4` ($s=4$) | `3.90 -> 0.45 ms` (**8.7×**) | `31 -> 9 KB` (**3.4×**) | `0.12 -> 0.12 ms` (1.0×) | `12.25 -> 6.07 ms` (**2.0×**) | `True` |
| `q64_add_m8_k24` ($s=8$) | `12.26 -> 0.53 ms` (**23.1×**) | `63 -> 10 KB` (**6.3×**) | `0.21 -> 0.14 ms` (**1.5×**) | `24.46 -> 16.81 ms` (**1.5×**) | `True` |
| `q256_add_m16_k28` ($s=16$) | `24.60 -> 0.64 ms` (**38.4×**) | `128 -> 15 KB` (**8.5×**) | `0.45 -> 0.22 ms` (**2.0×**) | `89.97 -> 38.18 ms` (**2.4×**) | `True` |
| `q256_mult_m15_k52` ($s=17$) | `20.29 -> 0.70 ms` (**29.0×**) | `132 -> 18 KB` (**7.3×**) | `0.47 -> 0.23 ms` (**2.0×**) | `86.44 -> 35.09 ms` (**2.5×**) | `False` |
| `q256_add_m16_k48_d5` ($s=16$) | `29.91 -> 0.72 ms` (**41.5×**) | `122 -> 18 KB` (**6.8×**) | `0.47 -> 0.24 ms` (**2.0×**) | `134.05 -> 73.59 ms` (**1.8×**) | `True` |
| `q1024_add_m32_k60` ($s=32$) | `54.92 -> 3.18 ms` (**17.3×**) | `416 -> 36 KB` (**11.6×**) | `3.36 -> 0.58 ms` (**5.8×**) | `635.35 -> 215.43 ms` (**3.0×**) | `True` |

*(At `q=4096, m=64, s=64`, current `ErrorDecoder.__init__` takes `336.6 ms` and `1.21 MB`; sharing 1 local decoder scales as $O(1)$ in $s$, projecting `~6 ms` and `~50 KB` — **~55× faster, ~24× less memory**!)*

---

## 4. Per-Word Latency & Scaling Sweeps

### 4.1 Per-Word Latency Across $q \in \{16, 64, 256, 1024\}$ and Varied $(r, k, \delta, \text{type})$

All latencies are **median [p10, p90] ms** (`per_word_and_profile.json`, cached `generator_matrix`):

| Config $(q, n, k, r, \delta, \text{type})$ | `enc_poly` | `enc.encode` | `EraDec` (0 / 1 / all max) | `ErrDec` (0 / 1 / all $t$ / 1 uncorr) |
|---|---:|---:|---:|---:|
| `q16_add_m4_k4` $(16,16,4,2,3,\text{add})$ | `0.08` `[0.08,0.10]` | `0.01` `[0.01,0.01]` | `0.06` / `0.17` / `0.56` `[0.49,4.73]` | `0.12` / `3.44` / `12.41` `[9.90,15.1]` / `2.96` |
| `q16_mult_m5_k6` $(16,15,6,3,3,\text{mult})$ | `0.09` `[0.09,0.10]` | `0.01` `[0.01,0.01]` | `0.06` / `0.24` / `0.58` `[0.52,4.75]` | `0.11` / `4.66` / `8.95` `[5.90,11.1]` / `2.41` |
| `q64_add_m8_k12` $(64,64,12,6,3,\text{add})$ | `0.31` `[0.30,4.37]` | `0.03` `[0.03,0.03]` | `0.09` / `0.43` / `11.33` `[7.32,11.9]` | `0.20` / `4.97` / `35.02` `[33.5,37.5]` / `4.84` |
| `q64_add_m8_k24` $(64,64,24,6,3,\text{add})$ | `0.78` `[0.75,4.88]` | `0.05` `[0.05,0.06]` | `0.09` / `0.46` / `11.51` `[7.96,13.3]` | `0.20` / `5.01` / `35.74` `[32.7,43.2]` / `5.22` |
| `q64_mult_m7_k36` $(64,63,36,5,3,\text{mult})$ | `0.91` `[0.83,4.97]` | `0.08` `[0.08,0.16]` | `0.09` / `0.45` / `11.16` `[7.14,11.3]` | `0.21` / `5.03` / `35.98` `[26.2,44.6]` / `4.67` |
| `q256_add_m16_k28` $(256,256,28,14,3,\text{add})$ | `4.98` `[1.02,7.03]` | `0.31` `[0.27,3.26]` | `0.18` / `5.48` / `70.73` `[66.5,91.1]` | `0.43` / `7.15` / `129.8` `[126,163]` / `5.95` |
| `q256_add_m16_k112` $(256,256,112,14,3,\text{add})$ | `21.77` `[17.5,29.7]` | `3.16` `[1.18,3.22]` | `0.19` / `5.29` / `72.57` `[71.1,75.8]` | `0.45` / `8.23` / `135.3` `[131,159]` / `6.02` |
| `q256_add_m8_k48` $(256,256,48,6,3,\text{add})$ | `10.97` `[7.00,11.2]` | `0.51` `[0.48,4.51]` | `0.20` / `3.01` / `52.67` `[48.5,56.8]` | `0.63` / `6.13` / `126.0` `[115,155]` / `3.93` |
| `q256_add_m32_k60` $(256,256,60,30,3,\text{add})$ | `8.32` `[5.80,10.0]` | `0.65` `[0.59,4.68]` | `0.18` / `13.46` / `133.4` `[112,147]` | `0.33` / `16.46` / `173.7` `[161,202]` / `9.37` |
| `q256_add_m16_k48_d5` $(256,256,48,12,5,\text{add})$ | `8.70` `[6.72,14.6]` | `0.51` `[0.47,5.40]` | `0.18` / `5.38` / `73.27` `[63.9,84.5]` | `0.47` / `8.89` / `143.7` `[128,160]` / `6.02` |
| `q256_mult_m15_k52` $(256,255,52,13,3,\text{mult})$ | `5.78` `[2.51,5.81]` | `0.58` `[0.53,4.57]` | `0.18` / `5.56` / `69.47` `[66.0,78.1]` | `0.45` / `7.05` / `131.6` `[112,168]` / `5.86` |
| `q1024_add_m32_k60` $(1024,1024,60,30,3,\text{add})$ | `9.60` `[7.03,12.3]` | **`152.5`** `[141,250]` | `0.57` / `13.08` / `452.6` `[438,513]` | `3.48` / `19.52` / `827.6` `[811,1026]` / `10.43` |
| `q1024_add_m16_k28` $(1024,1024,28,14,3,\text{add})$ | `5.37` `[1.36,5.67]` | **`77.05`** `[15.7,131]` | `0.61` / `5.99` / `298.4` `[277,333]` | `3.94` / `10.46` / `575.7` `[534,632]` / `11.89` |

### 4.2 Empirical Scaling Exponents vs Theory ($m = \sqrt{n}$, $s = \sqrt{n}$, $q = n \in \{16, 64, 256, 1024\}$)

| Operation | Family | Empirical Log–Log Slope | Theoretical Complexity (in $n$) | Explanation of Deviation |
|---|---|---:|---|---|
| `TamoBargCode.__init__` | $k \approx n/2$ ($16..4096$) | **$n^{1.29}$** | $O(n \log n)$ (`n^1.1`–`1.2`) | Matches $O(q \log q)$ sorting + `divisors(q-1)` subgroup enumeration |
| `VectorEncoder.__init__` | $k \approx n/2$ ($16..4096$) | **$n^{1.57}$** | $O(s \cdot n \log n) = O(n^{1.5}\log n)$ | Matches `algebra_basis` ($s$ NTL poly mults) + unused `annihilator` |
| `generator_matrix()` (uncached) | $k = 2r$ ($16..1024$) | **$n^{2.04}$** | $O(k n D_{\text{avg}}) = O(\sqrt{n}\cdot n\cdot\sqrt{n}) = O(n^2)$ | Exact match (`2.04` vs `2.00`) |
| `generator_matrix()` (uncached) | $k \approx n/2$ ($64 \to 256$) | **$n^{3.03}$** ($n^{2.58}$ over $16..256$) | $O(k n D_{\text{avg}}) = O(n^3)$ | Exact match at $n \ge 64$ |
| `ErasureDecoder.__init__` | $16..4096$ | **$n^{0.00}$** | $O(1)$ | Constant-time lazy space creation |
| `ErrorDecoder.__init__` | $k \approx n/2$ ($16..4096$) | **$n^{0.71}$** | $O(s) + O(s m^2) = O(n^{0.5}) + O(n^{1.5})$ | Dominated by $O(s) = O(n^{0.5})$ Sage GRS/Decoder creation up to $m=64$ |
| `enc_poly(msg)` | $k = 2r$ ($16..1024$) | **$n^{1.24}$** | $O(k \cdot m) = O(n)$ | Slightly superlinear due to Givaro→NTL `ZZ_pE` degree overhead ($\log q$) |
| `enc.encode(msg)` | $k = 2r$ ($16..1024$) | **$n^{2.23}$** | $O(k n) = O(n^{1.5})$ | Skewed sharply upward by the **134× Givaro `elem_cache` cliff at $q=1024$** (slope $16..256$ is $n^{1.21}$) |
| `EraDec` (1 coset erased) | $k = 2r$ ($16..1024$) | **$n^{1.12}$** | $O(n + m^2) = O(n)$ | Matches $O(m^2) = O(n)$ Lagrange interpolation |
| `EraDec` (all $s$ cosets erased) | $k = 2r$ ($16..1024$) | **$n^{1.58}$** | $O(s m^2) = O(n^{1.5})$ | Exact match (`1.58` vs `1.50`) |
| `ErrDec` (0 errors, syndrome gate) | $k = 2r$ ($16..1024$) | **$n^{0.79}$** | $O(s + n \delta) = O(n)$ | Sublinear at small $n$ due to constant per-call setup, jumping at $q=1024$ (`elem_cache` cliff) |
| `ErrDec` (1 coset errored) | $k = 2r$ ($16..1024$) | **$n^{0.40}$** | $O(n + m^3) = O(n^{1.5})$ | Constant `~3 ms` Sage `right_kernel()` framework overhead masks $O(m^3)$ math up to $m=32$! |
| `ErrDec` (all $s$ cosets errored) | $k = 2r$ ($16..1024$) | **$n^{1.00}$** | $O(s m^3) = O(n^2)$ | Constant per-coset Sage framework overhead ($s \times 3$–$20\text{ ms}$) dominates $O(s m^3)$ math up to $m=32$! |

---

## 5. Framework Overhead vs Actual Math Split (%)

Measured via `cProfile`/`pstats` (`analyze_profiles.py`) combined with direct sub-step micro-instrumentation (`bench_framework_vs_math.py`, plus uncached `S.right_kernel()` vs M4RIE `echelonize()` separation).

| Operation | Config | Total Time (ms) | Core Math (ms / %) | Sage Framework & Wrapper Overhead (ms / %) | Dominant Overhead Mechanism (with Sage Source Citation) |
|---|---|---:|---:|---:|---|
| `TamoBargCode.__init__` | `GF(16) [16,5,2]` | `0.094` | `0.048` (**50.9%**) | `0.046` (**49.1%**) | `AbstractLinearCode.__init__` (`linear_code.py:346`) + `Parent.__init__` |
| `TamoBargCode.__init__` | `GF(256) [256,28,14]` | `0.879` | `0.612` (**69.6%**) | `0.267` (**30.4%**) | `AbstractLinearCode.__init__` (plus `0.15 ms` wasted computing unused `mult` subgroups) |
| `VectorEncoder.__init__` | `GF(256) [256,28,14]` | `63.78` | `13.47` (**21.1%**) *(useful)* + `41.12` *(wasted ann.)* | `9.20` (**14.4%**) *(+64.5% wasted annihilator!)* | Unused `build_annihilator_polynomial` (`41.1 ms`) + `PolynomialRing` setup |
| `code.encode(msg)` | `GF(16) [16,5,2]` | `0.0117` | `0.0067` (**56.6%**) *(vs `v*G`)* | `0.0050` (**43.4%**) | `code.encoder()` cache lookup (`0.001 ms`) + `message_space()` / `word in M` / `vector(word)` (`encoder.py:119`) |
| `code.encode(msg)` | `GF(256) [256,28,14]` | `0.270` | `0.055` (**20.5%**) *(M4RIE `mzed_mul`)* | `0.215` (**79.5%**) | `Matrix._vector_times_matrix_` (`matrix0.pyx:5312`) row-unpacking (`0.21 ms`) + `Encoder.encode` checks (`0.004 ms`) |
| `enc.encode(msg)` | `GF(1024) [1024,30,30]` | `134.03` | `1.02` (**0.8%**) *(M4RIE `mzed_mul`)* | `133.01` (**99.2%**) | `matrix0.pyx:5312` + Givaro `elem_cache = (order < 500)` (`finite_field_constructor.py:755`, `element_givaro.pyx:1760`) heap-allocating 60k+ objects |
| `EraDec.decode` (1 coset) | `GF(16) [16,5,2]` | `0.186` | `0.075` (**40.5%**) | `0.111` (**59.5%**) | Python listcomp coercion (`tamo_barg.py:397-398`) + final `vector(F, ...)` |
| `EraDec.decode` (1 coset) | `GF(256) [256,28,14]` | `3.354` | `2.740` (**81.7%**) | `0.615` (**18.3%**) | `lagrange_polynomial` (`polynomial_ring.py:2361`) + NTL `_element_constructor_` (`16%` of cProfile) |
| `EraDec.decode` (all cosets) | `GF(256) [256,28,14]` | `60.15` | `54.25` (**90.2%**) | `5.89` (**9.8%**) | Dominated by $s=16$ calls to $O(m^2)$ `lagrange_polynomial` |
| `ErrDec.decode` (0 errors) | `GF(16) [16,5,2]` | `0.129` | `0.012` (**9.6%**) *(M4RIE `H0*R`)* | `0.116` (**90.4%**) | `copy(vector(...))` + per-coset `vector()` slicing + `matrix0.pyx:5346` column unpacking |
| `ErrDec.decode` (0 errors) | `GF(256) [256,28,14]` | `0.441` | `0.017` (**3.9%**) *(M4RIE `H0*R`)* | `0.424` (**96.1%**) | `matrix0.pyx:5346` column unpacking (`0.30 ms`) + `FreeModule` / `_element_constructor_` (`cProfile`: 28%!) + `vector` slicing |
| `ErrDec.decode` (1 error) | `GF(256) [256,28,14]` | `4.57–6.20` | `0.664` (**10.7%–14.5%**) *(M4RIE syn + `echelonize` + `quo_rem` + re-encode)* | `3.91–5.54` (**85.5%–89.3%**) | `S.right_kernel()` `VectorSpace` + `time.process_time` (25% of cProfile!) + `importlib` (10%!) = `2.00 ms` vs `0.023 ms` actual M4RIE `echelonize`; redundant `r in C` (`grs_code.py:1194`); `matrix0.pyx:5346` |
| `ErrDec.decode` (all cosets) | `GF(256) [256,28,14]` | `116.8–132.9` | `10.4` (**8.9%**) *(M4RIE syn + $16\times$ core BW)* | `106.4` (**91.1%**) | Same per-coset `right_kernel()` + `importlib` + `process_time` overhead × 16 cosets |

*Note on `code.encode` / `code.decode_to_code` registration dispatch overhead:* Calling through `code.encode(msg)` vs `enc.encode(msg)` adds only `~1.0 µs` (`@cached_method` lookup on `AbstractCode.encoder`), and `code.decode_to_code(r, "ErrorDecoder")` vs `err_dec.decode_to_code(r)` adds only `~1–5 µs` (`AbstractCode.decode_to_code` does **not** perform an extra `input_space` membership check — validation occurs inside `GRSBerlekampWelchDecoder` and `Encoder.encode`). The overwhelming framework penalty is **not** top-level registration dispatch; it is **inside** Sage's `Matrix._vector_times_matrix_`, `Matrix._matrix_times_vector_`, `Matrix.right_kernel()`, `Polynomial_ZZ_pEX.__call__`, and Givaro `elem_cache` cutoff.

---

## 6. Sage-Replacement Assessment & Priority Summary

`tamo_barg.py` has **low-to-moderate Sage replacement difficulty (max 3/5)**:
- **Finite field arithmetic (`GF(q)`):** Easily replaced for $q \le 2^{16}$ with log/antilog lookup tables (`uint8` for $q \le 256$, `uint16` for $q \le 65536$) in NumPy/Cython/Rust (or `galois` / `gf256`), eliminating **all** Givaro boxing, `elem_cache` cliffs, and NTL `ZZ_pE` marshaling.
- **Polynomials & generator matrix:** Pure 1D/2D NumPy `uint8`/`uint16` arrays; Khatri–Rao `generator_matrix` build is 15 lines of NumPy broadcasting.
- **Local erasure & error decoders:** A single canonical local parity-check matrix $H_0$ + inline Gaussian elimination handles erasures (difficulty 2/5), and a ~100-LOC Berlekamp–Massey + Chien search + Forney formula (or Euclidean key-equation solver) replaces `GeneralizedReedSolomonCode` + `GRSBerlekampWelchDecoder` completely (difficulty 3/5), with expected **50×–500× end-to-end speedup** over current Sage code.

### Prioritized Action Roadmap (Speed-up × Usage Frequency)

| Priority | Target | Concrete Change | Expected / Measured Speedup |
|---|---|---|---|
| **P0** | `TamoBargVectorEncoder.generator_matrix` | Replace `[[poly(alpha)...]]` with Khatri–Rao / Vandermonde build ($O(k n)$, 0 NTL evals) | **162×–352×** ($>1{,}000\times$ at $q=256, k=112$) |
| **P0** | `TamoBargErrorDecoder` (`__init__` + `decode_to_code`) | Share **1** canonical local GRS decoder via affine coset equivalence + batched M4RIE `H0 * R_mat` syndrome check + `KeyEquationSyndrome` (nonzero-coset shift for `add`) | **17×–38×** init, **8.5×–11.6×** less mem, **2×–5.8×** 0-err, **2.4×–3.4×** errored (in-Sage); **50×–200×** Sage-free |
| **P0** | `TamoBargVectorEncoder.encode` (override `Encoder.encode`) | Use M4RIE `(matrix(F, 1, k, list(word)) * G).row(0)` (in-Sage) or 2-stage factorized $O(k(s+m))$ (Sage-free) | **2.4×–3.7×** at $q=256$, **134×** at $q=1024$ (eliminates 5 MB heap cliff!) |
| **P0** | `TamoBargErasureDecoder.decode_to_code` | Replace $O(m^2)$ `lagrange_polynomial` with shared canonical $H_0$ erasure solve $O(e_c m + e_c^3)$ | **7.3×–476×** measured |
| **P1** | `TamoBargVectorEncoder.__init__` & `enc_poly` | Drop unused annihilator (65% of init), generate only $s' = \lceil k/r \rceil$ powers of $g$, Horner-in-$g$ `enc_poly` | **3×–10×** init & `enc_poly` |
| **P2** | `TamoBargCode.__init__`, `__hash__`, `minimum_distance` | Build only requested subgroup of size $m$; cache `_cached_hash`; accept `algorithm=None` in `minimum_distance` | **2×–5×** code init, **500×** `__hash__`, API cleanliness |
