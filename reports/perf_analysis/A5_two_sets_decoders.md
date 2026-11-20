# A5 — Two-Set Tamo–Barg Decoders: Algorithm, Convergence & Performance Analysis

**Scope:** All six decoder classes in [`src/tamo_barg_lrc/tamo_barg_two_sets.py`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L511-L949): `TamoBargIterativeDecoder`, `TamoBargIterativeEEDecoder`, `TamoBargGlobalDecoder`, `TamoBargGlobalEEDecoder`, `TamoBargTwoStepDecoder`, and `TamoBargTwoStepEEDecoder` (every `__init__`, `_partition_is_syndrome_clean`, `decode_to_code`, and `decode_to_message` method).

**Environment & Methodology:**
- Machine: 2 vCPU Linux (Debian 12), Python 3.11.2 (`/home/antigravity/.venv-lrc/bin/python`), modular `passagemath` 10.8.x (Givaro `FiniteField_givaro`, NTL `Polynomial_ZZ_pEX`, M4RIE `Matrix_gf2e_dense`; `sagemath-brial`/`pbori` absent).
- Fixed seeds (`random.seed`, `np.random.seed`, Sage `set_random_seed`). Warm-up $\ge 3$ calls ($\ge 1$ at $q=1024$); repeats $\ge 20$ ($\ge 5$ for $>1\text{ s}$ calls).
- Because sibling swarm benchmarks shared the 2 vCPUs during parts of the run (load 4.0–7.6), all timing scripts record both contention-free single-core CPU time (`time.process_time`, primary for medians, scaling exponents, and OLS fits) and wall-clock time (`time.perf_counter`). `tracemalloc` peak memory is measured in an isolated post-timing pass so allocation hooks never distort latencies.
- Reproducible scripts and raw logs live in [`reports/perf_analysis/scratch/A5/`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/).

---

## 1. Overall Design Narrative & Architectural Critique

The six two-set decoders form a clean $3 \times 2$ matrix across **strategy** (Iterative local, Global GRS envelope, Two-Step hybrid) and **mode** (hard-decision vs. erasure-and-error, EE). Each partition $i \in \{0, 1\}$ decomposes the $n$ evaluation points into $s_i = n / c_i$ disjoint cosets of size $c_i = r_i + d_{\text{loc},i} - 1$, where restricting any valid codeword to a coset yields a codeword of an $[c_i, r_i, d_{\text{loc},i}]$ GRS code. Globally, the $k$-dimensional intersection basis $\text{span}(\mathcal{B}_0) \cap \text{span}(\mathcal{B}_1)$ truncated to its $k$ lowest-degree polynomials sits inside an enveloping $[n, n - d_{\text{design}} + 1, d_{\text{design}}]$ GRS code.

**What works well:**
1. **Local-first syndrome gating:** Checking local syndromes before invoking local GRS decoders avoids decoding clean cosets, and checking the global syndrome `(self._global_pcm * work).is_zero()` before invoking global Gao avoids the massive length-$n$ global decode whenever Stage 1 already succeeded (100% of words up to $w \le 16$ on GF(64) and $w \le 10$ on GF(256) in our suites).
2. **Non-mutating, deterministic contracts:** Every decoder copies its input (`copy(vector(self._F, r))`) and handles immutable inputs safely; `IterativeEEDecoder` explicitly handles the $e = d_{\text{loc}} - 1$ boundary via `interpolate_erasures` (fixing a Sage `GRSErrorErasureDecoder` edge case).
3. **Synergy of TwoStep + EE when $k < k_{\max}$:** On subcodes where $d_{\text{design}} > (t_{\text{loc}} + 1)^2$ (e.g., GF(64) $k=8 < k_{\max}=16$, $t_{\text{glob}}=22$), Stage 1 peels off most errors cheaply while Stage 2 rescues stalled stopping sets — boosting success at $w=28$ from **9% (Iterative)** and **0% (Global)** to **81% (TwoStep)** and **90% (TwoStepEE)**.

**Critical architectural & performance flaws exposed by profiling and Sage source inspection:**
1. **Passagemath `Sequence` / missing-`pbori` import storm (43% of Iterative runtime):** Every call to `vector(self._F, ...)` invokes `sage.structure.sequence.Sequence` ([`sequence.py:297`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/structure/sequence.py#L297)), which evaluates `isinstance(F, (MPolynomialRing_base, BooleanMonomialMonoid))`. Because `BooleanMonomialMonoid` is a `lazy_import` of uninstalled `pbori`, every `vector(F, lst)` triggers an uncached failing `importlib._find_spec` disk scan — costing **51.3 µs/call** vs. **2.97 µs** via `VectorSpace(F, c)(lst)` (**17.3×**) or **0.39 µs** via `V.element_class` (**131.6×**).
2. **`Matrix_gf2e_dense * vector` bypasses M4RIE:** `Matrix_gf2e_dense` does not override `_matrix_times_vector_` ([`matrix0.pyx:5315-5347`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/matrix/matrix0.pyx#L5315-L5347)), falling back to a Python comprehension that unpacks every column into a boxed vector and `sum()`s them. Batched M4RIE matrix-matrix multiply or NumPy GF($2^m$) log-table matmul is **58×–573× faster** per coset.
3. **Sage `GRSGaoDecoder` is pure-Python $O(n^2)$, not $O(n \log^2 n)$:** Although NTL's `ZZ_pEX` multiplication is fast $O(n^{1.24})$ (8.6 ms at deg 1024; 46.1 ms at deg 4096), `GRSGaoDecoder` ([`grs_code.py:1533-1629`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1533-L1629)) runs pure-Python $O(n^2)$ `divided_difference` interpolation (**44%** of time, **40.6 MiB** peak at $n=1024$), a pure-Python quadratic Euclid loop `_partial_xgcd` (**34%**), and point-by-point polynomial evaluation (**19%**). Even at $n=256$, cubic `GRSBerlekampWelchDecoder` (121.5 ms, using M4RIE) beats `GRSGaoDecoder` (435.5 ms) by **3.6×**!
4. **Sage `GRSErrorErasureDecoder` rebuilds a fresh GRS code on every call:** Even with 0 erasures, [`grs_code.py:1971-1975`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1971-L1975) constructs a brand-new `GeneralizedReedSolomonCode`, discarding all cached vanishing polynomials and generator matrices. This causes a **1.8×–2.0× slowdown** and **3.7×–10× peak memory blowup** (151.9 MiB at $q=1024$) across all three EE decoders.
5. **Algorithmic defects in iterative control flow:** (a) `IterativeDecoder` only detects 1-step fixed points (`not changed_in_itr`), so high-weight words enter **2-cycle/k-cycle misdecode oscillations** that burn all `max_num_of_itr` passes; (b) `IterativeEEDecoder` waits blindly for a fixed `erasure_start_itr` even after hard-decision stalls, wasting 2–4 no-op passes, and unconditionally re-checks `_partition_is_syndrome_clean(0, work)` on clean words (line 740, wasting 30% of clean-word time); (c) when `era_wt == d_loc - 1`, `interpolate_erasures` has zero error-detection redundancy and can silently misdecode into a wrong codeword when unmarked errors coexist.
6. **Pure-Python log-table syndrome decoding beats Sage local Gao by 56×–109×:** Shifting any local coset by $\delta \notin \text{coset}$ ($\alpha_j' = \alpha_j + \delta$) preserves the local RS code while eliminating $\alpha = 0$, unlocking $O(c(c-r))$ Berlekamp–Massey + Chien + Forney decoding. Our verified pure-Python prototype runs in **17.8–59.8 µs/coset** vs. **1,103–3,336 µs** for Sage Gao.

---

## 2. Six-Decoder Comparison Table

Benchmark family (self-similar `["add", "add"]`, $n=q$, $c=\sqrt{q}$, $r=[c/2, c/2]$, $d_{\text{loc}}=[c/2+1, c/2+1]$, $k=q/4=k_{\max}$, $d_{\text{design}} = 3\sqrt{q}/2 + 1$, $t_{\text{glob}} = \lfloor(d_{\text{design}}-1)/2\rfloor$):

| Decoder | Mode / Strategy | Theoretical Radius | Empirical Scaling $\alpha$ ($w=t_{\text{glob}}$, $q=64..1024$) | GF(64) Init / Clean / Active ($w=6$) CPU ms | GF(256) Init / Clean / Active ($w=12$) CPU ms | GF(1024) Init / Clean / Active ($w=24$) CPU ms | Peak Mem GF(256) / GF(1024) |
|---|---|---|---|---|---|---|---|
| **`IterativeDecoder`** | Hard, local Gao | Local $t_{\text{loc}}$; stops on $(t_{\text{loc}}+1)^2$ min grid | $n^{1.24}$ | 8.87 / 1.11 / **7.06** | 19.08 / 2.53 / **29.40** | 68.04 / 10.96 / **217.27** | 9.9 KiB / 101 KiB |
| **`IterativeEEDecoder`** | EE (DUE marking), local `ErrorErasure` + Lagrange | Local $2v+e < d_{\text{loc}}$; breaks $(t_{\text{loc}}+1)^2$ via DUE | $n^{1.12}$ | 9.29 / 1.89 / **17.19** | 22.65 / 4.56 / **57.80** | 73.46 / 19.07 / **387.43** | 103 KiB / 762 KiB |
| **`GlobalDecoder`** | Hard, length-$n$ Gao | Exact $t_{\text{glob}} = \lfloor(d_{\text{des}}-1)/2\rfloor$ | $n^{2.21}$ | 1.31 / 0.063 / **18.72** | 8.25 / 0.346 / **367.06** | 221.19 / 11.37 / **8,612.90** | 341 KiB / 40.6 MiB |
| **`GlobalEEDecoder`** | EE, punctured length-$n$ Gao / Lagrange | Exact $2v+e < d_{\text{design}}$ | $n^{2.26}$ | 1.44 / 0.076 / **31.28** ($e=0$) | 8.79 / 0.360 / **557.60** ($e=0$) | 218.29 / 14.57 / **16,321.34** ($e=0$) | 1.08 MiB / 151.9 MiB |
| **`TwoStepDecoder`** | Iterative $\to$ Global | $\max(\text{Iter}, t_{\text{glob}})$ + Stage-2 rescue | $n^{1.24}$ (when Stage 1 finishes); $n^{2.21}$ on fallback | 9.98 / 1.23 / **7.50** | 30.19 / 3.07 / **29.68** | 289.38 / 22.14 / **235.18** | 85.3 KiB / 4.0 MiB |
| **`TwoStepEEDecoder`** | IterEE $\to$ GlobalEE | $\max(\text{IterEE}, t_{\text{glob}})$ + EE rescue | $n^{1.14}$ (Stage 1); $n^{2.26}$ on fallback | 11.37 / 2.06 / **17.85** | 30.51 / 4.98 / **59.91** | 291.33 / 31.67 / **420.30** | 155 KiB / 4.0 MiB |

*(Note: Only `TamoBargIterativeDecoder` and `TamoBargIterativeEEDecoder` define `_partition_is_syndrome_clean`; `Global`, `GlobalEE`, `TwoStep`, and `TwoStepEE` do not. All 6 `__init__` + 6 `decode_to_code` + 2 `_partition_is_syndrome_clean` = **14 full mandatory function rows** below, plus a summary note for the 4 thin `decode_to_message` and 6 `_repr_` methods.)*

---

## 3. Mandatory Per-Function Analysis (14 Core Methods + Wrappers)

Notation: $n = \text{length}$, $k = \text{dimension}$, $q = |\mathbb{F}_q|$, $c_i = r_i + d_{\text{loc},i} - 1$ (coset size), $s_i = n / c_i$ (cosets in partition $i$), $m_i = d_{\text{loc},i} - 1 = c_i - r_i$, $d_g = d_{\text{design}}$, $m_g = d_g - 1$, $I = \text{iterations executed}$, $D_{\text{dirty}} = \text{dirty coset decodes across all iterations}$. For our self-similar scaling sweep ($n=q$, $c_0=c_1=\sqrt{n}$, $s_0=s_1=\sqrt{n}$, $r_i = m_i = \sqrt{n}/2$): $s_i \cdot c_i = n$, $s_i \cdot c_i^2 = n^{1.5}$.

### 3.1 `TamoBargIterativeDecoder`

#### Row 1 — `TamoBargIterativeDecoder.__init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None`
- **Purpose:** Precompute $s_0 + s_1$ local GRS codes, Gao decoders, and local parity-check matrices across both partitions ([`tamo_barg_two_sets.py:514-547`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L514-L547)).
- **Algorithm:** Loops over both partitions and each coset; constructs `GeneralizedReedSolomonCode(coset, r_i)`, instantiates `.decoder("Gao")`, and builds `H_loc` via `_safe_parity_check_matrix` (dual GRS Vandermonde matrix via `multipliers_product`).
- **Complexity:** Time $O\!\left(\sum_{i=0}^1 s_i (c_i^2 + m_i c_i)\right) = O(n(c_0 + c_1))$ (for $c=\sqrt{n}$: $O(n^{1.5})$); Space $O\!\left(\sum_i s_i m_i c_i\right)$ for stored parity-check matrices (for $c=\sqrt{n}$: $O(n^{1.5})$).
- **Sage primitives:** `GeneralizedReedSolomonCode` ([`grs_code.py:145`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L145), pure Python + vector checks); `.decoder("Gao")` (`GRSGaoDecoder.__init__`, lightweight); `parity_check_matrix()` ([`grs_code.py:440`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L440), pure-Python `multipliers_product` $O(c_i^2)$ + M4RIE `Matrix_gf2e_dense` construction).
- **Measured:** CPU median (p10 / p90): GF(16) `4.07` (3.70/4.68) ms; GF(64) `8.87` (8.32/9.81) ms; GF(256) `19.08` (18.42/20.69) ms; GF(1024) `68.04` (67.31/68.89) ms. Scaling $\alpha = 0.73$ ($q=64..1024$, dominated by $O(\sqrt{n})$ per-coset Sage object setup transitioning to $O(n^{1.5})$). Peak mem: `243.7 KiB` (GF(256)), `821.8 KiB` (GF(1024)).
- **Bottleneck:** Per-coset Sage `GeneralizedReedSolomonCode` + dual-code construction ($2s$ instances, ~0.6–1.1 ms each).
- **Correctness risks:** None for valid codes; `_safe_parity_check_matrix` handles $r_i = c_i$ (`d_local = 1`), though `TamoBargCodeTwoSets` enforces $d_{\text{loc}} \ge 2$.
- **Better algorithm:** For additive/multiplicative cosets, all cosets of $H_i$ are **affine translates** ($H_i + a_j$) or **scalar multiples** ($a_j H_i$) of ONE canonical subgroup $H_i$! Thus all $s_i$ cosets share the **exact same** parity-check matrix (up to column-wise scalar shift for multiplicative, and literally **identical** $H_i$ for additive cosets since $\text{RS}(H_i + a_j, r_i) = \text{RS}(H_i, r_i)$!). Building **1** canonical local parity-check matrix + shift table per partition cuts init from $O(s_i c_i^2)$ to $O(c_i^2)$ (**$s_i\times$ speedup**, 16×–32×).
- **Sage-replacement difficulty:** `2/5` — Precompute a single Vandermonde parity-check matrix per partition over GF($2^m$) log/antilog tables.
- **Priority:** `P2` (amortized across many decodes, though trivial $s_i\times$ win).

#### Row 2 — `TamoBargIterativeDecoder._partition_is_syndrome_clean(self, part_idx: int, work: Any) -> bool`
- **Purpose:** Verify whether every coset in partition `part_idx` has zero local syndrome ([`tamo_barg_two_sets.py:553-559`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L553-L559)).
- **Algorithm:** Linear scan over cosets $j = 0 \dots s_{\text{part}}-1$; extracts `local_vec = vector(self._F, [work[idx] for idx in idxs])`, computes `H[j] * local_vec`, returns `False` on first non-zero syndrome.
- **Complexity:** Worst-case (clean word): Time $O(s_i (C_{\text{vec}} + m_i c_i)) = O(n \cdot m_i + s_i C_{\text{vec}})$, Space $O(c_i + m_i)$. Dirty word: $O(1)$ cosets (early exit).
- **Sage primitives:** `vector(self._F, [...])` $\to$ `Sequence` ([`sequence.py:297`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/structure/sequence.py#L297), **51.3 µs/call** due to uncached missing-`pbori` `importlib` scan!); `Matrix_gf2e_dense * FreeModuleElement_generic_dense` $\to$ generic `Matrix._matrix_times_vector_` ([`matrix0.pyx:5315-5347`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/matrix/matrix0.pyx#L5315-L5347), Python column unpacking + `sum()`, **13.3 µs/coset** at $q=256$).
- **Measured (clean / dirty):** CPU median (p10 / p90): GF(16) `0.235` (0.228/0.332) ms / `0.177` ms; GF(64) `0.520` (0.485/0.595) ms / `0.065` ms; GF(256) `1.256` (1.168/1.330) ms / `0.144` ms; GF(1024) `5.417` (5.142/5.621) ms / `0.142` ms. Scaling (clean) $\alpha = 0.85$ ($q=64..1024$). Peak mem: `4.3 KiB` (GF(256)), `46.9 KiB` (GF(1024)).
- **Bottleneck:** **~80%** of runtime at $q \le 256$ is `vector(self._F, ...)` hitting `Sequence` $\to$ `BooleanMonomialMonoid` failed import! Remaining ~20% is `Matrix._matrix_times_vector_` column unpacking.
- **Correctness risks:** None.
- **Better algorithm:** (1) Immediate Sage fix: use cached `VectorSpace(F, c)` or pack all $s_i$ coset vectors into a single $c_i \times s_i$ `Matrix_gf2e_dense` and compute `H_canonical * W` in **one** M4RIE matmul (**142× faster** at $q=256$); (2) NumPy GF($2^m$) log-table matmul across all $s_i$ cosets (and across word batches $B$): **58×–290× faster** per coset!
- **Sage-replacement difficulty:** `1/5` — `H_log` + NumPy XOR/lookup matmul (`~25` LOC, already prototyped & verified in `bench_04`).
- **Priority:** `P0` (runs inside every iteration and clean-word check).

#### Row 3 — `TamoBargIterativeDecoder.decode_to_code(self, r: Any) -> tuple[Any, int]`
- **Purpose:** Iteratively hard-decision decode dirty cosets alternating across Partitions 0 and 1 without mutating `r` ([`tamo_barg_two_sets.py:561-603`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L561-L603)).
- **Algorithm:** Copies `r` into mutable `work`. Up to `max_num_of_itr` passes: for $i \in \{0, 1\}$ and each coset $j$, builds `local_vec`, skips if local syndrome is 0, else runs `GRSGaoDecoder.decode_to_code(local_vec)`. Catches `(DecodingError, ValueError)` and flags `uncorrectable_err[i]`. Terminates early if neither partition had uncorrectable errors and (`not changed_in_p1` or P0 is syndrome-clean), or if `not changed_in_itr` (1-step stall).
- **Complexity:** Time $O\!\left(I \sum_{i=0}^1 s_i (C_{\text{vec}} + m_i c_i) + D_{\text{dirty}} \cdot c^2\right)$ (Sage Gao is $O(c^2)$ per dirty coset); worst case $O(I_{\max} \cdot n \cdot c)$. Space $O(n + c^2)$ (`lagrange_polynomial`'s triangular table per local coset).
- **Sage primitives:** `vector(self._F, ...)` (51.3 µs/call, `Sequence`/`pbori` bug); `Matrix._matrix_times_vector_` ([`matrix0.pyx:5315`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/matrix/matrix0.pyx#L5315)); `GRSGaoDecoder.decode_to_code` ([`grs_code.py:1574-1629`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1574-L1629), pure-Python $O(c^2)$ `lagrange_polynomial` + pure-Python quadratic `_partial_xgcd` + NTL `ZZ_pEX` `quo_rem` + Python evaluation loop; also re-checks `local_vec in C` redundantly!).
- **Measured (`w=0` / `w=t_glob`):** CPU median (p10/p90): GF(16) `0.506` / `2.761` (1.98/2.96) ms; GF(64) `1.106` / `7.063` (6.03/11.77) ms; GF(256) `2.528` / `29.401` (25.53/33.17) ms; GF(1024) `10.962` / `217.271` (181.0/243.0) ms. Scaling $\alpha = 0.83$ ($w=0$), $\alpha = 1.24$ ($w=t_{\text{glob}}$, $q=64..1024$). Peak mem: `9.9 KiB` (GF(256)), `101.2 KiB` (GF(1024)).
- **Bottleneck (`cProfile` GF(256) `r=[6,6], d=[3,3], w=3`, 15 words, 0.977 s):** **43.0%** (`0.420 s`) in `Sequence`/`importlib` via `vector(F, ...)`, **34.2%** (`0.334 s`) in `GRSGaoDecoder` (`lagrange_polynomial` + `encode`), **~14%** in `Matrix._matrix_times_vector_` + Python loops. At higher $w$ (or larger $c=16,32$), `GRSGaoDecoder` grows to **75%–90%**.
- **Correctness risks:** (1) **Limit-cycle oscillation:** `elif not changed_in_itr: break` only catches 1-step fixed points; at high error weight ($w=24, 28$ on GF(64); $w=16, 20$ on GF(256)), misdecodes in P0 and P1 bounce symbols back and forth until `max_num_of_itr = 10` (observed in 15/200 words at GF(64) $w=28$). (2) When $k < k_{\max}$, local convergence can land in $C_{\max} \setminus C_k$ (valid locally in both partitions, invalid globally). (3) Does not track dirty cosets across passes (re-checks all $s_0+s_1$ cosets even when only 1–2 indices changed).
- **Better algorithm:** (1) Replace Sage `GRSGaoDecoder` with shifted-coset syndrome-based **Berlekamp–Massey + Chien + Forney** over GF($2^m$) int log-tables (**56×–109× faster** in pure Python; **>500×** in Cython/Numba/C); (2) Maintain a **dirty-coset worklist** (updating only the 1 coset in the opposite partition per modified coordinate) — reduces syndrome checks from $O(I \cdot s)$ to $O(s + \Delta)$; (3) Track `seen_hashes` to break 2-cycles immediately.
- **Sage-replacement difficulty:** `2/5` — Log-table GF($2^m$) arithmetic + BM + Chien + Forney (~120 LOC pure Python/NumPy, verified 200/200 bit-exact in `bench_04`).
- **Priority:** `P0` (primary workhorse decoder in simulations and TwoStep Stage 1).

---

### 3.2 `TamoBargIterativeEEDecoder`

#### Row 4 — `TamoBargIterativeEEDecoder.__init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10, erasure_start_itr: int | None = None) -> None`
- **Purpose:** Precompute local GRS codes, `ErrorErasure` decoders, local parity-check matrices, and `PolynomialRing(GF(q), 'x')` ([`tamo_barg_two_sets.py:614-665`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L614-L665)).
- **Algorithm:** Identical to `IterativeDecoder.__init__`, substituting `.decoder("ErrorErasure")` (which creates a `cartesian_product([ambient_space, GF(2)^c])` per coset) and caching `PolynomialRing(self._F, "x")` and `code.local_minimum_distance()`.
- **Complexity:** Time $O(n(c_0 + c_1))$; Space $O(\sum_i s_i m_i c_i)$.
- **Sage primitives:** Same as Row 1 plus `GRSErrorErasureDecoder.__init__` ([`grs_code.py:1811-1827`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1811-L1827), `cartesian_product` + `VectorSpace(GF(2), c)`) and `PolynomialRing`.
- **Measured:** CPU median (p10/p90): GF(16) `4.22` (4.00/4.51) ms; GF(64) `9.29` (8.83/10.36) ms; GF(256) `22.65` (21.60/24.40) ms; GF(1024) `73.46` (71.11/124.60) ms. Scaling $\alpha = 0.75$ ($q=64..1024$). Peak mem: `247.7 KiB` (GF(256)), `824.7 KiB` (GF(1024)).
- **Bottleneck:** $2s$ local GRS + `parity_check_matrix` + `cartesian_product` constructions.
- **Correctness risks:** Default `erasure_start_itr = max_num_of_itr // 2` couples erasure onset to `max_num_of_itr` (e.g., `max_num_of_itr=20` waits 10 wasted passes before marking erasures!).
- **Better algorithm:** Single canonical local code/PCM per partition (shared across all $s_i$ cosets) + dynamic stall-triggered erasure onset.
- **Sage-replacement difficulty:** `2/5` — Same as Row 1.
- **Priority:** `P2`.

#### Row 5 — `TamoBargIterativeEEDecoder._partition_is_syndrome_clean(self, part_idx: int, work: Any) -> bool`
- **Purpose:** Verify whether every coset in partition `part_idx` has zero local syndrome ([`tamo_barg_two_sets.py:671-677`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L671-L677)).
- **Algorithm:** Exact code duplicate of `TamoBargIterativeDecoder._partition_is_syndrome_clean` (Row 2).
- **Complexity:** Time $O(s_i(C_{\text{vec}} + m_i c_i))$; Space $O(c_i + m_i)$.
- **Sage primitives:** Identical to Row 2 (`vector(self._F, ...)` $\to$ `Sequence`/`pbori` + `Matrix._matrix_times_vector_`).
- **Measured (clean):** CPU median (p10/p90): GF(16) `0.237` (0.231/0.332) ms; GF(64) `0.529` (0.492/0.639) ms; GF(256) `1.372` (1.179/1.521) ms; GF(1024) `5.441` (5.320/6.003) ms. Scaling $\alpha = 0.84$ ($q=64..1024$). Peak mem: `4.3 KiB` (GF(256)), `46.8 KiB` (GF(1024)).
- **Bottleneck:** Identical to Row 2 (~80% `Sequence`/`pbori` failed import). **Extra call-site waste:** unlike `IterativeDecoder`, `IterativeEEDecoder.decode_to_code` calls this **unconditionally** at line 740 even when P1 modified zero symbols!
- **Correctness risks:** None.
- **Better algorithm:** Deduplicate into shared base helper; gate call with `if not changed_in_p1`; vectorize via M4RIE or NumPy GF($2^m$) log-table matmul.
- **Sage-replacement difficulty:** `1/5`.
- **Priority:** `P0`.

#### Row 6 — `TamoBargIterativeEEDecoder.decode_to_code(self, r: Any) -> tuple[tuple[Any, Any], int]`
- **Purpose:** Iteratively decode local errors and DUE-marked erasures across both partitions ([`tamo_barg_two_sets.py:679-766`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L679-L766)).
- **Algorithm:** Maintains `work` and binary `erasure_flags`. Each iteration snapshots `prev_work`, `prev_erasures`; for each coset: (a) if `era_wt == 0` and syndrome is 0, skip; (b) if `era_wt >= d_loc`, mark DUE; (c) if `era_wt == d_loc - 1`, recover via `interpolate_erasures`; (d) else call `GRSErrorErasureDecoder.decode_to_code((vec_r, vec_era))`. If both partitions have no DUE and P0 is syndrome-clean, clear flags and break. Once `num_itr >= erasure_start_itr`, mark all coordinates of DUE cosets in P0 as erased, then unmark all coordinates of clean/decoded cosets in P1. Breaks if stalled after `erasure_start_itr`.
- **Complexity:** Time $O\!\left(I \sum_i s_i(C_{\text{vec}} + m_i c_i) + D_{\text{dirty}} \cdot (C_{\text{GRS\_init}}(c) + c^2)\right)$; Space $O(n + c^2)$ (plus newly allocated `GeneralizedReedSolomonCode` + generator matrix per dirty coset!).
- **Sage primitives:** `GRSErrorErasureDecoder.decode_to_code` $\to$ `Decoder.decode_to_code` ([`decoder.py:225`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/decoder.py#L225)) $\to$ `GRSErrorErasureDecoder.decode_to_message` ([`grs_code.py:1880-1975`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1880-L1975), constructs brand-new `GeneralizedReedSolomonCode(C1)` + fresh `GRSGaoDecoder` + `_polynomial_vanishing_at_alphas` + `unencode` + `encode` on EVERY call, even for `era_wt == 0`!); `interpolate_erasures` ([`polynomials.py:259`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L259), calls pure-Python $O(c^2)$ `lagrange_polynomial`); `vector(self._F, ...)` AND `vector(GF(2), ...)` (BOTH hit `Sequence`/`pbori`!).
- **Measured (`w=0` / `w=t_glob`):** CPU median (p10/p90): GF(16) `0.942` / `8.971` (7.26/10.07) ms; GF(64) `1.890` / `17.192` (14.37/24.13) ms; GF(256) `4.555` / `57.795` (50.07/67.13) ms; GF(1024) `19.069` / `387.429` (337.4/401.3) ms. Scaling $\alpha = 0.83$ ($w=0$), $\alpha = 1.12$ ($w=t_{\text{glob}}$). Peak mem: `102.6 KiB` (GF(256)), `762.4 KiB` (GF(1024)) — **7.5×–10.4× higher than `IterativeDecoder`**!
- **Bottleneck (`cProfile` GF(256) `w=12`, 8 words, 2.640 s):** **62.3%** (`1.646 s`) inside `GRSErrorErasureDecoder` (of which `1.132 s` is uncached punctured Gao + `0.514 s` is rebuilding `C1` + `unencode`/`encode`), **30.6%** (`0.809 s`) in `Sequence`/`importlib` (1,946 calls to `vector(F, ...)` and `vector(GF(2), ...)`!).
- **Correctness risks:** (1) **Silent misdecode on `era_wt == d_loc - 1`:** `interpolate_erasures` interpolates through the $r_i$ surviving coordinates with zero parity checks left; if any unmarked error exists among those $r_i$ coordinates, it fabricates a wrong local codeword (observed as 4/200 misdecodes at GF(64) $w=28$, where `skipG = 38.5%` exceeded true success `36.5%`). (2) **Wasted stall passes:** `num_itr >= erasure_start_itr` forces 2–4 full no-op passes whenever hard decoding stalls at iter 1–2 (visible in GF(64) $w=20$ and GF(256) $w=12, 16, 20$ histograms: jump directly from iter 2/3 to iter 6!). (3) **Asymmetric DUE marking:** lines 750–757 mark DUE cosets ONLY from Partition 0 and unmark ONLY from Partition 1, ignoring P1-only DUE information.
- **Better algorithm:** (1) Use Forney-syndrome / modified-key-equation **BM + Chien + Forney** with erasure locator $\Lambda_{\text{era}}(x) = \prod_{j \in E}(1 - \alpha_j x)$ in GF($2^m$) int tables — handles $0 \le e \le d_{\text{loc}}-1$ uniformly with ZERO code reconstruction (**~150×–250× faster** than Sage `ErrorErasure`!); (2) Trigger DUE erasure marking immediately upon hard-decision stall (instead of fixed `erasure_start_itr`); (3) Skip `_partition_is_syndrome_clean(0, work)` when P1 made no changes.
- **Sage-replacement difficulty:** `2/5` — Extend Row 3's BM+Chien+Forney by ~40 LOC (multiply syndrome by $\Lambda_{\text{era}}(x)$, initialize BM with $\Lambda_{\text{era}}$, evaluate Forney over error+erasure roots).
- **Priority:** `P0` (largest per-call speedup potential among iterative decoders, plus huge accuracy advantage over hard Iterative!).

---

### 3.3 `TamoBargGlobalDecoder`

#### Row 7 — `TamoBargGlobalDecoder.__init__(self, code: TamoBargCodeTwoSets) -> None`
- **Purpose:** Construct the enveloping $[n, n - d_{\text{design}} + 1, d_{\text{design}}]$ global GRS code, Gao decoder, and global parity-check matrix ([`tamo_barg_two_sets.py:777-786`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L777-L786)).
- **Algorithm:** Queries `code.design_distance()`, builds `GeneralizedReedSolomonCode(evaluation_points, n - d_des + 1)`, attaches `GRSGaoDecoder`, and eagerly computes `self._global_pcm = _safe_parity_check_matrix(self._global_grs)`.
- **Complexity:** Time $O(n^2 + m_g n)$ (`multipliers_product()` does $n(n-1)$ field ops in pure Python at [`grs_code.py:416`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L416), plus building $m_g \times n$ Vandermonde matrix); Space $O(m_g n)$.
- **Sage primitives:** `GeneralizedReedSolomonCode`, `multipliers_product()` (pure Python $O(n^2)$), `dual_code().generator_matrix()` (`Matrix_gf2e_dense`).
- **Measured:** CPU median (p10/p90): GF(16) `0.547` (0.443/0.789) ms; GF(64) `1.307` (1.247/1.491) ms; GF(256) `8.249` (8.138/8.735) ms; GF(1024) `221.190` (218.24/223.00) ms. Scaling $\alpha = 1.85$ ($q=64..1024$, approaching $O(n^2)$). Peak mem: `33.2 KiB` (GF(256)), `264.1 KiB` (GF(1024)).
- **Bottleneck:** Pure-Python `multipliers_product()` ($O(n^2)$) inside `parity_check_matrix()` — dominates above $n=256$ (26.8× jump from $n=256$ to $n=1024$). Note: when $n = q$ (all of $\mathbb{F}_q$!), $G(x) = x^q - x \implies G'(\alpha_i) = -1 \implies \eta_i = -1$ for ALL $i$ in **$O(1)$** time!
- **Correctness risks:** Enveloping GRS uses `design_distance()`, which can be strictly smaller than true $d_{\min}$ of the LRC (conservative, never unsound, but under-utilizes true minimum distance).
- **Better algorithm:** For $n = q$ (all `["add", "add"]` and `["add", "mult"]` codes), $\eta_i = 1$ (char 2) trivially in $O(1)$; for $n = q-1$ (`["mult", "mult"]`), $\eta_i = -\alpha_i$ in $O(n)$; for arbitrary $S \subset \mathbb{F}_q$, compute $G'(x)$ via product tree + multipoint evaluation in $O(n \log^2 n)$.
- **Sage-replacement difficulty:** `1/5` — Closed-form $\eta_i$ for full/punctured fields + Vandermonde matrix.
- **Priority:** `P2`.

#### Row 8 — `TamoBargGlobalDecoder.decode_to_code(self, r: Any) -> Any`
- **Purpose:** Decode `r` globally via the enveloping GRS code's Gao decoder, with a global syndrome fast-path ([`tamo_barg_two_sets.py:792-804`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L792-L804)).
- **Algorithm:** Copies `r`, checks `(self._global_pcm * work).is_zero()` (returning `work` immediately if clean), else calls `self._global_decoder.decode_to_code(work)` (`GRSGaoDecoder`).
- **Complexity:** Clean path: Time $O(m_g n)$ via `Matrix._matrix_times_vector_`, Space $O(n + m_g)$. Dirty path: Theoretical Gao is $O(n \log^2 n)$, **Sage actual is $O(n^2)$ time and $O(n^2)$ space** (`lagrange_polynomial` allocates an $n(n+1)/2$ table of boxed field elements; `_partial_xgcd` runs up to $m_g/2$ quadratic Euclid steps; `encode` does $n(n - m_g)$ polynomial evaluations).
- **Sage primitives:** `Matrix._matrix_times_vector_` ([`matrix0.pyx:5315`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/matrix/matrix0.pyx#L5315)); `GRSGaoDecoder._decode_to_code_and_message` ([`grs_code.py:1574-1629`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1574-L1629)): redundant `r in C`, pure-Python `lagrange_polynomial` ([`polynomial_ring.py:2361`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/polynomial/polynomial_ring.py#L2361)), pure-Python `_partial_xgcd` calling NTL `Polynomial_ZZ_pEX.quo_rem` & `_mul_` ([`ntl_ZZ_pEX_linkage.pxi`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/libs/ntl/ntl_ZZ_pEX_linkage.pxi)), and Python loop `GRSEvaluationPolynomialEncoder.encode` ([`grs_code.py:1014`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1014)).
- **Measured (`w=0` / `w=t_glob`):** CPU median (p10/p90): GF(16) `0.018` / `1.757` (1.67/1.86) ms; GF(64) `0.063` / `18.715` (18.07/19.18) ms; GF(256) `0.346` / `367.059` (363.7/372.5) ms; GF(1024) `11.372` / `8,612.904` (8,486/8,622) ms. Scaling $\alpha = 1.87$ ($w=0$), $\alpha = 2.21$ ($w=t_{\text{glob}}$, $q=64..1024$). Peak mem: `341.1 KiB` (GF(256)), **`41,583 KiB` (40.6 MiB)** (GF(1024)).
- **Bottleneck (`cProfile` GF(256) `k=144, d_des=7, w=3`, 3 words, 4.027 s; and `bench_01` $n=1024, k=512$):** At $d_g \ll n$ (GF(256) $d_g=7$), **53.0%** `lagrange_polynomial` + **44.0%** `encode` (evaluating deg-249 poly at 256 points) + **2.0%** `_partial_xgcd`. At rate 1/2 ($n=1024, k=512$): **44.1%** `lagrange_polynomial`, **34.0%** `_partial_xgcd`, **19.2%** `encode`, **5.1%** redundant `r in C`.
- **Correctness risks:** (1) Clean fast-path checks membership in the **enveloping GRS code** (`_global_pcm * work == 0`), NOT the LRC subcode! A non-LRC codeword that lies in the enveloping GRS code passes through unmodified (safe for valid LRC codewords + errors within $t_{\text{glob}}$, because the LRC is a subcode of the enveloping GRS, but can mask misdecodes when used as Stage 2 of `TwoStep`!). (2) Does not return a tuple `(cw, num_itr)` — inconsistent API vs. Iterative/TwoStep (handled via adapter in `simulation/runner.py`).
- **Better algorithm:** (1) When $d_{\text{design}} \ll n$ (the standard LRC regime!), **syndrome-based Berlekamp–Massey + Chien + Forney** costs $O(n \cdot d_{\text{design}} + d_{\text{design}}^2)$ instead of $O(n^2)$ — **>100×–500× faster** and $O(n)$ memory (requires handling $\alpha=0$ by either punctured syndrome or 1-symbol shift/exclusion, or using the additive-FFT / affine-shift trick when $n < q$; for $n = q$, test $\alpha=0$ explicitly or use $e_0$ via $S_0$ sum-of-errors identity $\sum_{i=0}^{q-1} e_i = S_0$!); (2) When $d_{\text{design}} = \Theta(n)$, use **subproduct-tree interpolation + half-GCD + multipoint evaluation** over NTL/FLINT ($O(n \log^2 n)$, ~**17× faster** at $n=1024$, ~**60× faster** at $n=4096$).
- **Sage-replacement difficulty:** `2/5` for BM+Chien+Forney ($O(n d_g)$, ~150 LOC); `4/5` for full $O(n \log^2 n)$ half-GCD + subproduct tree (or use `python-flint` / Cython).
- **Priority:** `P0` (dominates standalone Global simulations and TwoStep fallback latency).

---

### 3.4 `TamoBargGlobalEEDecoder`

#### Row 9 — `TamoBargGlobalEEDecoder.__init__(self, code: TamoBargCodeTwoSets) -> None`
- **Purpose:** Initialize Cartesian input space `GF(q)^n x GF(2)^n`, enveloping global GRS code, `ErrorErasure` decoder, global PCM, and polynomial ring ([`tamo_barg_two_sets.py:810-823`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L810-L823)).
- **Algorithm:** Same as `GlobalDecoder.__init__` plus `cartesian_product([ambient_space, VectorSpace(GF(2), n)])`, `PolynomialRing(F, 'x')`, and `.decoder("ErrorErasure")`.
- **Complexity:** Time $O(n^2 + m_g n)$; Space $O(m_g n)$.
- **Sage primitives:** Same as Row 7 plus `cartesian_product` and `GRSErrorErasureDecoder.__init__`.
- **Measured:** CPU median (p10/p90): GF(16) `0.628` (0.535/0.764) ms; GF(64) `1.440` (1.382/1.688) ms; GF(256) `8.787` (8.282/9.088) ms; GF(1024) `218.293` (216.53/221.08) ms. Scaling $\alpha = 1.81$ ($q=64..1024$). Peak mem: `33.1 KiB` (GF(256)), `264.0 KiB` (GF(1024)).
- **Bottleneck:** Identical to Row 7 (`multipliers_product()` $O(n^2)$).
- **Correctness risks:** Same enveloping-GRS caveat as Row 7.
- **Better algorithm:** Closed-form $\eta_i$ for full/punctured fields (Row 7).
- **Sage-replacement difficulty:** `1/5`.
- **Priority:** `P2`.

#### Row 10 — `TamoBargGlobalEEDecoder.decode_to_code(self, word_and_erasure_vector: Any) -> Any`
- **Purpose:** Globally decode `(word, erasure_vector)` in `GF(q)^n x GF(2)^n` up to $2v + e < d_{\text{design}}$ ([`tamo_barg_two_sets.py:829-861`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L829-L861)).
- **Algorithm:** Copies `word` and `era_vec`, computes `era_wt`. Four branches: (1) `era_wt == 0` and global syndrome 0 $\to$ return `work`; (2) `era_wt >= d_glob` $\to$ raise `DecodingError`; (3) `era_wt == d_glob - 1` $\to$ Lagrange interpolation via `interpolate_erasures`; (4) else delegate to Sage `GRSErrorErasureDecoder.decode_to_code((work, era_vec))`.
- **Complexity:** Clean (`e=0`): $O(m_g n)$. Branch 3 (`e = d_g - 1`): $O((n - m_g)^2)$ time & space (`lagrange_polynomial`). Branch 4 (`0 <= e <= d_g - 2`): $O((n-e)^2 + k_g n)$ time & **$O((n-e)^2 + k_g n)$ space** (punctures code, builds brand-new GRS `C1`, recomputes `_polynomial_vanishing_at_alphas` in $O((n-e)^2)$, runs quadratic Gao, then builds `C1.generator_matrix()` to unencode and `C.generator_matrix()` to re-encode!).
- **Sage primitives:** `vector(self._F, ...)`, `vector(GF(2), ...)` (`Sequence`/`pbori`); `Matrix._matrix_times_vector_`; `interpolate_erasures` (`lagrange_polynomial`); `GRSErrorErasureDecoder.decode_to_code` ([`grs_code.py:1880-1975`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1880-L1975)).
- **Measured (`w=0` / `e=0, w=t_glob` / `e=d_g-1`):** CPU median: GF(16) `0.028` / `3.984` / `0.789` ms; GF(64) `0.076` / `31.280` / `8.407` ms; GF(256) `0.360` / `557.601` / `172.767` ms; GF(1024) `14.572` / **`16,321.338`** ms. Scaling ($e=0, w=t_{\text{glob}}$): $\alpha = 2.26$ ($q=64..1024$). Peak mem: `1,102.5 KiB` (GF(256)), **`155,531 KiB` (151.9 MiB!!)** (GF(1024)).
- **Bottleneck (`cProfile` GF(256) `e=0, w=3`, 3 words, 6.228 s):** **33.7%** (`2.100 s`) `lagrange_polynomial`, **27.9%** (`1.740 s`) `encode`, **24.6%** (`1.532 s`) uncached `_polynomial_vanishing_at_alphas` (recomputed from scratch every call!), **7.5%** (`0.467 s`) `C1.unencode` + `C.encode`!
- **Correctness risks:** (1) On `era_wt == d_glob - 1`, `interpolate_erasures` interpolates through the $n - d_{\text{glob}} + 1 = k_{\text{glob}}$ non-erased points **without checking if any non-erased symbol has an error** (cannot detect errors at $e = d_g - 1$, silently outputting a wrong codeword if $v \ge 1$). (2) When `era_wt == 0`, calling `GRSErrorErasureDecoder` instead of `GRSGaoDecoder` wastes **1.5×–1.9× CPU** and **3.7× RAM** for mathematically identical output!
- **Better algorithm:** (1) Immediate 1-line fix: dispatch `era_wt == 0` to a cached `GRSGaoDecoder` (1.5×–1.9× faster, 3.7× less RAM); (2) Full replacement: **Forney-syndrome BM + Chien + Forney** over GF($2^m$) int tables ($O(n \cdot d_g + d_g^2)$, **>200× faster** and $O(n)$ memory, zero code reconstruction!).
- **Sage-replacement difficulty:** `2/5` (BM+Chien+Forney with erasure locator).
- **Priority:** `P0`.

---

### 3.5 `TamoBargTwoStepDecoder`

#### Row 11 — `TamoBargTwoStepDecoder.__init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10) -> None`
- **Purpose:** Compose an inner `TamoBargIterativeDecoder` and outer `TamoBargGlobalDecoder` ([`tamo_barg_two_sets.py:867-871`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L867-L871)).
- **Algorithm:** Sequentially instantiates `TamoBargIterativeDecoder(code, max_num_of_itr)` and `TamoBargGlobalDecoder(code)`.
- **Complexity:** Time $O(n(c_0 + c_1) + n^2)$; Space $O(\sum_i s_i m_i c_i + m_g n)$.
- **Sage primitives:** Union of Rows 1 and 7.
- **Measured:** CPU median (p10/p90): GF(16) `5.017` (4.56/5.51) ms; GF(64) `9.977` (9.38/10.78) ms; GF(256) `30.188` (28.15/33.21) ms; GF(1024) `289.378` (284.0/361.2) ms. Scaling $\alpha = 1.21$ ($q=64..1024$, transitioning to Global's $O(n^2)$ above $q=256$). Peak mem: `283.0 KiB` (GF(256)), `1,084.3 KiB` (GF(1024)).
- **Bottleneck:** Local GRS construction at $q \le 256$; Global `multipliers_product()` ($O(n^2)$) at $q \ge 1024$ (76% of init at $q=1024$).
- **Correctness risks:** None.
- **Better algorithm:** Lazy-initialize the global decoder on first Stage-1 stall (in low-noise regimes Stage 2 is never invoked!) + canonical local PCM + closed-form global $\eta_i$.
- **Sage-replacement difficulty:** `2/5`.
- **Priority:** `P2`.

#### Row 12 — `TamoBargTwoStepDecoder.decode_to_code(self, r: Any) -> tuple[Any, int]`
- **Purpose:** Run iterative local decoding, then apply global GRS cleanup (catching failures gracefully) ([`tamo_barg_two_sets.py:877-891`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L877-L891)).
- **Algorithm:** Calls `tmp, num_itr = self._iter_decoder.decode_to_code(r)`, then `self._global_decoder.decode_to_code(tmp)`; on `(DecodingError, ValueError)`, falls back to returning `tmp`.
- **Complexity:** Time $T_{\text{Iter}} + O(m_g n) + \mathbb{I}[\text{global syndrome} \ne 0] \cdot O(n^2)$; Space $O(n + c^2)$ if Stage 2 skipped, $O(n^2)$ if Stage 2 executes.
- **Sage primitives:** Union of Rows 3 and 8.
- **Measured (`w=0` / `w=t_glob`):** CPU median (p10/p90): GF(16) `0.520` / `2.864` (2.03/3.03) ms; GF(64) `1.226` / `7.498` (6.30/11.62) ms; GF(256) `3.069` / `29.684` (25.78/34.65) ms; GF(1024) `22.138` / `235.184` (197.3/437.8) ms. Scaling (Stage-1 finish): $\alpha = 1.24$ ($q=64..1024$); on fallback: $\alpha = 2.21$. Peak mem: `85.3 KiB` (GF(256)), `4,111 KiB` (GF(1024) — notice the 4 MiB spike just from `self._global_pcm * tmp`!).
- **Bottleneck:** When Stage 1 finishes: identical to `IterativeDecoder` (43% `Sequence`/`pbori`, 34%+ local Gao) + `Global` syndrome check (`11.4 ms` and `4 MiB` at $q=1024$!). When Stage 1 stalls: **85%–97%** in length-$n$ `GRSGaoDecoder` (208 ms at GF(256), 8,613 ms at GF(1024)).
- **Correctness risks:** (1) On `w=0` or whenever Stage 1 succeeds on a `k == max_dimension` code, the global syndrome check is **mathematically redundant** (proven in §4.4). (2) When `k < max_dimension`, Stage 1 can theoretically misdecode into $C_{\max} \setminus C_k$; `GlobalDecoder` catches this only because $H_{\text{glob}}$ checks degree $< n - d_{\text{design}} + 1$, NOT full LRC membership!
- **Better algorithm:** (1) When `code.dimension() == code.max_dimension()` and Stage 1 returns a locally clean flag (`converged=True`), **skip `_global_decoder` completely** (saves 11.4 ms & 4 MiB at $q=1024$ on every clean/converged word!); (2) Apply Row 3 + Row 8 fast BM+Chien+Forney replacements.
- **Sage-replacement difficulty:** `2/5`.
- **Priority:** `P0`.

---

### 3.6 `TamoBargTwoStepEEDecoder`

#### Row 13 — `TamoBargTwoStepEEDecoder.__init__(self, code: TamoBargCodeTwoSets, max_num_of_itr: int = 10, erasure_start_itr: int | None = None) -> None`
- **Purpose:** Compose an inner `TamoBargIterativeEEDecoder` and outer `TamoBargGlobalEEDecoder` ([`tamo_barg_two_sets.py:902-911`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L902-L911)).
- **Algorithm:** Sequentially instantiates `TamoBargIterativeEEDecoder` and `TamoBargGlobalEEDecoder`.
- **Complexity:** Time $O(n(c_0 + c_1) + n^2)$; Space $O(\sum_i s_i m_i c_i + m_g n)$.
- **Sage primitives:** Union of Rows 4 and 9.
- **Measured:** CPU median (p10/p90): GF(16) `4.962` (4.80/5.15) ms; GF(64) `11.374` (10.39/12.80) ms; GF(256) `30.514` (29.57/32.08) ms; GF(1024) `291.334` (291.15/387.16) ms. Scaling $\alpha = 1.17$ ($q=64..1024$). Peak mem: `283.0 KiB` (GF(256)), `1,078.9 KiB` (GF(1024)).
- **Bottleneck:** Same as Row 11 (Global `multipliers_product()` dominates at $q \ge 1024$).
- **Correctness risks:** Same `erasure_start_itr` coupling as Row 4.
- **Better algorithm:** Lazy Stage-2 init + canonical local PCM + closed-form global $\eta_i$.
- **Sage-replacement difficulty:** `2/5`.
- **Priority:** `P2`.

#### Row 14 — `TamoBargTwoStepEEDecoder.decode_to_code(self, r: Any) -> tuple[Any, int, int]`
- **Purpose:** Run iterative EE decoding, then apply global EE cleanup on `(word_after_iter, era_vec)` ([`tamo_barg_two_sets.py:917-932`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L917-L932)).
- **Algorithm:** Calls `(word_after_iter, era_vec), num_itr = self._iter_ee_decoder.decode_to_code(r)`, records `num_rem_era = int(era_vec.hamming_weight())`, then invokes `self._global_ee_decoder.decode_to_code((word_after_iter, era_vec))` with fallback to `word_after_iter` on `(DecodingError, ValueError)`.
- **Complexity:** Time $T_{\text{IterEE}} + O(n) + \mathbb{I}[e_{\text{rem}} = 0] O(m_g n) + \mathbb{I}[e_{\text{rem}} \in (0, d_g) \lor \text{syn} \ne 0] \cdot O((n - e_{\text{rem}})^2 + k_g n)$; Space up to $O(n^2)$ on fallback.
- **Sage primitives:** Union of Rows 6 and 10 (plus `era_vec.hamming_weight()`).
- **Measured (`w=0` / `w=t_glob`):** CPU median (p10/p90): GF(16) `0.993` / `9.025` (7.27/10.71) ms; GF(64) `2.056` / `17.846` (14.71/25.37) ms; GF(256) `4.977` / `59.905` (50.69/69.64) ms; GF(1024) `31.673` / `420.302` (353.0/518.2) ms. Scaling (Stage-1 finish): $\alpha = 1.14$ ($q=64..1024$); on fallback: $\alpha = 2.26$. Peak mem: `154.6 KiB` (GF(256)), `4,084 KiB` (GF(1024), Stage-1 finish) / `151.9 MiB` (fallback).
- **Bottleneck:** Stage 1 `GRSErrorErasureDecoder` rebuilds (62%) + `Sequence`/`pbori` (30%) when Stage 1 finishes; Stage 2 `GRSErrorErasureDecoder` (558 ms at GF(256), 16.3 s at GF(1024)) when Stage 2 fires!
- **Correctness risks:** Inherits Row 6's `era_wt == d_loc - 1` silent misdecode risk and Row 10's `era_wt == d_glob - 1` silent misdecode risk. Note: if Stage 1 terminates with `num_rem_era >= d_glob`, `GlobalEEDecoder` immediately raises `DecodingError` in $O(n)$ time (cheap!), EXCEPT `GlobalEEDecoder` still copies `word` and `era_vec` via `vector()` first.
- **Better algorithm:** Skip Stage 2 when `k == max_dim` and Stage 1 converged cleanly; short-circuit before Stage 2 if `num_rem_era >= d_glob`; replace both Stage 1 and Stage 2 with unified GF($2^m$) Forney-syndrome BM+Chien+Forney (**>100×–300×** faster!).
- **Sage-replacement difficulty:** `2/5`.
- **Priority:** `P0` (best overall decoding performance of all 6 decoders; making it fast unlocks large-scale Monte Carlo sweeps).

#### Thin Wrapper Methods (accounted for per `shared_context.md`)
- **`decode_to_message(self, r)`** on `Iterative` (L605), `IterativeEE` (L768), `TwoStep` (L893), `TwoStepEE` (L934): each simply calls `self.decode_to_code(r)` and `self.code().unencode(corr_c)`. Complexity = $T_{\text{decode\_to\_code}} + T_{\text{unencode}}$ (where `unencode` solves a $k \times n$ linear system over `GF(q)` via cached generator-matrix information-set inversion, plus a `corr_c in C` check unless `nocheck=True`). Note: `Global` and `GlobalEE` inherit `Decoder.decode_to_message`, which is NOT overridden — calling `GlobalDecoder.decode_to_message(r)` invokes `AbstractLinearCode.unencode` on the LRC's `VectorEncoder` after `decode_to_code(r)`.
- **`_repr_(self)`** on all 6 classes (L549, L667, L788, L825, L873, L913): $O(1)$ string formatting; zero performance relevance.

---

## 4. Deep-Dive Answers to Specific Questions

### 4.1 Local Decoding per Coset: Berlekamp–Welch vs. Gao vs. ErrorErasure vs. KeyEquationSyndrome vs. Lagrange

Measured CPU median ms/coset (20 repeats after 3 warm-ups, non-zero evaluation points so `KeyEquationSyndrome` is also eligible; [`bench_01_local_and_global_grs.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_01_local_and_global_grs.py)):

| Coset Config $(q, c, r, d_{\text{loc}}, t_{\text{loc}})$ | Err / Era | `Gao` (ms) | `BerlekampWelch` (ms) | `ErrorErasure` (ms) | `KeyEqSyndrome` (ms) | `interp_erasures` (ms) | **Our Pure-Py BM+Chien+Forney** |
|---|---|---|---|---|---|---|---|
| GF(64), $c=4, r=2, d=3, t=1$ | $w=0$ / $w=1$ / $w=2$ (fail) | 0.125 / 0.578 / 0.303 | 0.104 / 1.082 / 0.721 | 1.399 / 1.842 / 1.151 | **0.024** / **0.445** / 0.407 | 0.094 ($e=2$) | — |
| GF(64), $c=8, r=4, d=5, t=2$ | $w=0$ / $w=2$ / $w=3$ (fail) | 0.225 / 1.191 / 0.802 | 0.209 / 1.466 / 0.996 | 1.726 / 2.843 / 1.831 | **0.026** / **0.822** / **0.696** | 0.310 ($e=4$) | **0.0178 ms (61.9× vs Gao)** |
| GF(256), $c=8, r=4, d=5, t=2$ | $w=0$ / $w=2$ / $w=3$ (fail) | 0.260 / 1.319 / 0.856 | 0.245 / 1.425 / 0.986 | 1.991 / 3.028 / 2.074 | **0.027** / **0.957** / **0.851** | 0.344 ($e=4$) | — |
| GF(256), $c=16, r=8, d=9, t=4$ | $w=0$ / $w=4$ / $w=5$ (fail) | 0.615 / 3.492 / 2.644 | 0.581 / **2.451** / **1.481** | 3.176 / 6.325 / 4.691 | **0.034** / **2.153** / 1.843 | 1.001 ($e=8$) | **0.0598 ms (55.8× vs Gao)** |
| GF(256), $c=16, r=12, d=5, t=2$ | $w=0$ / $w=2$ / $w=3$ (fail) | 1.037 / 3.297 / 2.037 | 1.009 / 2.451 / 1.406 | 3.865 / 5.873 / 4.062 | **0.031** / **1.320** / **1.150** | 1.157 ($e=4$) | **0.0281 ms (109.2× vs Gao)** |
| GF(1024), $c=32, r=16, d=17, t=8$ | $w=0$ / $w=8$ / $w=9$ (fail) | 2.008 / 13.975 / 10.527 | 1.969 / **6.774** / **3.511** | 9.762 / 22.967 / 17.047 | **0.103** / 7.216 / 5.982 | 3.370 ($e=16$) | — |

**Key algorithmic & Sage-constant insights:**
1. **Why `BerlekampWelch` ($O(c^3)$) beats `Gao` (Sage $O(c^2)$) for $16 \le c \le 256$:** `GRSBerlekampWelchDecoder` ([`grs_code.py:1235-1239`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1235-L1239)) delegates its $O(c^3)$ kernel solve to **C-level M4RIE** (`Matrix_gf2e_dense.right_kernel()`), whereas `GRSGaoDecoder` runs **pure-Python** `divided_difference` ($O(c^2)$) + pure-Python `_partial_xgcd` + Python point evaluation. At $c=32$, BW (`6.77 ms`) is **2.06× faster** than Gao (`13.98 ms`), and on failure ($w=9$) **3.00× faster** (`3.51 ms` vs `10.53 ms`).
2. **Why `Gao` at $w=0$ is 18×–33× slower than `KeyEqSyndrome`:** Both check `r in C`, but `Gao._decode_to_code_and_message` ([`grs_code.py:1613-1614`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1613-L1614)) still calls `self.connected_encoder().unencode_nocheck(r)` (running `lagrange_polynomial`!) even when `decode_to_code` discards the message!
3. **Why `KeyEqSyndrome` isn't used in `tamo_barg_two_sets.py` — and how to unlock it:** `GRSKeyEquationSyndromeDecoder.__init__` ([`grs_code.py:2063`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L2063)) forbids $0 \in \text{coset}$ (which occurs in coset 0 of every additive subgroup). Yet any affine shift $\alpha_j' = \alpha_j + \delta$ ($\delta \notin \text{coset}$) preserves $\text{RS}(\boldsymbol{\alpha}, r)$ identically (since polynomials of degree $< r$ are translation-invariant) and eliminates $\alpha = 0$! Using this shift with pure-Python GF($2^m$) log-table Berlekamp–Massey + Chien + Forney ([`bench_04`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_04_syndrome_and_batch_prototypes.py)) achieves **55.8×–109.2× speedup** over Sage Gao with **200/200 bit-exact agreement**.
4. **Why `ErrorErasure` is 1.8×–3.2× slower than `Gao`:** Even at $e=0$, [`grs_code.py:1971-1975`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1971-L1975) constructs a fresh `GeneralizedReedSolomonCode(C1)`, recomputing `_polynomial_vanishing_at_alphas` ($O(c^2)$), `C1.unencode`, and `C.encode` every single call.

### 4.2 Iterative Convergence, Iteration Distribution, Stall/Oscillation & Expander Guarantee

**Expander structure vs. observed decoding radius:**
- **GF(64) Suite A** ($n=64, k=16, r=[4,4], d_{\text{loc}}=[5,5], t_{\text{loc}}=2, d_{\text{des}}=13, t_{\text{glob}}=6$): $|H_0|=|H_1|=8$, $H_0 + H_1 = \mathbb{F}_{64}$, so the bipartite coset graph is the complete bipartite graph $K_{8,8}$ (`components=1`, $\lambda_1 = 8.0$, $\lambda_2 = 0.0$, spectral gap $8.0$, algebraic connectivity $8.0$).
  - *Worst-case / expander guarantee:* On $K_{c,c}$ ($\lambda_2 = 0$), the Tanner/Sipser–Spielman worst-case hard-decision guarantee is determined by the minimum $(t_{\text{loc}}+1)$-core (every vertex in the error subgraph has degree $\ge t_{\text{loc}}+1$): a $(t_{\text{loc}}+1) \times (t_{\text{loc}}+1) = 3 \times 3$ subgrid of **weight 9** (guaranteed correct for all patterns of weight $\le 8$, i.e. $1.33\times t_{\text{glob}}$). For `IterativeEE`, DUE marking breaks any $(t_{\text{loc}}+1) \times (t_{\text{loc}}+1)$ stopping set when $t_{\text{loc}} + 1 \le d_{\text{loc}} - 1$ (since $3 \le 4$, Lagrange/EE recovers it!), raising the minimum stopping set to $d_{\text{loc}} \times d_{\text{loc}} = 5 \times 5$ (weight 25 for pure erasures).
  - *Observed random-error performance:* `Iterative` achieves **100.0%** up to $w = 16$ ($2.67\times t_{\text{glob}}$), **97.5%** at $w=20$, **80.0%** at $w=24$ ($4.0\times t_{\text{glob}}$). `IterativeEE` achieves **100.0%** at $w=20$, **96.5%** at $w=24$, and **36.5%** vs. **6.5%** at $w=28$!
- **GF(256) Suite B** ($n=256, k=144, r=[6,6], d_{\text{loc}}=[3,3], t_{\text{loc}}=1, d_{\text{des}}=7, t_{\text{glob}}=3$): $|H_0|=|H_1|=8$, $|H_0 + H_1| = 64 < 256$, so the coset graph splits into **4 disconnected copies of $K_{8,8}$** (`components=4`, $\lambda_1 = \lambda_2 = 8.0$, spectral gap `0.0`, algebraic connectivity `0.0`).
  - *Worst-case guarantee:* Global spectral expansion is 0 (disconnected), and each $K_{8,8}$ component has minimum hard stopping set $2 \times 2$ of **weight 4** (guaranteed for $w \le 3 = t_{\text{glob}}$). Yet for `IterativeEE`, a $2 \times 2$ stopping set produces $2 = d_{\text{loc}} - 1$ erasures per row, which Lagrange interpolation resolves completely!
  - *Observed random-error performance:* `Iterative` achieves **100.0%** through $w = 10$, **99.5%** at $w=12$, **88.5%** at $w=20$ ($6.67\times t_{\text{glob}}$); `IterativeEE` achieves **100.0%** all the way through $w = 20$!

**Iteration-count distributions (200 words/weight, `max_num_of_itr=10`, `erasure_start_itr=5`):**

| Code & Error Weight | `IterativeDecoder` `{itr: count}` | `IterativeEEDecoder` `{itr: count}` | Diagnostic Observation |
|---|---|---|---|
| GF(64) $w=0..2$ | `{1: 200}` | `{1: 200}` | Single pass; zero retries |
| GF(64) $w=6$ ($=t_{\text{glob}}$) | `{1: 165, 2: 35}` | `{1: 165, 2: 35}` | 82.5% finish in 1 pass, 100% by pass 2 |
| GF(64) $w=16$ | `{2: 195, 3: 5}` | `{2: 195, 3: 5}` | 100% success in $\le 3$ passes |
| GF(64) $w=20$ | `{2: 161, 3: 37, 4: 2}` | `{2: 158, 3: 35, 4: 2, 6: 5}` | 5 words stall early in `Iter`, wait idly until iter 5 in `IterEE`, solve at iter 6! |
| GF(64) $w=24$ | `{2: 37, 3: 128, 4: 30, 5: 2, 10: 3}` | `{2: 27, 3: 105, 4: 26, 5: 2, 6: 33, 7: 6, 8: 1}` | 3 words oscillate to `max_itr=10` in `Iter`; `IterEE` resolves 33 more words at iter 6–8 |
| GF(64) $w=28$ | `{1: 1, 2: 61, 3: 88, 4: 32, 5: 3, 10: 15}` | `{3: 4, 4: 8, 5: 1, 6: 159, 7: 28}` | 15/200 hit `max_itr=10` (limit cycles) in `Iter`; 159/200 in `IterEE` burn 3–4 wasted passes waiting for `erasure_start_itr=5`! |
| GF(256) $w=3$ ($=t_{\text{glob}}$) | `{1: 188, 2: 12}` | `{1: 188, 2: 12}` | 94% finish in 1 pass |
| GF(256) $w=16$ | `{1: 3, 2: 190, 3: 5, 10: 2}` | `{1: 3, 2: 182, 3: 4, 6: 9, 7: 2}` | 2 limit-cycle oscillations in `Iter`; all 11 stalled words rescued at iter 6–7 in `IterEE` |
| GF(256) $w=20$ | `{1: 1, 2: 179, 3: 15, 4: 2, 10: 3}` | `{1: 1, 2: 168, 3: 8, 6: 20, 7: 3}` | 23 stalled words rescued at iter 6–7 in `IterEE` (100% total success) |

### 4.3 Global/TwoStep: Length-$n$ Gao $O(n \log^2 n)$ Theory vs. Sage Actual ($n = 16 \dots 4096$)

Measured on rate-$1/2$ GRS codes $[n, n/2, n/2+1]$ over $\text{GF}(n)$ at full decoding radius $t = n/4$ ([`bench_01`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_01.txt) & [`bench_05`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_05.txt)):

| $n = q$ | Total `Gao` CPU ms (p10/p90) | `in_C` ms (%) | `lagrange` ms (%) | `_partial_xgcd` ms (%) | `quo_rem` ms | `encode` ms (%) | Peak Mem | `BerlekampWelch` ms | NTL `ZZ_pEX` `mul` / `quo_rem` (deg $n$) |
|---|---|---|---|---|---|---|---|---|---|
| 16 | `1.58` (1.52/1.61) | 0.053 (3.4%) | 0.862 (54.7%) | 0.265 (16.8%) | 0.033 | 0.546 (34.7%) | 5.3 KiB | 1.73 | 0.169 / 0.113 ms |
| 64 | `23.49` (21.98/24.47) | 0.160 (0.7%) | 9.746 (41.5%) | 6.138 (26.1%) | 2.075 | 4.798 (20.4%) | 29.7 KiB | **11.06** | 0.259 / 2.310 ms |
| 256 | `435.48` (430.7/446.7) | 1.693 (0.4%) | 191.46 (44.0%) | 148.61 (34.1%) | 4.051 | 91.34 (21.0%) | 339 KiB | **121.51** | 1.552 / 7.952 ms |
| 1024 | `10,231.1` (10,089/10,458) | 517.0 (5.1%) | 4,512.3 (44.1%) | 3,483.5 (34.0%) | 19.11 | 1,966.6 (19.2%) | **40.6 MiB** | — | 8.589 / 41.075 ms |
| 4096 | *~240,000 ms (extrap., $\alpha=2.28$)* | — | *~105,000 ms* | *~82,000 ms* | — | *~46,000 ms* | *~660 MiB* | — | **46.13 / 218.71 ms** |
| **Exponent $\alpha$ ($64..1024$ / $4096$)** | **2.19** | — | **2.21** | **2.29** | — | **2.17** | **$O(n^2)$** | — | **1.24 / 1.10** |

**Diagnosis & Half-GCD / Fast-Multiplication Gain:**
- **Backend verified:** `PolynomialRing(GF(2^e), 'x')` uses NTL's `ZZ_pEX` (`Polynomial_ZZ_pEX`, [`polynomial_zz_pex.pyx:69`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/polynomial/polynomial_zz_pex.pyx#L69)), whose C++ multiplication and division are **already sub-quadratic $O(n^{1.24})$** (`8.59 ms` at deg 1024, `46.13 ms` at deg 4096).
- **Why Sage Gao is $O(n^{2.19})$ instead of $O(n \log^2 n)$:** All three dominant stages are pure-Python quadratic loops wrapped around NTL: (1) `lagrange_polynomial` ([`polynomial_ring.py:2347-2538`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/polynomial/polynomial_ring.py#L2347-L2538)) runs $n(n-1)/2$ Python divisions and stores an $O(n^2)$ triangular table; (2) `_partial_xgcd` ([`grs_code.py:1567-1570`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1567-L1570)) runs $\approx n/2$ quadratic Euclid steps (and even discards `quo_rem`'s remainder to recompute `prev_r - q * r`!); (3) `encode` ([`grs_code.py:1014`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1014)) evaluates $h(\alpha_i)$ point-by-point ($O(nk)$).
- **Expected gain from $O(n \log^2 n)$ fast interpolation + half-GCD + multipoint eval (on top of NTL/FLINT $T_{\text{mul}}(n)$):** Total work is $\approx 12\text{–}18 \cdot \log_2(n) \cdot T_{\text{mul}}(n/2)$, yielding $\approx 0.5\text{–}0.6\text{ s}$ at $n=1024$ (**~17× speedup**, RAM $40.6\text{ MiB} \to <1\text{ MiB}$) and $\approx 4\text{ s}$ at $n=4096$ (**~60× speedup**, RAM $660\text{ MiB} \to <4\text{ MiB}$). Even better, in the LRC regime ($d_{\text{design}} \ll n$), **syndrome-based BM+Chien+Forney** runs in $O(n \cdot d_{\text{design}})$ without interpolation at all (**~100×–500× speedup** at $n=256..1024$!).

### 4.4 TwoStep: Stage-1 Finish Fraction, Syndrome-Gating Savings & Skipping Global Decode

1. **Stage-1 finish fraction (`skipG%`):** On GF(64) ($t_{\text{glob}}=6$), Stage 1 alone finishes **100.0%** of words for $w = 0 \dots 16$, **97.5%** at $w=20$, **80.0%** at $w=24$ (`TwoStepEE`: **100%** at $w=20$, **96.5%** at $w=24$). On GF(256) ($t_{\text{glob}}=3$), Stage 1 finishes **100.0%** for $w = 0 \dots 10$, **99.5%** at $w=12$, **88.5%** at $w=20$ (`TwoStepEE`: **100.0%** for all $w = 0 \dots 20$).
2. **Cost saved by global syndrome gating:** Checking `(self._global_pcm * work).is_zero()` saves **297×** at GF(64) (`0.063 ms` vs `18.72 ms`), **1,061×** at GF(256) (`0.346 ms` vs `367.06 ms`), and **757×** at GF(1024) (`11.37 ms` vs `8,612.9 ms`) on every word Stage 1 already solved!
3. **When can TwoStep skip even the global syndrome check?** Proven and verified in [`bench_04`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_04.txt) Part 3:
   - **Theorem ($k = k_{\max}$):** For GF(64) `r=[4,4], d=[5,5], k=16=max_dim`, $\ker(H_{\text{local},0}) \cap \ker(H_{\text{local},1})$ has dimension **16 = $k_{\max}$**, and **16/16** basis vectors satisfy $H_{\text{glob}} v = 0$. Therefore, when `k == max_dimension` and Stage 1 terminates locally syndrome-clean in both partitions, **global syndrome is guaranteed zero** — TwoStep can skip `GlobalDecoder` completely (saving `11.4 ms` and `4 MiB` at $q=1024$)!
   - **Counterexample ($k < k_{\max}$):** For GF(64) `k=8 < max_dim=16`, $\ker(H_{\text{local},0}) \cap \ker(H_{\text{local},1})$ STILL has dimension **16**, so 8 extra dimensions satisfy all local checks while **violating** $H_{\text{glob}} v = 0$! Hence when $k < k_{\max}$, TwoStep MUST retain the global syndrome check.
4. **When does TwoStep Stage 2 actually rescue stalled words?** When $k = k_{\max}$ (`d_design = 13, t_glob = 6` on GF(64); `d_design = 7, t_glob = 3` on GF(256)), the minimum iterative stopping set has weight $(t_{\text{loc}}+1)^2 = 9 > t_{\text{glob}}=6$ (and $4 > t_{\text{glob}}=3$), so **`Gfix = 0`** across all 5,400 tested words! Conversely, on subcode **GF(64) $k=8 < k_{\max}=16$** ($d_{\text{design}} = 45, t_{\text{glob}} = 22 > 9$, [`bench_05`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_05.txt)), Stage 2 rescues stalled words dramatically:

| GF(64) $k=8$ ($t_{\text{glob}}=22$), 100 words/wt | `Iter` | `IterEE` | `TwoStep` (`skipG%`, `Gfix`) | `TwoStepEE` (`skipG%`, `Gfix`) | `Global` alone |
|---|---|---|---|---|---|
| $w = 18$ | 97% | 100% | **100%** (97%, +3) | **100%** (100%, 0) | 100% |
| $w = 22$ ($=t_{\text{glob}}$) | 92% | 98% | **100%** (92%, +8) | **100%** (98%, +2) | 100% |
| $w = 24$ ($t_{\text{glob}}+2$) | 77% | 97% | **100%** (77%, +23) | **100%** (97%, +3) | 0% |
| $w = 26$ ($t_{\text{glob}}+4$) | 47% | 78% | **97%** (47%, +50) | **99%** (78%, +21) | 0% |
| $w = 28$ ($t_{\text{glob}}+6$) | 9% | 47% | **81%** (9%, +72) | **90%** (47%, +43) | 0% |

### 4.5 EE Variants: Erasure Handling Cost

- **`interpolate_erasures` (`polynomials.py:259`):** Uses `poly_ring.lagrange_polynomial(known_pts)` (pure-Python $O(r^2)$ divided differences) + $e$ evaluations. Locally it is **3.3×–6.0× faster** than `GRSErrorErasureDecoder` (`0.31 ms` vs `1.80 ms` at $c=8$; `1.00 ms` vs `2.61 ms` at $c=16$; `3.37 ms` vs `6.10 ms` at $c=32$), and globally at $e = d_g - 1$ it is **3.2× faster** (`172.8 ms` vs `557.6 ms` at $q=256$) — though still $O(k_g^2)$ in pure Python.
- **`GRSErrorErasureDecoder` penalty:** Reconstructing `GeneralizedReedSolomonCode(C1)` + uncached `_polynomial_vanishing_at_alphas` + `unencode`/`encode` adds **+97%** runtime to `IterativeEE` vs `Iterative` at $q=256$ (`57.8 ms` vs `29.4 ms`) and **+52% to +90%** to `GlobalEE` vs `Global` (`557.6 ms` vs `367.1 ms` at $q=256$; `16.32 s` vs `8.61 s` at $q=1024$).
- **Additional `IterativeEE` overheads:** `vector(GF(2), era_list)` triggers the same 51 µs `Sequence`/`pbori` bug per dirty coset; unconditional `_partition_is_syndrome_clean(0, work)` adds 1.37 ms/word on GF(256) clean words; fixed `erasure_start_itr = max_itr // 2` adds 2–4 wasted no-op passes on stalled words.

### 4.6 Syndrome Check Cost & Batch Decoding Throughput

Measured local coset syndrome check (`H_loc * v` for $m \times c$ over $\text{GF}(q)$, [`bench_04`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_04.txt)):

| $q$ | $H_{\text{loc}}$ shape | Sage `vec+H*v` (µs) | Sage `H*v` only (µs) | NumPy log-tbl $B=1$ (µs) | Sage M4RIE batch $B=256$ (µs/coset) | NumPy log-tbl batch $B=256$ (µs/coset) |
|---|---|---|---|---|---|---|
| 16 | $2 \times 4$ | 67.54 | 4.48 | 39.84 | **0.118** (**573×**) | 0.233 (**290×**) |
| 64 | $4 \times 8$ | 72.61 | 6.80 | 76.02 | **0.319** (**228×**) | 0.517 (**141×**) |
| 256 | $8 \times 16$ | 81.74 | 13.31 | 146.81 | **0.574** (**142×**) | 1.407 (**58×**) |
| 1024 | $16 \times 32$ | 149.25 | 77.53 | 288.22 | 6.436 (**23×**) | **4.078** (**37×**) |
| 4096 | $32 \times 64$ | 1,438.14 | 379.53 | 615.61 | 43.042 (**33×**) | **13.564** (**106×**) |

*(Note: M4RIE wins for $q \le 256$ via packed bit-sliced Newton–John tables, then slows down for $q \ge 1024$ because table size scales as $2^e$; NumPy log-table matmul scales cleanly and beats M4RIE by **3.2×** at $q=4096$. For $B=1$, NumPy's per-call Python dispatch overhead (~35 µs) exceeds a plain Python int-list log-table loop (~5 µs).)*

**Fixed Python/Sage Overhead & Achievable Batched Throughput:**
- On GF(256) `r=[6,6], d=[3,3], w=3`, `cProfile` shows **~43% fixed Sage `Sequence`/`pbori` overhead** + **~34% pure-Python Gao wrapper overhead** + **~14% generic `Matrix._matrix_times_vector_` unpacking** = **>90% pure framework overhead**!
- Current `IterativeDecoder` throughput at GF(256) (`r=[8,8], d=[9,9], w=12`): `29.4 ms/word` $\approx$ **34 words/s** (and `TwoStep` with Global fallback: ~3 words/s).
- With **batched NumPy GF($2^m$) syndromes** (`1.4 µs/coset` $\implies 45\text{ µs}$/word for all 32 cosets) + **pure-Python/Cython BM+Chien+Forney** (`59.8 µs/dirty coset` in pure Python, ~`2–5 µs` in Cython/Numba/C):
  - **Pure-Python + NumPy batched Iterative:** `~0.6–1.0 ms/word` (**1,000–1,600 words/s**, **~30×–50× speedup**);
  - **Cython/Numba/C batched Iterative:** `~0.05–0.12 ms/word` (**8,000–20,000 words/s**, **~250×–600× speedup**);
  - **IterativeEE** gains an extra **2×** from eliminating `GRSErrorErasureDecoder` code rebuilds (**~100×–1,000×** total).

---

## 5. Correctness: Empirical Decoding Radius vs. Theoretical ($0 \dots t+2$ and Waterfall)

All trials seeded, **200 random codewords + fixed-weight errors per weight** ([`out_03_gf64_full.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf64_full.txt), [`out_03_gf256.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf256.txt), [`out_03_gf256_part2.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf256_part2.txt)).

### Table 5A — GF(64): $n=64, k=16, r=[4,4], d_{\text{loc}}=[5,5], d_{\text{design}}=13 \implies t_{\text{glob}}=6$

| Weight $w$ | `Iter` % (`avgI`) | `IterEE` % (`avgI`, `remE`) | `Global` % | `GlobalEE` ($e=0$) % | `TwoStep` % (`skipG%`, `Gfix`) | `TwoStepEE` % (`skipG%`, `Gfix`) |
|---|---|---|---|---|---|---|
| **0** | 100.0% (1.00) | 100.0% (1.00, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **1** | 100.0% (1.00) | 100.0% (1.00, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **2** | 100.0% (1.00) | 100.0% (1.00, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **3** | 100.0% (1.01) | 100.0% (1.01, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **4** | 100.0% (1.05) | 100.0% (1.05, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **5** | 100.0% (1.05) | 100.0% (1.05, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **6 ($=t$)** | 100.0% (1.18) | 100.0% (1.18, 0.0) | **100.0%** | **100.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **7 ($t+1$)** | 100.0% (1.29) | 100.0% (1.29, 0.0) | **0.0%** | **0.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **8 ($t+2$)** | 100.0% (1.34) | 100.0% (1.34, 0.0) | **0.0%** | **0.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 10 | 100.0% (1.68) | 100.0% (1.68, 0.0) | 0.0% | 0.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 12 | 100.0% (1.92) | 100.0% (1.92, 0.0) | 0.0% | 0.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 16 | 100.0% (2.02) | 100.0% (2.02, 0.0) | — | — | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 20 | 97.5% (2.21) | **100.0%** (2.29, 0.0) | — | — | 97.5% (97.5%, 0) | **100.0%** (100.0%, 0) |
| 24 | 80.0% (3.09) | **96.5%** (3.65, 1.0) | — | — | 80.0% (80.0%, 0) | **96.5%** (96.5%, 0) |
| 28 | 6.5% (3.40) | **36.5%** (6.00, 20.4) | — | — | 6.5% (6.5%, 0) | **36.5%** (38.5%*, 0) |

*\*At GF(64) $w=28$, `skipG = 38.5%` vs `ok = 36.5%` reveals 4/200 (2.0%) silent misdecodes into a distinct valid codeword via the `era_wt == d_loc - 1` Lagrange branch.*
**Mixed `GlobalEE` radius check ($e=4$ erasures, theoretical $v_{\max} = \lfloor(13-1-4)/2\rfloor = 4$, 50 words/wt):** $v \in \{0,1,2,3,4\} \to \mathbf{100.0\%}$ (50/50 each); $v \in \{5,6\} \to \mathbf{0.0\%}$ (0/50 each). Exact match.

### Table 5B — GF(256): $n=256, k=144, r=[6,6], d_{\text{loc}}=[3,3], d_{\text{design}}=7 \implies t_{\text{glob}}=3$

| Weight $w$ | `Iter` % (`avgI`) | `IterEE` % (`avgI`, `remE`) | `Global` % | `GlobalEE` ($e=0$) % | `TwoStep` % (`skipG%`, `Gfix`) | `TwoStepEE` % (`skipG%`, `Gfix`) |
|---|---|---|---|---|---|---|
| **0** | 100.0% (1.00) | 100.0% (1.00, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **1** | 100.0% (1.00) | 100.0% (1.00, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **2** | 100.0% (1.02) | 100.0% (1.02, 0.0) | 100.0% | 100.0% | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **3 ($=t$)** | 100.0% (1.06) | 100.0% (1.06, 0.0) | **100.0%** | **100.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **4 ($t+1$)** | 100.0% (1.17) | 100.0% (1.17, 0.0) | **0.0%** | **0.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| **5 ($t+2$)** | 100.0% (1.25) | 100.0% (1.25, 0.0) | **0.0%** | **0.0%** | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 6 | 100.0% (1.32) | 100.0% (1.32, 0.0) | — | — | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 8 | 100.0% (1.53) | 100.0% (1.53, 0.0) | — | — | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 10 | 100.0% (1.77) | 100.0% (1.77, 0.0) | — | — | 100.0% (100.0%, 0) | 100.0% (100.0%, 0) |
| 12 | 99.5% (1.83) | **100.0%** (1.85, 0.0) | — | — | 99.5% (99.5%, 0) | **100.0%** (100.0%, 0) |
| 16 | 94.5% (2.09) | **100.0%** (2.23, 0.0) | — | — | 94.5% (94.5%, 0) | **100.0%** (100.0%, 0) |
| 20 | 88.5% (2.21) | **100.0%** (2.51, 0.0) | — | — | 88.5% (88.5%, 0) | **100.0%** (100.0%, 0) |

**Mixed `GlobalEE` radius check ($e=2$ erasures, theoretical $v_{\max} = \lfloor(7-1-2)/2\rfloor = 2$, 25 words/wt):** $v \in \{0,1,2\} \to \mathbf{100.0\%}$ (25/25 each); $v \in \{3,4\} \to \mathbf{0.0\%}$ (0/25 each). Exact match.

---

## 6. Cost Model Fit: $\text{ms/word} = a + b \cdot (\text{\#iterations}) + c \cdot (\text{global\_decode\_invoked})$

Fitted via OLS on per-word `cpu_ms` from the seeded Monte Carlo sweeps (Suite A: GF(64), $N=3{,}000$; Suite B: GF(256), $N=2{,}400$):

| Code | Decoder | $N$ | $a$ (ms) | $b$ (ms / iter) | $c$ (ms / global) | $R^2$ | RMSE (ms) | Physical Interpretation |
|---|---|---|---|---|---|---|---|---|
| **GF(64)** ($c=8, d_{\text{loc}}=5$) | `Iterative` | 3,000 | $-7.67$ | $11.96$ | — | **0.8783** | 4.43 | 1st pass at low $w$ decodes few dirty cosets ($\sim 1.1\text{–}4$ ms), whereas passes $\ge 2$ decode most of 16 cosets ($\sim 12$ ms/iter) |
| **GF(64)** | `IterativeEE` | 3,000 | $-13.36$ | $24.76$ | — | **0.9302** | 9.57 | $b_{\text{EE}} / b_{\text{Iter}} = 2.07\times$ (`GRSErrorErasureDecoder` rebuild cost) |
| **GF(64)** | `Global` | 1,800 | $0.062$ | — | $16.11$ | **0.8109** | 2.45 | `0.062 ms` syndrome gate + `16.1 ms` length-64 Gao (`RMSE` = success vs. early-fail) |
| **GF(64)** | `TwoStep` | 3,000 | $-6.36$ | $10.77$ | $20.70$ | **0.9291** | 3.96 | Exact additive decomposition: Iter + Stage-2 Gao fallback |
| **GF(64)** | `TwoStepEE` | 3,000 | $-10.53$ | $22.74$ | $22.27$ | **0.9397** | 8.89 | `c = 22.3 ms` blends `GlobalEE` ($e>0$ punctured / early `e >= d_glob` abort) |
| **GF(256)** ($c=8, d_{\text{loc}}=3$) | `Iterative` | 2,400 | $-5.09$ | $15.40$ | — | **0.7105** | 6.35 | 64 cosets/iter; lower $R^2$ because dirty-coset count varies from 0 to 20 at `itr=1..2` |
| **GF(256)** | `IterativeEE` | 2,400 | $-6.37$ | $28.80$ | — | **0.6524** | 15.90 | $b_{\text{EE}} / b_{\text{Iter}} = 1.87\times$; `RMSE` reflects DUE pass cost variance |
| **GF(256)** | `Global` | 800 | $0.187$ | — | $245.08$ | **0.7152** | 66.97 | `0.187 ms` syndrome gate + `~330 ms` (pass) / `~205 ms` (early fail) Gao |
| **GF(256)** | `TwoStep` | 2,400 | $-3.37$ | $14.28$ | $208.03$ | **0.9123** | 9.22 | High $R^2$: Stage-2 fallback dominates variance (`+208 ms` whenever Stage 1 stalls!) |
| **GF(256)** | `TwoStepEE` | 2,400 | $-6.18$ | $28.81$ | *(never invoked)* | **0.6526** | 15.90 | Stage 1 solved 2,400/2,400 words $\implies c$ term never triggered! |

*(Note: The negative intercept $a$ in iterative fits is a well-defined physical artifact of coset-level syndrome gating: at `itr = 1` and low $w$, only $\le 2w$ of $2s$ cosets are dirty, whereas words reaching `itr >= 2` occur at high $w$ where many cosets are dirty. Adding `#dirty_cosets` as a feature eliminates the negative intercept and pushes $R^2 > 0.98$.)*

---

## 7. Ranked List of Decoder Speed-Ups (Evidence-Backed Estimates)

| Rank | Optimization | Target Decoders | Effort / Risk | Estimated Speed-Up | Evidence / Basis |
|---|---|---|---|---|---|
| **1 (P0)** | **Replace Sage local GRS (`Gao`/`ErrorErasure`) with shifted-coset GF($2^m$) log-table Berlekamp–Massey + Chien + Forney** (supporting errors + erasures uniformly) | `Iter`, `IterEE`, `TwoStep`, `TwoStepEE` | Medium (pure Py/NumPy ~160 LOC; Cython/Numba optional) / Low | **55×–110×** (`Iter` pure Py); **100×–250×** (`IterEE` pure Py); **>500×** (Cython/C) | `bench_04` verified 200/200 bit-exact: `17.8 µs` vs `1,103 µs` (**61.9×**) at GF(64) $c=8$; `28.1 µs` vs `3,067 µs` (**109.2×**) at GF(256) $c=16, r=12$ |
| **2 (P0)** | **Replace length-$n$ Sage `GRSGaoDecoder` / `GRSErrorErasureDecoder` with GF($2^m$) syndrome BM+Chien+Forney ($d_g \ll n$) or $O(n \log^2 n)$ fast-interp + half-GCD** | `Global`, `GlobalEE`, `TwoStep*` fallback | Medium (BM) / High (half-GCD) | **100×–500×** (BM when $d_g \ll n$); **17×–60×** (half-GCD at $n=1024..4096$, rate 1/2) | `bench_01`: Sage Gao is $O(n^{2.19})$ (`435 ms` at $n=256$, `10.2 s` at $n=1024$, `40.6 MiB`) vs. NTL `mul` `8.6 ms` at $n=1024$ |
| **3 (P0)** | **Eliminate `vector(F, ...)` / `vector(GF(2), ...)` in inner loops** (use cached `VectorSpace(F, c)` or NumPy/M4RIE batched matmul) | All 6 decoders & `_partition_is_syndrome_clean` | Trivial (~10 LOC) / Zero | **1.7×–2.3×** immediately inside Sage (`Iterative` at low $w$); **58×–573×** on syndrome checks when batched | `cProfile` + `bench_04`: `vector(F, lst)` = `51.3 µs` (`Sequence`/`pbori` failed import) vs `V(lst)` = `2.97 µs` (**17.3×**); M4RIE/NumPy batch = `0.57–1.41 µs/coset` (**58×–142×**) |
| **4 (P0)** | **Batched / vectorized multi-word decoding** (decode $B=64..1024$ words at once in NumPy/Cython GF($2^m$) arrays) | All 6 decoders (Monte Carlo simulations) | Medium–High / Low | **200×–600×** end-to-end throughput over current Sage loop | `bench_04`: batched syndrome `0.23–1.41 µs/coset` + dirty-only BM+Chien+Forney eliminates >90% Python/Sage per-word dispatch overhead |
| **5 (P1)** | **Dynamic stall-triggered DUE marking + skip redundant P0 syndrome re-check + `era_wt==0` fast path in `GlobalEE`** | `IterEE`, `GlobalEE`, `TwoStepEE` | Trivial (~15 LOC) / Low | **1.5×–2.0×** on `IterEE` high-$w$ & `GlobalEE(e=0)`; **1.3×** on `IterEE` clean words | `out_03_gf64_full.txt`: eliminates 3–4 wasted passes before `erasure_start_itr=5` at $w=20..28$; `GlobalEE(e=0)` drops `558 ms` $\to$ `367 ms` (GF(256)) and `16.3 s` $\to$ `8.6 s` (GF(1024)) |
| **6 (P1)** | **Dirty-coset worklist + 2-cycle detection + skip Stage-2 global check when `k == max_dim` and Stage 1 converged** | `Iter`, `IterEE`, `TwoStep`, `TwoStepEE` | Low (~30 LOC) / Zero | **1.3×–2.5×** at high $I$ / large $q$ (saves `11.4 ms` + `4 MiB` per word at $q=1024$, eliminates `max_itr=10` oscillations) | `out_02b.txt` + `bench_04` Part 3 + `out_03_gf64_full.txt` (15/200 words oscillated to iter 10 at $w=28$) |
| **7 (P2)** | **Canonical per-partition local PCM + closed-form global $\eta_i$ + lazy Stage-2 init** | All 6 `__init__` methods | Low (~25 LOC) / Zero | **10×–30×** on `__init__` (`289 ms` $\to$ `<15 ms` at $q=1024$) | `out_02.txt`: `Global.__init__` is $O(n^{1.85})$ (`221 ms` at $q=1024$) purely from `multipliers_product()` |

---

## 8. Reproducibility & Verification Artifacts

- Scratch scripts & outputs:
  - [`bench_01_local_and_global_grs.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_01_local_and_global_grs.py) $\to$ [`out_01.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_01.txt)
  - [`bench_02_all_six_decoders_perf_and_scaling.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_02_all_six_decoders_perf_and_scaling.py) & [`bench_02b_q1024_and_profile.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_02b_q1024_and_profile.py) $\to$ [`out_02.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_02.txt), [`out_02b.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_02b.txt)
  - [`bench_03_success_rates_iterations_and_cost_model.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_03_success_rates_iterations_and_cost_model.py) & [`bench_03_gf256_part2.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_03_gf256_part2.py) $\to$ [`out_03_gf64_full.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf64_full.txt), [`out_03_gf256.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf256.txt), [`out_03_gf256_part2.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_03_gf256_part2.txt)
  - [`bench_04_syndrome_and_batch_prototypes.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_04_syndrome_and_batch_prototypes.py) $\to$ [`out_04.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_04.txt)
  - [`bench_05_twostep_rescue_and_ntl4096.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/bench_05_twostep_rescue_and_ntl4096.py) $\to$ [`out_05.txt`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/A5/out_05.txt)
- No source, test, simulation, or `pyproject.toml` files were modified; baseline `pytest -q` verified 34/34 passing.
