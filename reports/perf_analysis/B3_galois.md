# B3 — Alternative Backend Evaluation: `galois` + NumPy

**Candidate:** [`galois` `0.4.11`](https://github.com/mhostetter/galois) + `numpy` `2.4.6` + `numba` `0.68.1` + `llvmlite` `0.50.0`  
**Target Interface:** `ff_backend` protocol (`L1–L5`) from [B2_sage_audit.md](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/B2_sage_audit.md)  
**Testbed:** `Linux x86_64` (Debian 12, `nproc = 2`, Intel Xeon Platinum 8481C), Python `3.11.2`, isolated venv `/tmp/venv-B3` (Sage baseline in `/home/antigravity/.venv-lrc`, `passagemath 10.8.14`, untouched).  
**Methodology:** Fixed seeds (`42`), warm-up `>= 3` (JIT warmed prior to steady-state timing), repeats `>= 20` (`3–5` for `>200 ms`, `1` for `>10 s`), `time.perf_counter` (wall median / p10 / p90) and `time.process_time` (contention-free single-core CPU median, since sibling swarm benchmarks ran concurrently on the 2-vCPU host), `tracemalloc` peak heap.  
**Artifacts & Reproducibility:** [probe_galois_api.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/probe_galois_api.py), [prototype_galois_backend.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/prototype_galois_backend.py), [bench_coldstart.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/bench_coldstart.py), [bench_sage_kernels.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/bench_sage_kernels.py), [bench_galois_kernels.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/bench_galois_kernels.py), [probe_matmul_parfor.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/probe_matmul_parfor.py), [verify_bitexact_sage_gen.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/verify_bitexact_sage_gen.py), [verify_bitexact_galois_check.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/verify_bitexact_galois_check.py).

---

## Executive Summary

1. **Exact algebraic alignment & MIT-compatible footprint:** `galois 0.4.11` supports **100% of required field families** ($\mathbb{F}_2$, $\mathbb{F}_{2^m}$ up to $2^{20}$ in LUT mode and arbitrarily large in calculate mode, odd primes $\mathbb{F}_p$, and odd extensions $\mathbb{F}_{p^m}$ including `GF(9)`, `GF(81)`, `GF(121)`, `GF(1369)`), uses the **exact same Conway polynomials and primitive elements** as Sage across all 13 tested fields (`GF(2)` through `GF(65536)`), shrinks venv disk footprint by **8.2×** (`257 MiB` vs. `2.1 GiB`), and eliminates Sage's **GPL-2.0+/GPLv3** license conflict (100% MIT/BSD/Apache-2.0/PSF-2.0).
2. **Bit-exact prototypes verified:** Our prototypes in [prototype_galois_backend.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/prototype_galois_backend.py) achieved **1,280 / 1,280 (100%) bit-exact equality** against Sage across syndrome computation (400 cases), `intersect_encoding_bases` (159 cases), RS Gao & Berlekamp–Welch decoding (460 cases), RS Error-Erasure decoding (250 cases), and additive subgroup enumeration (11 field/size configs = 2,358 subgroups).
3. **Four major structural pitfalls uncovered by source inspection & profiling:**
   - **Uncached per-field LLVM JIT cold-start penalty:** Because [`galois/_domains/_function.py:92`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_function.py#L92) compiles closures over mutated module globals without `cache=True`, **every fresh Python process** pays `637 ms` import + **`6.44 s`** first-touch JIT compilation for `GF(256)` + **`3.79 s`** per additional field. Across the 10 finite fields exercised by `tests/`, cold-start JIT alone takes **`26.0 s`** — making `pytest` **3.7× slower** than Sage's current `~7 s` suite.
   - **`row_reduce_jit` is pure Python (not JIT-compiled):** Despite its name, [`galois/_domains/_linalg.py:316–352`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_linalg.py#L316-L352) runs a Python `for j in range(ncols)` loop with per-column `np.multiply.outer`, making `row_reduce` / `left_null_space` **5×–50× slower** than Sage's M4RIE C backend on `GF(2^m)` (and slower even than Sage's generic Cython fallback on `GF(81)`).
   - **`matmul_jit` `_PARALLEL = True` 8 ms barrier cliff:** [`galois/_domains/_linalg.py:284`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_linalg.py#L284) forces `numba.prange(BATCH)` even for `BATCH = 1`, inducing an **`~8.0 ms`** thread-pool wakeup spike on small matvecs (`H_loc @ v` takes `8.00 ms` vs. `0.071 ms` with `NUMBA_NUM_THREADS=1` or `0.043 ms` via `np.dot(H, v)` — a **112× regression**).
   - **`galois.ReedSolomon` cannot replace Sage GRS & `galois.lagrange_poly` is $O(n^3)$:** [`galois/_codes/_reed_solomon.py:23`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_codes/_reed_solomon.py#L23) only implements cyclic BCH-style RS codes ($n \mid q - 1$, or shortened), with zero support for arbitrary evaluation points (additive cosets or $n = q$). Meanwhile, [`galois/_polys/_lagrange.py:124–131`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_polys/_lagrange.py#L124-L131) runs an $O(n^3)$ loop (though our 20-line $O(n^2)$ `newton_interpolate_galois` and `@njit(cache=True)` LUT Gao decoder bypass both, achieving **5–47×** and **444–1,427×** steady-state speedups over Sage!).
4. **Weighted Score & Verdict:** **`3.55 / 5.00` (`71.0%`)** — **Adopt partially (conditional)** only for L1 field definitions / vectorized batch workloads in long-running simulations, or **prefer a compiled non-JIT backend** (`python-flint` / Cython / Rust) if sub-second CLI & test startup matters.

---

## 1. Fit Against `B2` `ff_backend` Protocol (`L1–L5`)

| Layer / Requirement | `galois 0.4.11` Support | Exact Alignment & Caveats |
|---|---|---|
| **L1: Field families** ($\mathbb{F}_2$, $\mathbb{F}_{2^m}$, $\mathbb{F}_p$, odd $\mathbb{F}_{p^m}$, $q \le 2^{12}+$) | **Full** (`galois.GF(q)`) | Verified on all 13 fields (`2, 4, 9, 13, 16, 64, 81, 121, 256, 1024, 1369, 4096, 65536`). Uses Frank Luebeck's Conway polynomials — **100% identical** irreducible polynomials and primitive elements to Sage! LUT mode (`jit-lookup`) up to $q \le 2^{20}$; `jit-calculate` for $q > 2^{20}$ (arbitrary size via `object` dtype). |
| **L1: Element repr, `to_int`, hashability** | **Partial** | Integer codes in `0..q-1` match Sage's `field_element_to_int()` (`base-p` polynomial evaluation) identically (`int(x)` is $O(1)$). `GF.repr("int"|"poly"|"power")` supported. **Gap:** 0-D `FieldArray` scalars subclass `np.ndarray` and are **unhashable** (`TypeError: unhashable type`) and mutable; scalar ops incur ~20 µs NumPy ufunc dispatch overhead. |
| **L1: Subfields / `is_primitive`** | **Partial** | Has `GF.primitive_element`, `GF.primitive_elements`, `GF.prime_subfield`, `Poly.is_primitive()`. **No** `GF.subfields` or `GF.is_primitive_element()` method (not required by B2; primitivity testable via `multiplicative_order() == q - 1`). |
| **L2: Vectors & Matrices (`row_reduce`, `null_space`, `left_null_space`, `rank`, `inv`, `solve`)** | **Full API, Weak LA Perf** | Native `row_reduce(ncols, eye="left"|"right")`, `null_space()`, `left_null_space()`, `np.linalg.{matrix_rank,inv,solve}`, 0-row matrices `(0, n) @ (n,)` work cleanly. **Caveats:** (1) `row_reduce_jit` is pure Python (`_linalg.py:316`), 5–50× slower than M4RIE; (2) `matmul_jit` has `_PARALLEL = True` (`_linalg.py:284`), causing 8 ms barrier spikes on `BATCH=1`; (3) `GF.Vandermonde(element, rows, cols)` builds only *geometric* DFT matrices $a^{ij}$, NOT general Vandermonde matrices over arbitrary `eval_points`. |
| **L2: Subspace enumeration** | **Missing** | Not provided in `galois`. Implemented in 45 LOC NumPy RREF pivot enumerator (`enumerate_subspaces_numpy` + `find_additive_subgroups_galois`) — **162×–540× faster** than Sage. |
| **L3: Univariate Polynomials (`galois.Poly`)** | **Full API, Algorithmic Caveats** | Supports `+`, `-`, `*`, `divmod`, `//`, `%`, `gcd`, `egcd`, `lagrange_poly`, vectorized Horner evaluation `p(xs)`, `coefficients(size=..., order="asc")`. **Critical caveats:** (1) Default coefficient order is **descending** (Sage is ascending); (2) `Poly.Zero(GF).degree == 0` (**not `-1`**); (3) `galois.lagrange_poly` is **$O(n^3)$** (`_lagrange.py:124–131`); (4) `convolve_jit` is schoolbook $O(n^2)$ for extension fields (though NumPy BLAS for prime fields). |
| **L4: RS / GRS Decoding** | **Missing (Incompatible)** | `galois.ReedSolomon` (`_codes/_reed_solomon.py:23`) is strictly a **cyclic BCH-style RS code** ($n \mid q-1$ or shortened) over consecutive powers $\alpha^c, \dots, \alpha^{c+d-2}$. Supports `erasures=` in 0.4.11, **but CANNOT accept arbitrary evaluation points** (additive cosets containing $0$, or $n = q$) or column multipliers. Must implement custom GRS decoders on top of `galois`. |
| **L4: Coding Framework** | **Missing** | No `AbstractLinearCode` / `Encoder` / `Decoder` / `DecodingError` base classes (~150 LOC pure Python needed, same for all non-Sage backends). |
| **Vectorization & Batching** | **Excellent** | Full NumPy broadcasting, batched `(B, M, K) @ (B, K, N)` matmul, vectorized `poly(xs)`, and single-call batched coset syndrome `H_canonical @ W` (`(m, c) @ (c, s)` or `(c, 64)`). |

### Lookup Table Memory & Compilation Modes (`q = 2^8 .. 2^{20}`)

In `jit-lookup` mode (default for $q \le 2^{20}$), `galois` allocates three `int64` arrays per field class: `_EXP` (`2q` elements), `_LOG` (`q`), and `_ZECH_LOG` (`q`), totaling exactly **`32 * q` bytes** (`8 bytes × 4q`):

| Field | Default Mode | Array Dtype | LUT RAM (`_EXP`+`_LOG`+`_ZECH_LOG`) | `mul(4096)` Median | `inv(4096)` Median |
|---|---|---|---:|---:|---:|
| `GF(256)` (`jit-lookup`) | `jit-lookup` | `uint8` | **8.0 KiB** | `0.0279 ms` | `0.0460 ms` |
| `GF(256)` (`jit-calculate`) | opt-in | `uint8` | `0 KiB` (8 KiB if already built) | `0.0661 ms` (`2.4×` slower) | `0.5711 ms` (`12.4×` slower) |
| `GF(256)` (`python-calculate`) | opt-in | `uint8` | `0 KiB` | `2.7074 ms` (`97×` slower) | `111.34 ms` (`2,420×` slower) |
| `GF(1024)` (`jit-lookup`) | `jit-lookup` | `uint16` | **32.0 KiB** | `0.0272 ms` | `0.0324 ms` |
| `GF(4096)` (`jit-lookup`) | `jit-lookup` | `uint16` | **128.0 KiB** (fits L2 cache!) | `0.0288 ms` | `0.0509 ms` |
| `GF(4096)` (`jit-calculate`) | opt-in | `uint16` | `0 KiB` | `0.0809 ms` (`2.8×` slower) | `1.3833 ms` (`27.2×` slower) |
| `GF(4096)` (`python-calculate`) | opt-in | `uint16` | `0 KiB` | `4.6478 ms` (`161×` slower) | `212.82 ms` (`4,181×` slower) |
| `GF(65536)` (`jit-lookup`) | `jit-lookup` | `uint16` | **2.0 MiB** | — | — |
| `GF(2^20)` (`jit-lookup`) | `jit-lookup` | `uint32` | **32.0 MiB** (cutoff for `auto`) | — | — |

**Conclusion on `q = 2^{12}` memory:** At `q = 4096`, `galois`'s lookup tables consume only **128 KiB** — completely negligible and L2-cache resident. `jit-lookup` is strictly superior to `jit-calculate` for all $q \le 2^{16}$.

---

## 2. Cold-Start Import & Numba JIT Latency Impact (CLI & Test Suite)

Measured in fresh subprocesses via [bench_coldstart.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/bench_coldstart.py) (`raw` data in [coldstart_results.json](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/coldstart_results.json)).

### 2.1 Process Startup & Import Overhead (7 Fresh Subprocesses)

| Import Target | Median (ms) | p10 (ms) | p90 (ms) | Notes |
|---|---:|---:|---:|---|
| `python -c "pass"` | 10.6 | 10.3 | 10.8 | Baseline interpreter |
| `import numpy` | 137.3 | 121.1 | 141.9 | NumPy 2.4.6 |
| `import numba` | 312.7 | 281.9 | 314.3 | Numba 0.68.1 + `llvmlite` 0.50.0 |
| `import galois` | **637.4** | 607.0 | 663.6 | Only **3.8% faster** than Sage! |
| `import sage.all__sagemath_modules` | **662.4** | 621.9 | 667.8 | Sage baseline (same host) |

### 2.2 Per-Field First-Call JIT Compilation Tax (Uncached Across Processes!)

In [`galois/_domains/_function.py:89–95`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_function.py#L89-L95), `Function.jit` keys `_CACHE` by `(characteristic, degree, irreducible_poly)` and calls `numba.jit(..., nopython=True)` **without `cache=True`** (because field LUTs are injected via mutable module globals in `set_globals()`). Consequently, **every distinct field in every new process triggers full LLVM IR compilation**:

| Operation | `GF(256)` Cold (ms) | `GF(256)` Warm (ms) | Cold / Warm | `GF(4096)` Cold (ms, same proc) | Why? |
|---|---:|---:|---:|---:|---|
| `galois.GF(q)` init | **754.0** | 0.312 (cached) | 2,417× | 11.6 | First binary ext. field compiles table-gen + `add` |
| `mul` / `div` / `pow` ufuncs | 85.1 / 62.2 / 84.0 | 0.032 / 0.039 / 0.028 | ~2,400× | 87.6 / 66.5 / 88.0 | Per-field ufunc LLVM compile |
| `matmul` (`A @ B`) | **1,410.1** | 0.071 (`1-thr`) | ~19,800× | **1,222.8** | `_PARALLEL=True` Parfor LLVM pass per field! |
| `row_reduce` / `left_null_space` | 2.3 / 2.1 | 1.95 / 1.97 | 1× | 6.9 / 2.1 | Pure Python (`_linalg.py:316`) — no JIT |
| `poly_mul` / `poly_divmod` | 184.1 / 197.6 | 0.056 / 0.105 | ~2,600× | 176.5 / 292.4 | Per-field polynomial JIT |
| `poly_eval` (`p(xs)`) | **455.0** | 0.051 | 8,946× | **463.8** | Per-field Horner JIT |
| `lagrange_poly` | **2,852.8** | 0.118 | 24,251× | **1,018.9** | Compiles nested poly ops + `lagrange_poly_jit` |
| `egcd` | **350.9** | 1.786 | 196× | **352.6** | Per-field EGCD JIT |
| **Total 1st-touch (all ops above)** | **6,440.5 ms** | — | — | **3,794.9 ms** | **~7.1 s** for 1 field (+ import); **+3.8 s** each extra field |

### 2.3 Impact on `pytest` Suite & Short CLI Runs

Simulating the exact 10 fields exercised by `tests/` (`GF(2), GF(4), GF(9), GF(13), GF(16), GF(64), GF(81), GF(121), GF(256), GF(1369)`) with just 8 basic ops per field (`+`, `*`, `@`, `row_reduce`, `left_null_space`, `poly_mul`, `divmod`, `poly_eval`) in a fresh process:
- **Total cold-start JIT time across 10 fields:** **`26,016 ms` (`26.0 s`)** (`488 ms` import + `0.8–3.4 s` per field, because `GF(2)`, `GF(9)`, `GF(13)`, `GF(121)`, `GF(1369)` each trigger distinct characteristic/prime-field JIT pipelines in addition to per-field `matmul` + `Poly` compilations!).
- **Comparison:** The entire current Sage `pytest` suite (34 tests) runs in **`~7 s`**. Migrating `pytest` to `galois` would make developer test feedback **~3.7× slower**, and short single-field CLI runs (`< 10 s`) would spend **30–70%** of wall time in LLVM compilation. Conversely, long Monte Carlo simulations (minutes to hours over a single field `GF(256)` or `GF(4096)`) easily amortize the one-time `~3–6 s` compile cost.

---

## 3. Kernel Micro-Benchmarks (`galois` vs. SageMath)

All benchmarks use identical parameters and seeds (`42`) across `q ∈ {16, 64, 256, 1024, 4096}` plus odd-characteristic `GF(81)` and `GF(13)`. Because sibling swarm agents shared the 2-vCPU VM intermittently, every table reports **contention-free single-core CPU median (`cpu`)** alongside **wall-clock median `[p10, p90]`**; speed-up ratios (`Sage / galois`) are reported on CPU median (with wall ratio noted where informative). Raw data: [sage_kernels.json](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/sage_kernels.json), [galois_kernels.json](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/galois_kernels.json), [parfor_probe.json](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/parfor_probe.json).

### 3.1 Field Arithmetic (Scalar & Array)

| Kernel | `q=16` (Sage / `galois` / Ratio) | `q=64` | `q=256` | `q=1024` | `q=4096` | CPU Slope (Sage / `galois`) |
|---|---|---|---|---|---|---|
| **Scalar mul (1,000 ops)** | `0.037` / `20.66 ms` (**0.002×**) | `0.034` / `19.75 ms` (**0.002×**) | `0.033` / `20.15 ms` (**0.002×**) | `0.101` / `20.09 ms` (**0.005×**) | `0.104` / `20.42 ms` (**0.005×**) | `0.23` / `0.00` |
| **Scalar inv (1,000 ops)** | `0.019` / `29.55 ms` (**0.001×**) | `0.019` / `27.56 ms` (**0.001×**) | `0.017` / `28.59 ms` (**0.001×**) | `0.085` / `29.03 ms` (**0.003×**) | `0.085` / `29.58 ms` (**0.003×**) | `0.33` / `0.00` |
| **Scalar pow (1,000 ops)** | `0.040` / `20.75 ms` (**0.002×**) | `0.038` / `20.76 ms` (**0.002×**) | `0.037` / `21.22 ms` (**0.002×**) | `0.110` / `22.13 ms` (**0.005×**) | `0.109` / `21.41 ms` (**0.005×**) | `0.22` / `0.01` |
| **Array mul (len = `q`)** | `0.0014` / `0.0181 ms` (`0.08×`) | `0.0030` / `0.0191 ms` (`0.16×`) | `0.0098` / `0.0188 ms` (`0.52×`) | `0.1071` / `0.0212 ms` (**5.04×**) | `0.4139` / `0.0281 ms` (**14.7×**) | `1.08` / `0.07` |
| **Array inv (len = `q`)** | `0.0009` / `0.0272 ms` (`0.03×`) | `0.0018` / `0.0279 ms` (`0.06×`) | `0.0055` / `0.0285 ms` (`0.19×`) | `0.0851` / `0.0325 ms` (**2.62×**) | `0.3363` / `0.0440 ms` (**7.65×**) | `1.14` / `0.08` |
| **Array pow (len = `q`)** | `0.0013` / `0.0191 ms` (`0.07×`) | `0.0036` / `0.0199 ms` (`0.18×`) | `0.0106` / `0.0266 ms` (`0.40×`) | `0.1120` / `0.0243 ms` (**4.61×**) | `0.4407` / `0.0376 ms` (**11.7×**) | `1.09` / `0.11` |
| **Array mul (len = 4,096)** | `0.141` / `0.0268 ms` (**5.25×**) | `0.134` / `0.0263 ms` (**5.09×**) | `0.130` / `0.0351 ms` (**3.72×**) | `0.432` / `0.0273 ms` (**15.8×**) | `0.413` / `0.0280 ms` (**14.8×**) | `0.24` / `0.01` |

*Key insight:* `galois` has a **`~18–28 µs` fixed Python/NumPy ufunc dispatch floor** per call. For scalar ops (1 element), `galois` is **200×–1,600× slower** than Sage's Cython Givaro elements (`20 µs` vs. `33–104 ns`). Once vectorized (`N >= 512`), `galois` amortizes dispatch to **~6.8 ns/element** (constant from `GF(16)` to `GF(4096)`), beating Sage by **5×–15.8×** (and avoiding Sage's 3× Givaro `GF(256)→GF(1024)` step).

### 3.2 Polynomial Ring Operations (`n = q`)

| Kernel | `q=16` (Sage / `galois` / CPU Ratio) | `q=64` | `q=256` | `q=1024` | `q=4096` | CPU Slope (Sage / `galois`) |
|---|---|---|---|---|---|---|
| **Poly mul** (`deg q/2 × q/2`) | `0.047` / `0.056 ms` (`0.83×`) | `0.113` / `0.056 ms` (**2.03×**) | `0.735` / `0.096 ms` (**7.64×**) | `4.036` / `0.760 ms` (**5.31×**) | `21.68` / `13.43 ms` (**1.61×**) | `1.14` / `0.98` *(q≥256: `1.78`)* |
| **Poly XGCD** (`deg q, q-1`) | `0.666` / `7.467 ms` (`0.09×`) | `21.73` / `29.07 ms` (`0.75×`) | `103.4` / `113.5 ms` (`0.91×`) | `779.4` / `468.7 ms` (**1.66×**) | `5,129` / `2,068 ms` (**2.48×**) | `1.55` / `1.01` |
| **Poly interp (`galois.lagrange_poly`)** | `0.633` / `0.101 ms` (**6.28×**) | `8.782` / `1.275 ms` (**6.89×**) | `169.1` / `57.71 ms` (**2.93×**) | `4,203 ms` / `~3.7 s` *(O(n³) skip)* | infeasible | `2.12` / `2.29` *(cubic above 256)* |
| **Poly interp (Newton `galois`, 20 LOC)** | `0.633` / `1.879 ms` (`0.34×`) | `8.782` / `7.721 ms` (**1.14×**) | `169.1` / `31.82 ms` (**5.31×**) | `4,203` / `129.7 ms` (**32.4×**) | `>100 s` / **`572.5 ms`** | `2.12` / `1.03` |
| **Poly eval (`deg q-1` at `q` pts)** | `0.411` / `0.047 ms` (**8.68×**) | `7.159` / `0.061 ms` (**117×**) | `157.2` / `2.733 ms` (**57.5×**) | `3,657` / `12.00 ms` (**305×**) | `>20 s` (TO) / **`173.9 ms`** | `2.19` / `1.56` |

*Wall-clock `[p10, p90]` highlights (`q=256` / `q=1024`):* `poly_eval_n` `galois`: `2.83 [0.26, 8.02] ms` / `12.00 [11.86, 13.03] ms` vs. Sage `246.9 [212.7, 261.6] ms` / `6,331 [6,123, 6,331] ms` (**528× wall** at `q=1024`!). `poly_mul` narrows at `q=4096` (`1.61×`) because `galois`'s `convolve_jit` is schoolbook $O(n^2)$ whereas NTL uses Karatsuba/FFT ($O(n^{1.14})$).

### 3.3 Linear Algebra: RREF, Left Kernel & `intersect_encoding_bases`

| Kernel (uncached fresh matrix) | `q=16` | `q=64` | `q=256` | `q=1024` | `GF(81)` (odd `p^m`) | `GF(13)` (prime) |
|---|---|---|---|---|---|---|
| **Matrix RREF (`k × q`)** — Sage / `galois` | `0.015` / `0.486 ms` (**0.03×**) | `0.040` / `2.028 ms` (**0.02×**) | `0.485` / `10.55 ms` (**0.05×**) | `48.89` / `226.0 ms` (**0.22×**) | `4.362` / `6.255 ms` (**0.70×**) | `0.011` / `0.808 ms` (**0.01×**) |
| **Left kernel (`k × q`)** — Sage / `galois` | `0.055` / `0.543 ms` (**0.10×**) | `0.079` / `2.297 ms` (**0.03×**) | `0.345` / `11.35 ms` (**0.03×**) | `13.96` / `284.3 ms` (**0.05×**) | `3.243` / `6.744 ms` (**0.48×**) | `0.026` / `0.855 ms` (**0.03×**) |
| **`intersect_encoding_bases`** — Sage / `galois` | `0.742` / `7.985 ms` (`0.09×`) | `7.921` / `25.13 ms` (`0.32×`) | `132.8` / `96.15 ms` (**1.38×**) | `~33,100` *(A1)* / **`2,214 ms`** (**~15×**) | `16.02` / `33.96 ms` (`0.47×`) | — |

*Crucial diagnosis:* Pure linear algebra in `galois` is **4.6×–50× slower than Sage M4RIE** (and **33–76× slower than LinBox on `GF(13)`**) because [`row_reduce_jit`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_linalg.py#L316-L352) is pure Python, not Numba-compiled! Yet on end-to-end `intersect_encoding_bases` at `q >= 256`, `galois` **still beats Sage by 1.4×–15×** because Sage wastes 89.5% of runtime on Givaro $\leftrightarrow$ NTL `Polynomial_ZZ_pEX` serialization (confirmed in `A1`). Writing a 40-LOC `@njit(cache=True)` Gauss-Jordan RREF over `GF._EXP`/`_LOG` would invert this loss into a decisive win.

### 3.4 Matrix-Vector Syndromes & The `matmul_jit` `_PARALLEL = True` Cliff

Testing `H @ v` exposed a severe upstream defect in [`galois/_domains/_linalg.py:284`](file:///tmp/venv-B3/lib/python3.11/site-packages/galois/_domains/_linalg.py#L284) (`_PARALLEL = True` with `numba.prange(BATCH)`): on `BATCH = 1`, Numba's thread-pool barrier spikes from `~71 µs` to **`~8.00 ms`** (`p90 = 8.02 ms`). Both `NUMBA_NUM_THREADS=1` and `np.dot(H, v)` (which dispatches to `np.sum(H * v, axis=-1)` without `parallel=True`) eliminate the cliff completely:

| Syndrome Workload | `q=16` | `q=64` | `q=256` | `q=1024` | `q=4096` |
|---|---:|---:|---:|---:|---:|
| **Sage local `H * vector(F, lst)` (prod)** | `0.058 ms` | `0.061 ms` | `0.069 ms` | `0.135 ms` | `0.523 ms` |
| **Sage local `H * v` (pre-allocated)** | `0.0046 ms` | `0.0073 ms` | `0.0133 ms` | `0.0722 ms` | `0.372 ms` |
| **`galois` `H_loc @ v` (default threads — BUG!)** | `0.080` `[p90: 7.99]` | `0.082` `[p90: 1.92]` | **`7.998 ms`** | **`7.997 ms`** | **`6.987 ms`** |
| **`galois` `H_loc @ v` (`NUMBA_NUM_THREADS=1`)** | `0.071 ms` | `0.072 ms` | `0.071 ms` | `0.072 ms` | `0.079 ms` |
| **`galois` `np.dot(H_loc, v)` (no parfor)** | `0.043 ms` | `0.042 ms` | `0.043 ms` | **`0.043 ms`** (**3.1× prod / 1.7× pre**) | **`0.049 ms`** (**10.6× prod / 7.5× pre**) |
| **All `s=√q` cosets: Sage prealloc loop** | `0.017 ms` | `0.054 ms` | `0.211 ms` | `2.748 ms` *(wall 4.98)* | `24.78 ms` *(wall 50.25)* |
| **All `s=√q` cosets: `galois` batched `H@W` (`1-thr`)** | `0.063 ms` (`0.27×`) | `0.065 ms` (`0.83×`) | **`0.068 ms`** (**3.1×**) | **`0.094 ms`** (**29.4× CPU / 53× wall**) | **`0.370 ms`** (**67.1× CPU / 136× wall**) |
| **Batch 64 words: `galois` `H_loc @ W64` (`1-thr`)** | `0.98 µs/word` | `1.03 µs/word` | **`1.18 µs/word`** | **`1.93 µs/word`** | **`5.72 µs/word`** |
| **Global syndrome: Sage `H_glob * v`** | `0.013 ms` | `0.049 ms` | `0.295 ms` | `41.69 ms` | N/A |
| **Global syndrome: `galois` `np.dot(H_glob, v)`** | `0.041 ms` (`0.31×`) | `0.045 ms` (`1.10×`) | **`0.059 ms`** (**5.0×**) | **`0.141 ms`** (**296×**) | **`1.114 ms`** |

### 3.5 Subspace Enumeration (Custom NumPy vs. Sage)

| Config (`GF(q)`, subgroup size, #subgroups) | Sage CPU / Wall (ms) | `galois`+NumPy CPU / Wall (ms) | Speed-up (CPU / Wall) |
|---|---:|---:|---:|
| `GF(16)`, `s=4` (35 subgroups) | `27.03` / `54.78` | **`0.167`** / `0.167` | **162× / 329×** |
| `GF(64)`, `s=4` (651 subgroups) | `535.1` / `1,144.5` | **`3.216`** / `5.155` | **166× / 222×** |
| `GF(64)`, `s=8` (1,395 subgroups) | `1,324.9` / `2,461.8` | **`7.459`** / `7.532` | **178× / 327×** |
| `GF(81)`, `s=9` (130 subgroups, odd `p=3`) | `107.7` / `183.4` | **`0.612`** / `0.611` | **176× / 300×** |
| `GF(256)`, `s=4` (10,795 subgroups) | `~31,800` *(A1 extrap.)* | **`58.61`** / `109.3` | **~540×** |

### 3.6 Complete Length-`n` RS Gao Decode & Local Length-`c` Berlekamp–Welch Decode

Global RS envelope parameters match `A5`/`B1`: `n = q`, `d = 3√q/2 + 1`, `k = n - d + 1`, active error weight `w = t = (d - 1) // 2`. Local BW coset parameters: `c = √q`, `r = c // 2`, `d_loc = c // 2 + 1`, `w = t_loc = (c - r) // 2`.

| Decoder & Implementation | `q=16` (`n=16, t=3`) | `q=64` (`n=64, t=6`) | `q=256` (`n=256, t=12`) | `q=1024` (`n=1024, t=24`) | `q=4096` (`n=4096, t=48`) | CPU Slope |
|---|---|---|---|---|---|---|
| **Global Gao — Sage `GRSGaoDecoder`** | `1.645 ms` `[1.61,3.70]` | `18.15 ms` `[17.7,36.5]` | `357.6 ms` `[702,747]` | `10,910 ms` *(wall 17.1 s)* | infeasible (`>100 s`) | **`2.12`** |
| **Global Gao — `galois` (`lagrange_poly`)** | `3.983 ms` (`0.41×`) | `19.97 ms` (`0.91×`) | `74.71 ms` (**4.79×**) | infeasible ($O(n^3)$) | infeasible | `1.06` *(cub. >256)* |
| **Global Gao — `galois` (`newton_galois`, pure API)** | `7.597 ms` (`0.22×`) | `27.38 ms` (`0.66×`) | `50.98 ms` (**7.01×**) | **`158.6 ms`** (**68.8×**) | **`758.9 ms`** | `0.79` |
| **Global Gao — `@njit(cache=True)` on `GF` LUTs** | **`0.0038 ms`** (**433×**) | **`0.0265 ms`** (**685×**) | **`0.451 ms`** (**793×**) | **`7.646 ms`** (**1,427×**) | **`170.8 ms`** | **`1.95`** |
| **Local BW — Sage `GRSBerlekampWelch`** (`c=√q`) | `0.878 ms` (`c=4`) | `1.177 ms` (`c=8`) | `2.140 ms` (`c=16`) | `5.405 ms` (`c=32`) | `18.13 ms` (`c=64`) | `0.55` |
| **Local BW — `galois` (`null_space`, default thr.)** | `4.015 ms` (`0.22×`) | `15.99 ms` (`0.07×`) | `19.50 ms` (`0.11×`) | `10.10 ms` (`0.53×`) | `27.44 ms` (`0.66×`) | `0.24` |
| **Local BW — `galois` (`null_space`, `1-thr`)** | `1.355 ms` (`0.65×`) | `1.903 ms` (`0.62×`) | `2.974 ms` (`0.72×`) | `5.563 ms` (`0.97×`) | **`10.51 ms`** (**1.73×**) | `0.37` |

*Key insight:* Using `galois`'s public `Poly` API alone gives **7×–69×** on global Gao (`q=256..1024`), while dropping down to `@njit(cache=True)` on `galois`'s `_EXP`/`_LOG` tables yields **433×–1,427×** — demonstrating that the real speed comes from Numba + flat log/antilog LUTs, **not** from `galois`'s high-level `Poly`/`FieldArray` wrappers! (And notice `@njit(cache=True)` compiles once and caches on disk across runs, unlike `galois`'s own ufuncs.)

---

## 4. Prototype & Bit-Exact Verification (`1,280` Seeded Cases)

Implemented in [prototype_galois_backend.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/prototype_galois_backend.py) and cross-verified against SageMath ground truth generated by [verify_bitexact_sage_gen.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/verify_bitexact_sage_gen.py) via [verify_bitexact_galois_check.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B3/verify_bitexact_galois_check.py) (seed `42`):

| Prototype Component | Fields & Edge Cases Tested | Cases Verified | Bit-Exact Result |
|---|---|---:|---|
| **(a) PCM & Syndrome Computation** (`build_rs_parity_check_matrix`, `compute_syndrome`, `has_zero_syndrome`) | `GF(13)`, `GF(16)` (incl. full-rate `k=n=4`), `GF(64)`, `GF(81)`, `GF(256)`; valid codewords & corrupted words | **400** | **400 / 400 PASS** (exact PCM rows + syndromes) |
| **(b) `intersect_encoding_bases_galois`** | 9 structured Tamo–Barg two-set partitions (`GF(16)`, `GF(64)`, `GF(81)`, `GF(256)`) + 150 random poly subspace pairs (`GF(13)`, `16`, `64`, `81`, `256`) | **159** | **159 / 159 PASS** (exact ascending coeff lists) |
| **(c) `GaloisGRSGaoDecoder` & `GaloisGRSBWDecoder`** | `GF(13)`, `GF(16)` (incl. `n=q` with `0 ∈ eval_pts`, `d=2`, `n=k`), `GF(64)`, `GF(81)`, `GF(256)`; within-radius, beyond-radius failures & misdecodes, input immutability | **460** *(+330 char-2 JIT)* | **460 / 460 PASS** (100% agreement incl. misdecodes & `DecodingError`) |
| **(d) `GaloisGRSErrorErasureDecoder`** | `GF(13)`, `GF(16)`, `GF(64)`, `GF(81)`, `GF(256)`; `e=0`, `0<e<d-1` (within & beyond), `e=d-1`, `e>=d`, cached `G // E_poly` puncturing | **250** *(+155 char-2 JIT)* | **250 / 250 PASS** (matches B2 §3.5 contract; never leaks `ValueError`) |
| **(e) `find_additive_subgroups_galois`** | `GF(16)` (sizes 1,2,4,8,16), `GF(64)` (2,4,8), `GF(81)` (3,9), `GF(121)` (11) | **11 configs** *(2,358 subgroups)* | **11 / 11 PASS** (exact canonical element order) |
| **Total** | All 4 required prototypes + subspace enumeration | **1,280 cases** | **100% BIT-EXACT** |

---

## 5. Gap List & Implementation Effort (`ff_backend` Target)

| Gap (Missing or Unusable in `galois`) | Why It's Needed | Estimated LOC | Difficulty |
|---|---|---:|---|
| **1. General evaluation-point RS codec** (`Gao` + `ErrorErasure`, or shifted `BM + Chien + Forney`) | `galois.ReedSolomon` is cyclic-only ($n \mid q-1$); cannot decode additive cosets or $n=q$ | ~220 LOC (general $\mathbb{F}_{p^m}$) + ~180 LOC (fast `@njit(cache=True)` char-2) | Moderate–Hard |
| **2. Fast $O(n^2)$ polynomial interpolation & barycentric erasure eval** | `galois.lagrange_poly` is $O(n^3)$ | ~35 LOC (`newton_interpolate_galois` + barycentric) | Easy |
| **3. Fast JIT Gaussian elimination (`rref` / `left_null_space`)** | `galois` `row_reduce_jit` is pure Python, 5–50× slower than M4RIE | ~70 LOC (`@njit(cache=True)` over `_EXP`/`_LOG`) | Moderate |
| **4. `matmul` parfor workaround** (`np.dot` / `NUMBA_NUM_THREADS=1` or custom `@njit` matmul) | Avoids 8 ms `prange(1)` thread-pool spike in `matmul_jit` | ~30 LOC | Easy |
| **5. Subspace enumeration** (`enumerate_subspaces` / `find_additive_subgroups`) | Not in `galois` | ~50 LOC (already prototyped) | Easy |
| **6. General Vandermonde & RS PCM builder** | `GF.Vandermonde` is geometric-only ($a^{ij}$) | ~25 LOC (already prototyped) | Trivial |
| **7. Thin `FFBackend` wrapper** (hashable `FieldElement`, immutable `FFVector`/`FFMatrix` for B4, ascending `FFPoly` w/ `deg(0)=-1`, `divisors`, RNG) | Impedance mismatch: unhashable `FieldArray`, descending `Poly`, `Poly.Zero.degree == 0` | ~260 LOC | Easy–Moderate |
| **8. Coding-theory base classes** (`LinearCode`, `Encoder`, `Decoder`, `DecodingError`, `CartesianProductSpace`) | Replaces Sage `AbstractLinearCode`/`Encoder`/`Decoder` | ~150 LOC | Easy |
| **Total effort to full Sage replacement on `galois`** | Complete `ff_backend` implementation | **~840–1,020 LOC** | **Moderate** |

---

## 6. Non-Functional Assessment

| Dimension | Finding (Verified) |
|---|---|
| **License (MIT-compatible?)** | **PASS (100% permissive):** `galois 0.4.11` (**MIT**), `numba 0.68.1` (**BSD**), `llvmlite 0.50.0` (**BSD-2-Clause & Apache-2.0 w/ LLVM-exception**), `numpy 2.4.6` (**BSD-3-Clause**), `typing-extensions 4.16.0` (**PSF-2.0**). Completely resolves Sage's GPL-2.0-or-later / GPLv3 conflict. |
| **Install footprint (`du -sh`)** | **`257 MiB`** total venv (`/tmp/venv-B3`) — **8.2× smaller** than Sage's `2.1 GiB` `.venv-lrc`. Breakdown: `llvmlite` `172 MiB` (67%), `numpy` + `numpy.libs` `58 MiB` (23%), `numba` `17 MiB` (7%), `galois` `11 MiB` (4%). |
| **Wheel availability (Linux / macOS / Windows)** | `galois` ships a universal pure-Python wheel (`galois-0.4.11-py3-none-any.whl`, 4.2 MB). Compiled runtime deps (`numpy`, `numba`, `llvmlite`) provide official prebuilt wheels on PyPI for Linux (`x86_64`, `aarch64`), macOS (`arm64`, `x86_64`), and Windows (`x86_64`; note `win_arm64` lags in Numba). |
| **Pure-Python vs. compiled** | `galois` itself is pure Python, **tightly coupled** to compiled LLVM via `numba` + `llvmlite` (cannot be installed or run fast without `numba`; `python-calculate` fallback is 100×–4,000× slower). |
| **Python versions & Numba/NumPy coupling** | `galois 0.4.11` declares `Requires-Python: >=3.7`, `numpy>=1.21.0`, `numba>=0.55`. Effective floor/ceiling is dictated by `numba`: `numba 0.68.1` requires **`Python >=3.10`** and **`numpy < 2.6, >= 1.22`**, supports Python 3.11 / 3.12 / 3.13 / 3.14 (standard GIL builds; free-threaded `3.14t` still experimental), and historically **lags new CPython minor releases by 3–9 months** while LLVM/bytecode bridges are updated. |
| **Maintenance health** | Active: `galois 0.4.11` released **May 2, 2026** (uploaded on CPython 3.13.7), mature docs & test vectors cross-checked against Sage/SymPy/Octave. Risks: pre-1.0 versioning (`0.4.x`) and single primary maintainer (Matt Hostetter). |

---

## 7. Score Matrix

Weights per evaluation rubric: Performance `25`, Coverage `20`, Correctness (inverted) `20`, Install `10`, Maintainability `10`, License `5`, Migration Effort (inverted) `10`.

| Criterion | Weight | Score (0–5) | Weighted | Rationale |
|---|---:|---:|---:|---|
| **Performance** | 25 | **3.0** | 75.0 | Strong steady-state vectorized array ops (`5–16×`), poly eval (`58–528×`), batched syndromes (`3–136×`), and Numba-JIT LUT Gao (`433–1,427×`); severely penalized by **6.4 s / 26.0 s uncached cold-start JIT**, pure-Python `row_reduce` (`0.02–0.22×`), `matmul_jit` 8 ms parfor cliff, and `200–1,600×` slower scalars. |
| **Feature coverage** | 20 | **3.5** | 70.0 | Complete general $\mathbb{F}_{p^m}$ support with exact Conway match, `Poly`, `row_reduce`/`left_null_space`; lacks GRS/arbitrary-eval-point RS decoders, subspace enumeration, general Vandermonde, hashable scalars, and coding base classes. |
| **Correctness risk (inv.)** | 20 | **4.0** | 80.0 | **1,280 / 1,280 bit-exact** match with Sage; exact Conway polynomials; minor gotchas (`Poly` descending order, `Poly.Zero.degree == 0`, mutable/unhashable `FieldArray`) easily guarded in adapter. |
| **Install simplicity** | 10 | **4.0** | 40.0 | Single `pip install galois`, `257 MiB` (`8.2×` smaller than Sage), cross-platform wheels; `-1.0` for `172 MiB` `llvmlite` and Numba's CPython/NumPy upper-bound lag. |
| **Maintainability** | 10 | **3.5** | 35.0 | Active May 2026 release, clean API; `-1.5` for single-maintainer bus factor, pre-1.0 status, and needing to reach into private `_EXP`/`_LOG` LUTs to bypass slow `row_reduce`/`lagrange_poly`/`matmul`. |
| **License** | 5 | **5.0** | 25.0 | 100% MIT/BSD/Apache-2.0/PSF-2.0 stack — eliminates Sage GPL conflict completely. |
| **Migration effort (inv.)** | 10 | **3.0** | 30.0 | ~840–1,020 LOC to bridge GRS decoders, subspace enum, fast RREF/Newton, and L4 base classes. |
| **Weighted Total** | **100** | **3.55 / 5.00** | **355 / 500 (71.0%)** | — |

---

## 8. Verdict: **Adopt Partially (Conditional)** — Do **Not** Adopt Fully as Primary Engine

**Recommendation:** **Do not adopt `galois` as a full drop-in replacement** for `tamo-barg-lrc`. **Adopt partially (or only as an optional fallback/validation layer)** under the following boundary:

- **Where `galois` fits well:**
  - **L1 (`FiniteField` metadata & LUT source for general $\mathbb{F}_{p^m}$):** Zero-effort exact Conway polynomial parity with Sage across $\mathbb{F}_2$, $\mathbb{F}_{2^m}$, $\mathbb{F}_p$, and odd $\mathbb{F}_{p^m}$ (`GF(9)`, `GF(81)`, `GF(121)`, `GF(1369)`), plus L2-resident `128 KiB` lookup tables at `GF(4096)`.
  - **Long-running vectorized batch simulations** (single field per process, runtime $\gg 10\text{ s}$) where the 6.4 s JIT compile is amortized and batched `poly(xs)` / `H @ W` (with `NUMBA_NUM_THREADS=1`) shine.
- **Why `galois` should NOT be adopted for L2 (LA), L3 (interpolation), L4 (RS decoding), or as the sole default backend:**
  1. **Test-suite & CLI startup regression:** Uncached per-field LLVM JIT compilation inflates a 10-field test suite by **`+26.0 s`** (`3.7×` slower than Sage's entire `7 s` `pytest` run) and adds `~7 s` to single-field CLI invocations.
  2. **Key hot paths must bypass `galois` anyway:** You cannot use `galois.ReedSolomon` (cyclic-only), `galois.lagrange_poly` ($O(n^3)$), `galois` `row_reduce`/`left_null_space` (pure-Python, `5–50×` slower than M4RIE), `galois` scalar arithmetic (`200–1,600×` slower than Givaro), or `galois` default-thread `matmul` (8 ms barrier cliff). Once you write custom `@njit(cache=True)` / C / Cython kernels for RREF, matmul, interpolation, and RS decoding over flat `uint16` log/antilog LUTs (~300 LOC), **`galois` becomes a 172 MiB `llvmlite` wrapper around 40 lines of table generation!**
  3. **Better architectural comparison:** Compare directly against `B4` (`python-flint` — precompiled C, zero JIT cold-start, native fast `fq_nmod_poly`/`fq_nmod_mat`) and `B5` (custom Cython/Rust/Numba-cached `uint16` LUT engine — same `433–1,427×` steady-state decoding speed with disk-cached or AOT binaries and `< 15 ms` startup).
