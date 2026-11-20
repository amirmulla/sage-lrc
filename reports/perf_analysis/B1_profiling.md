# B1 — Whole-Program Profiling, Memory, Import-Time & Scaling Report

## Executive Summary

Whole-program profiling of `tamo-barg-lrc` v0.2.0 across 36 micro/macro workloads, 13 `cProfile` traces, 10 `py-spy` (Python + `--native`) flame captures, `tracemalloc` snapshots, `-X importtime` subprocess measurements, and a `q ∈ {16, 64, 256, 1024, 4096}` scaling sweep reveals **four systemic bottlenecks** that dominate wall-clock time and block scaling past `q = 256–1024`:

1. **Modular `passagemath` `vector()` penalty (~59–63% of iterative decoding, ~38% of end-to-end simulation)**: Every `sage.modules.free_module_element.vector(F, [...])` call executes two framework-level defects:
   - `site-packages/sage/structure/sequence.py:297` runs `isinstance(universe, (MPolynomialRing_base, BooleanMonomialMonoid))` where `BooleanMonomialMonoid` (`sequence.py:81`) is a `LazyImport` for uninstalled `passagemath-brial` (`sage.rings.polynomial.pbori.pbori`). The failed `ModuleNotFoundError` is **not negatively cached**, triggering a full `importlib._bootstrap._find_and_load` + `posix.stat` (`fstatat64`) filesystem scan on every coset vector (**57.3%** of `TamoBargIterativeDecoder.decode_to_code`!).
   - `site-packages/sage/modules/free_module_element.pyx:573` unconditionally invokes `numpy.set_printoptions(legacy="1.25")` under NumPy 2.x on every `vector()` call (**5.5%** of iterative decoding).
2. **NTL `Polynomial_ZZ_pEX` (arbitrary-precision GMP) backend + $O(k n^2)$ / $O(n^2)$ polynomial ops**: For all `GF(2^m)` (`q = 16..4096`), `PolynomialRing(GF(q), 'x')` dispatches to `Polynomial_ZZ_pEX` (NTL `ZZ_pE` over GMP limbs — confirmed via `py-spy --native`: `_ntl_gmul`, `_ntl_gadd`, `_ntl_gcopy`, `NTL::PlainMul`, `__gmpn_invert_limb`). Combined with:
   - Naive nested Horner evaluation `[[poly(alpha) for alpha in alphas] for poly in basis]` (`tamo_barg.py:334`, `tamo_barg_two_sets.py:505`): **$O(k n^2)$** (`2.84 s` at `GF(256)` single-set `k=64`; empirical exponent **$n^{2.90\text{–}3.03}$** at constant rate; infeasible at `q=1024`).
   - Sage's pure-Python $O(n^2)$ `PolynomialRing_field.lagrange_polynomial` (`polynomial_ring.py:2361`): **76.0%** of `TamoBargGlobalDecoder` (`222.6 ms` at `q=256`, **`5.00 s`** and **`41.7 MiB`** peak at `q=1024`).
   - Sage's `GRSErrorErasureDecoder` instantiating a fresh punctured GRS code per call and running an uncached naive $O(n^2)$ loop in `_polynomial_vanishing_at_alphas` (`grs_code.py:1505–1531`): **35.1%** of `TamoBargGlobalEEDecoder` (`405.6 ms` at `q=256`).
3. **Eager import tax (~500 ms Sage + ~425 ms Matplotlib/NetworkX)**: `import tamo_barg_lrc` takes **~596 ms** wall-clock (`1,169` modules via `sage.all__sagemath_modules`), and importing `tamo_barg_two_sets`, `expander`, or even the pure-stdlib `simulation/postprocess.py` (via `simulation/__init__.py:4`) pulls in `matplotlib.pyplot` + `networkx` + `numpy`, totaling **~1,136–1,155 ms** (`1,750` modules).
4. **M4RIE `mzed_t` Newton-John table cliff above `GF(256)`**: Crossing `GF(2^8) → GF(2^10)` triggers a **12.2× per-column cliff** in `Matrix_gf2e_dense` (`matrix_gf2e_dense.pyx:21–27`), pushing `two_set_init_k2` from `310 ms` (`563 KiB`) at `q=256` to `6.70 s` (`17.5 MiB`) at `q=1024` and `minimum_distance_py(k=2)` from `8.17 ms` to `840.7 ms`.

---

## 1. Testbed & Reproducible Harness

| Property | Value |
|---|---|
| Host OS / Kernel | `Linux sandbox 6.12.105+ x86_64` (Debian 12 container) |
| CPU / Cores | `Intel(R) Xeon(R) Platinum 8481C`, `nproc = 2` (single-threaded runs) |
| RAM | `7.8 GiB` total (`~7.0 GiB` available, `0B` swap) |
| Python / SageMath | `Python 3.11.2` (`/home/antigravity/.venv-lrc/bin/python`), modular `passagemath 10.8.14` |
| Key libraries | `numpy 2.4.6`, `networkx 3.6.1`, `matplotlib 3.11.2`, `passagemath-{flint,ntl,m4ri-m4rie,pari,linbox,gap}` |
| Native sampler | `py-spy 0.4.4` installed in isolated scratch venv `/tmp/venv-B1/bin/py-spy` |
| Seeds & methodology | `random.seed(42)`, `np.random.seed(42)`, `set_random_seed(42)`; warm-up `>= 3`, repeats `>= 20` (`>= 5` for `> 1 s` calls); `time.perf_counter`, `cProfile`, `tracemalloc`, `py-spy --rate 200` |
| Sage backends (`q = 16..4096`) | Field: `FiniteField_givaro` · Poly: `Polynomial_ZZ_pEX` (NTL) · Matrix: `Matrix_gf2e_dense` (M4RIE) · Vector: `FreeModuleElement_generic_dense` |

**One-command reproduction** (harness at [bench.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B1/bench.py)):
```bash
/home/antigravity/.venv-lrc/bin/python reports/perf_analysis/scratch/B1/bench.py --all --scaling-timeout 35
```
Individual stages: `--imports`, `--workloads`, `--profile`, `--memory`, `--scaling`, `--pyspy`. All raw artifacts are stored under `reports/perf_analysis/scratch/B1/` (`results.json`, `profiles/*_{cumulative,tottime}_top30.txt`, `profiles/*.prof`, `flames/*_{py,native}.raw`).

---

## 2. Import-Time Analysis (`python -X importtime`)

Measured over 5 fresh subprocesses (warm OS page cache) per target.

### 2.1 Standalone Process Import Cost vs Incremental Lazy-Import Cost

| Target | Standalone Wall Median [p10, p90] (ms) | `-X importtime` Cum. (ms) | Modules Loaded | Incremental After `import tamo_barg_lrc` (ms) |
|---|---:|---:|---:|---:|
| `sage.all__sagemath_modules` | 611.0 [594.0, 656.5] | 511.4 | 1,167 | — |
| `tamo_barg_lrc` (pkg `__init__.py`) | 596.0 [594.4, 617.1] | 497.3 | 1,169 | 0.00 (base) |
| `tamo_barg_lrc.subgroups` | 599.8 [592.7, 619.5] | 499.2 | 1,170 | 0.28 [0.27, 0.28] |
| `tamo_barg_lrc.polynomials` | 592.3 [587.2, 613.0] | 494.7 | 1,171 | 0.43 [0.43, 0.46] |
| `tamo_barg_lrc.tamo_barg` | 599.3 [588.8, 604.4] | 494.7 | 1,180 | 3.83 [3.77, 3.89] |
| `tamo_barg_lrc.expander` | 1,096.0 [1,085.0, 1,115.5] | 925.1 | 1,735 | **425.03** [422.00, 443.20] |
| `tamo_barg_lrc.tamo_barg_two_sets` | 1,138.1 [1,124.0, 1,158.9] | 961.6 | 1,745 | **463.82** [460.26, 530.61] |
| `tamo_barg_lrc.simulation.runner` | 1,154.5 [1,137.6, 1,235.5] | 984.0 | 1,750 | **488.28** [480.68, 496.11] |
| `tamo_barg_lrc.simulation.postprocess` | 1,135.6 [1,125.5, 1,150.1] | 974.1 | 1,750 | **484.49** [475.61, 488.15] |

### 2.2 Top Cumulative Import Contributors (`tamo_barg_two_sets` full import: 967.1 ms)

| Rank | Module | Cum. (ms) | Self (ms) | Triggered By |
|---:|---|---:|---:|---|
| 1 | `sage.all__sagemath_modules` | 485.6 | 0.8 | `tamo_barg_lrc/__init__.py:11` (eager on any package import) |
| 2 | `tamo_barg_lrc.expander` | 361.0 | 1.1 | `tamo_barg_two_sets.py:21` (top-level import of `create_bipartite_coset_graph`) |
| 3 | `sage.all__sagemath_categories` | 317.6 | 0.4 | Inside `sage.all__sagemath_modules` |
| 4 | `matplotlib.pyplot` | 197.0 | 7.6 | `expander.py:8` (eager even when plotting is unused!) |
| 5 | `matplotlib` (+ `figure`, `projections`) | 121.8 | 7.2 | `expander.py:7` (total Matplotlib subtree ≈ 319 ms) |
| 6 | `sage.rings.all__sagemath_categories` | 117.2 | 0.4 | Inside `sage.all__sagemath_categories` |
| 7 | `sage.all__sagemath_flint` | 106.6 | 15.3 | Inside `sage.all__sagemath_modules` |
| 8 | `sage.all__sagemath_objects` | 90.9 | 0.2 | Inside `sage.all__sagemath_modules` |

**Key import takeaways**:
- PEP 562 lazy exports in `tamo_barg_lrc/__init__.py:114` only save `tamo_barg_two_sets`/`expander` (~464 ms) when importing single-set symbols; the **486 ms** `sage.all__sagemath_modules` bootstrap is paid unconditionally.
- Moving `import matplotlib.pyplot as plt` inside `save_bipartite_graph_plot` (`expander.py:308`) eliminates **~319 ms** (`~33%` of two-set import time).
- Making `tamo_barg_lrc/simulation/__init__.py:4` lazy would drop `simulation.postprocess` (pure-stdlib CSV parser) from **1,136 ms** (`1,750` modules) to **<5 ms** if `tamo_barg_lrc/__init__.py` didn't eagerly boot Sage.

---

## 3. End-to-End Workload Benchmarks & `tracemalloc` Memory

All timings separate construction from per-call execution and report cached vs. uncached `generator_matrix()`. Memory values show Python-heap `tracemalloc` delta: **Peak** during call, **Retained** while return object is live, and **After GC** steady-state leakage/cache retention.

### 3.1 Construction & Encoding Workloads (Reference: `GF(256), n = 256, r = 2, d_local = 3`)

| Workload | Parameters | Median (ms) | p10 (ms) | p90 (ms) | Peak Mem (KiB) | Retained / After-GC (KiB) |
|---|---|---:|---:|---:|---:|---:|
| `subgroups_cosets_gf256` | `GF(256)`, `\|H\|=4` add | 0.506 | 0.497 | 0.526 | 24.2 | — |
| `construct_single_code` | `[256, 64, r=2, d=3]` `TamoBargCode` | 0.712 | 0.692 | 0.744 | — | — |
| `construct_single_encoder` | `[256, 64]` `TamoBargVectorEncoder` | 88.591 | 87.168 | 92.286 | — | — |
| `single_enc_plus_gm_uncached` | `[256, 64]` encoder + uncached GM | **2,835.049** | 2,830.136 | 2,836.682 | 165.6 | 29.6 / 0.8 |
| `single_gm_cached` | `[256, 64]` cached GM (`@cached_method`) | 0.0002 | 0.0002 | 0.0003 | — | — |
| `construct_single_erasure_dec` | `[256, 64]` `TamoBargErasureDecoder` | 0.025 | 0.022 | 0.035 | — | — |
| `construct_single_error_dec` | `[256, 64]` `TamoBargErrorDecoder` (64 GRS) | 26.192 | 25.683 | 27.464 | — | — |
| `construct_two_set_code` | `[256, 16, [2,2], [3,3]]` init (incl. enc+graph) | 310.281 | 309.335 | 317.544 | — | — |
| `two_enc_plus_gm_uncached` | `[256, 16]` encoder + uncached GM | **691.406** | 687.102 | 707.612 | 547.6 | 269.2 / 1.7 |
| `two_gm_cached` | `[256, 16]` cached GM (`@cached_method`) | 0.0002 | 0.0002 | 0.0003 | — | — |
| `construct_two_iter_dec` | `[256, 16]` `IterativeDecoder` (128 GRS) | 51.505 | 50.911 | 62.648 | 911.3 | 909.4 / 27.7 |
| `construct_two_iter_ee_dec` | `[256, 16]` `IterativeEEDecoder` (128 GRS) | 57.245 | 56.201 | 67.284 | — | — |
| `construct_two_global_dec` | `[256, 16]` `GlobalDecoder` (1 GRS) | 18.731 | 18.448 | 19.728 | — | — |
| `construct_two_global_ee_dec` | `[256, 16]` `GlobalEEDecoder` (1 GRS) | 18.888 | 18.591 | 19.441 | — | — |
| `construct_two_twostep_dec` | `[256, 16]` `TwoStepDecoder` | 70.712 | 68.787 | 79.549 | 941.8 | 928.1 / 26.3 |
| `construct_two_twostep_ee_dec` | `[256, 16]` `TwoStepEEDecoder` | 76.884 | 75.123 | 159.497 | — | — |
| `encode_single_set` | `[256, 64]` `encode(msg)` (cached GM) | 0.619 | 0.612 | 0.630 | — | — |
| `enc_poly_single_set` | `[256, 64]` `enc_poly(msg)` | 5.974 | 5.769 | 6.845 | — | — |
| `encode_two_set` | `[256, 16]` `encode(msg)` (cached GM) | 0.136 | 0.132 | 0.177 | 43.3 | 4.2 / 0.1 |
| `enc_poly_two_set` | `[256, 16]` `enc_poly(msg)` | 0.833 | 0.815 | 0.858 | — | — |

*(Note: `single_gm_cached` and `two_gm_cached` execute in ~0.2 µs, >10⁶× faster than uncached GM construction. Note also that `TamoBargCodeTwoSets.__init__` calls `self.max_dimension()` at line 294, which eagerly constructs `TamoBargTwoSetsVectorEncoder` and runs `intersect_encoding_bases`, whereas `TamoBargCode.__init__` defers encoder construction until first `.encoder()` call.)*

### 3.2 All 8 Decoders, `minimum_distance_py`, Expander, & End-to-End Simulation (Per-Call)

| Workload | Parameters | Median (ms) | p10 (ms) | p90 (ms) | Peak Mem (KiB) | Retained / After-GC (KiB) |
|---|---|---:|---:|---:|---:|---:|
| `decode_single_erasure` | `GF(256)` `[256,64]` `ErasureDecoder`, 16 era | 2.711 | 2.686 | 2.761 | 14.4 | 9.6 / 1.0 |
| `decode_single_error` | `GF(256)` `[256,64]` `ErrorDecoder`, 8 err | 8.960 | 8.304 | 11.344 | 50.2 | 43.1 / 1.0 |
| `decode_single_error_gf16` | `GF(16)` `[16,5,r=2,d=3]` `ErrorDecoder`, 1 err | 1.020 | 0.918 | 1.241 | — | — |
| `decode_two_iter_w3` | `GF(256)` `[256,16]` `IterativeDecoder`, w=3 | 10.238 | 9.813 | 11.551 | 15.4 | 11.3 / 1.0 |
| `decode_two_iter_w12` | `GF(256)` `[256,16]` `IterativeDecoder`, w=12 (stopping) | 24.999 | 23.496 | 26.444 | — | — |
| `decode_two_iter_ee_w3` | `GF(256)` `[256,16]` `IterativeEEDecoder`, w=3 | 19.332 | 18.780 | 20.480 | — | — |
| `decode_two_iter_ee_w16` | `GF(256)` `[256,16]` `IterativeEEDecoder`, w=16 (DUE) | 108.721 | 107.248 | 113.108 | 157.0 | 119.6 / 10.9 |
| `decode_two_global_w3` | `GF(256)` `[256,16]` `GlobalDecoder` (Gao), w=3 | **222.583** | 221.399 | 223.355 | 442.9 | 37.4 / 0.9 |
| `decode_two_global_ee_w3_e2` | `GF(256)` `[256,16]` `GlobalEEDecoder`, w=3 + 2 era | **405.581** | 403.453 | 407.026 | 944.1 | 542.4 / 1.5 |
| `decode_two_twostep_w3` | `GF(256)` `[256,16]` `TwoStepDecoder`, w=3 (clean) | 12.468 | 11.993 | 12.867 | — | — |
| `decode_two_twostep_w12` | `GF(256)` `[256,16]` `TwoStepDecoder`, w=12 (iter+Gao) | **250.373** | 249.003 | 251.355 | 452.4 | 39.4 / 0.9 |
| `decode_two_twostep_ee_w16` | `GF(256)` `[256,16]` `TwoStepEEDecoder`, w=16 | 110.901 | 109.060 | 113.518 | — | — |
| `minimum_distance_py_gf64_k2` | `GF(64)` `[64,2,[2,2],[3,3]]` (65 codewords) | 0.819 | 0.799 | 0.850 | — | — |
| `minimum_distance_py_gf64_k3` | `GF(64)` `[64,3,[2,2],[3,3]]` (4,161 codewords) | 61.555 | 61.312 | 62.578 | 31.3 | 25.6 / 0.3 |
| `expander_metrics_gf256` | `GF(256)` 128-node, 256-edge `MultiGraph` | 1.039 | 1.012 | 1.089 | 211.4 | 25.9 / 0.3 |
| `e2e_sim_error_weight` | `GF(64)` `[64,6]`, `TwoSteps`, `sim_itr=20`, `err=1..6` | 796.380 | 783.181 | 843.970 | 1,980.1 | 1,800.8 / 77.3 |

**Memory observations (`tracemalloc`)**:
- Steady-state memory leakage (`After-GC`) is negligible (`< 2 KiB` per decode, `< 28 KiB` for decoder init caching in Sage's `UniqueRepresentation` pools, `77.3 KiB` in `e2e_simulation` from Matplotlib font/path caches).
- `tracemalloc` only tracks Python-heap allocations (not C/C++ heap inside NTL/M4RIE/GMP/Givaro); even on Python heap alone, `GlobalEEDecoder` allocates `944 KiB` per call at `q=256`, and `GlobalDecoder` spikes to **`41.7 MiB`** at `q=1024` (from `divided_difference`'s triangular list of $n^2/2$ boxed `FiniteField_givaroElement` objects).
- `TwoStepDecoder` exhibits a **20× bimodal latency cliff**: `12.5 ms` when iterative decoding cleans all errors (global syndrome check is zero) vs. `250.4 ms` when a stopping set forces global Gao cleanup (`25 ms` iterative + `225 ms` global Gao).

---

## 4. Category Time Attribution & `py-spy` Native Cross-Validation

### 4.1 7-Category Self-Time Attribution (`cProfile` via `pr.getstats()`, 13 Workloads)

> **Important profiling methodology note**: Python's `pstats.Stats` merges dict entries by `(filename, lineno, funcname)`, which drops self-time on nested comprehensions sharing one line (such as `tamo_barg_two_sets.py:505`: `[[poly(alpha) ...]]`). Our harness sums `inlinetime` directly from `cProfile.Profile.getstats()`, preserving 100% of wall-clock self-time. Furthermore, because `passagemath` Cython extensions are compiled without `profile=True`, C/C++ time inside `Polynomial_ZZ_pEX.__call__`/`__mul__` and `Matrix_gf2e_dense` is charged by `cProfile` to the calling Python frame (`Pure Python loops` column, marked with `†`). Section 4.2 disambiguates these frames using `py-spy --native`.

| Workload | Profiled (s) | Sage Field Arith. | Sage Polynomial | Sage Matrix / LA | Sage Framework Overhead | Pure Python Loops | NetworkX | I/O |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_set_construct` | 2.872 | 2.9% | 3.6% | 0.1% | 0.0% | **93.4%** `†` | 0.0% | 0.0% |
| `two_set_construct` | 2.213 | 9.4% | 16.5% | 1.9% | 0.1% | **72.0%** `†` | 0.1% | 0.0% |
| `two_set_encode` (`encode`+`enc_poly`) | 0.427 | 0.5% | 9.0% | 1.3% | 16.5% | **72.8%** `†` | 0.0% | 0.0% |
| `single_set_erasure_decode` | 0.391 | 3.7% | **66.1%** | 1.3% | 4.3% | 24.4% | 0.0% | 0.3% |
| `single_set_error_decode` | 0.642 | 2.1% | 16.8% | 23.3% | **43.0%** | 13.4% | 0.0% | 1.4% `*` |
| `two_set_iter_decode_w3` | 1.016 | 0.4% | 5.2% | 10.2% | **58.8%** | 23.3% | 0.0% | 2.0% `*` |
| `two_set_iter_ee_decode_w16` | 1.618 | 1.2% | 6.6% | 11.8% | **59.0%** | 19.7% | 0.0% | 1.9% `*` |
| `two_set_global_decode_w3` | 1.158 | 0.7% | **76.0%** | 0.1% | 21.4% `‡` | 1.8% | 0.0% | 0.0% |
| `two_set_global_ee_decode` | 1.776 | 0.4% | **39.7%** | 4.8% | **54.6%** `‡` | 0.5% | 0.0% | 0.0% |
| `two_set_twostep_decode_w12` | 1.416 | 0.7% | **64.4%** | 1.7% | 27.9% `‡` | 5.1% | 0.0% | 0.3% `*` |
| `minimum_distance_py_gf64_k3` | 0.851 | 0.2% | 0.1% | 38.3% | 0.1% | **61.4%** `†` | 0.0% | 0.0% |
| `expander_metrics_gf256` | 0.070 | 0.0% | 0.0% | 0.0% | 5.2% | 20.9% | **72.9%** | 0.9% |
| `e2e_simulation` (`TwoSteps`, `GF(64)`) | 1.533 | 1.2% | 9.3% | 6.8% | **38.0%** | 24.9% | 2.9% | 16.8% |
| **Unweighted mean (13 workloads)** | — | **1.8%** | **24.1%** | **7.8%** | **25.3%** | **33.3%** | **5.8%** | **1.8%** |

- `†` Cython-inlined native work charged by `cProfile` to the calling Python frame:
  - `single_set_construct` (`93.4%`): `90.2%` is `tamo_barg.py:334` calling Cython `Polynomial_ZZ_pEX.__call__` (NTL) + `2.5%` `build_algebra_basis` (`Polynomial_ZZ_pEX.__mul__`).
  - `two_set_construct` (`72.0%`): `46.9%` is `tamo_barg_two_sets.py:505` (`Polynomial_ZZ_pEX.__call__`) + `20.4%` `polynomials.py:76` (`Polynomial_ZZ_pEX.__mul__`). True native attribution is **~84% Sage polynomial (NTL)**, ~10% field arithmetic, ~2% matrix/LA, ~4% Python.
  - `two_set_encode` (`72.8%`): `63.6%` is `polynomials.py:254` (`base_ring(m[i]) * basis[i]`, Cython scalar-poly mul + poly add).
  - `minimum_distance_py_gf64_k3` (`61.4%`): `57.8%` is `tamo_barg_two_sets.py:407–408` (Cython `tail_vec * sub_gen`, `lead_row + ...`, `hamming_weight()`).
- `‡` Inside `Sage framework overhead` for `GlobalDecoder`/`GlobalEEDecoder`: `grs_code.py:1014` (`17.1%` in `GlobalDecoder`) calls Cython `p(alphas[i])` (NTL poly eval) and `grs_code.py:1505` (`34.9%` in `GlobalEEDecoder`) calls Cython `G * (x - alpha)` (NTL poly mul).
- `*` That `0.3–2.0%` `I/O` inside pure decoders is **literally `posix.stat` (`fstatat64`)** triggered by `sage/structure/sequence.py:297`'s failed `LazyImport` lookup on every `vector()` call!

### 4.2 `py-spy --native` Ground-Truth Verification (200 Hz, 1,662–3,676 Samples/Workload)

| Workload | Native Samples | Top Native Leaf Frames (C/C++/Cython/libc) | Confirms |
|---|---:|---|---|
| `two_set_construct` | 2,892 | `_ntl_gadd` (4.7%), `_PyEval_EvalFrameDefault` (4.5%), `_ntl_gcopy` (3.8%), `_ntl_gmul` (3.6%), `__gmpn_invert_limb` (2.2%), `_ntl_gsub` (2.1%), `_ntl_gdiv` (1.9%) | `tamo_barg_two_sets.py:505` + `build_algebra_basis` spend native CPU in **NTL `ZZ_pE` + GMP multi-precision limbs** for 8-bit `GF(256)`! |
| `two_set_iter_decode` | 3,676 | `_PyEval_EvalFrameDefault` (15.6%), **`fstatat64` (3.5%)**, `_PyObject_GenericGetAttrWithDict` (2.9%), `_ntl_gmul` (1.8%), `_ntl_gcopy` (1.4%), `_ntl_gadd` (1.4%), **`PyTraceBack_Here` (1.3%)** | Confirms kernel `fstatat64` syscalls + Python exception tracebacks (`ModuleNotFoundError`) on every `vector()` call via `sequence.py:297`! |
| `two_set_global_decode` | 2,697 | `_ntl_gmul` (8.9%), `_ntl_gcopy` (6.9%), `_ntl_gadd` (6.8%), `_PyEval_EvalFrameDefault` (4.3%), `NTL::PlainMul` (2.8%), `_ntl_gzero` (2.8%), `__gmpn_invert_limb` (2.7%), `_ntl_gdiv` (2.7%) | `lagrange_polynomial` + `grs_code.py:1014` are overwhelmingly NTL `PlainMul` / GMP limb arithmetic (`>35%` in top-8 NTL/GMP leaf symbols alone). |
| `minimum_distance_py` | 1,662 | `_PyEval_EvalFrameDefault` (7.7%), `_PyObject_GenericGetAttrWithDict` (3.9%), `__Pyx_GetItemInt_Fast (element_givaro.cpp)` (3.1%), `Py_DECREF` (2.2%), `Givaro::GFqDom<int>::add` (1.8%) | Confirms `tamo_barg_two_sets.py:407–408` bottleneck is boxed Givaro element access + `FreeModule` Python object allocation (`41,620` vectors/call). |
| `e2e_simulation` | 1,810 | `_PyEval_EvalFrameDefault` (13.0%), `_PyObject_GenericGetAttrWithDict` (2.8%), `fstatat64` (2.4%), `open64` (1.3%), `read` (0.9%) | Confirms simulation is dominated by Python/framework `vector()` overhead + Matplotlib PNG rendering (`bezier.py` + `ImagingEncoder.encode`). |

*(Flame files stored in `reports/perf_analysis/scratch/B1/flames/*_{py,native}.raw`.)*

---

## 5. Scaling Sweep (`q ∈ {16, 64, 256, 1024, 4096}`)

All sweeps use `n = q`, `r = 2`, `d_local = 3`, additive subgroups of size `4` (disjoint shifted subspaces for two-set), isolated subprocesses, and a `35 s` per-point wall-clock budget (with two targeted `q=1024` verification runs for `two_set_init_k2` and `two_set_global_decode_w3` where warmup+repeats exceeded `35 s` while single-call latency was `5–6.7 s`).

### 5.1 Wall-Clock Time (Median ms) & Peak Python Memory (`tracemalloc` KiB)

| Workload | `q=16` | `q=64` | `q=256` | `q=1024` | `q=4096` |
|---|---|---|---|---|---|
| **`subgroups_cosets`** | 0.038 ms · 3.2 KiB | 0.130 ms · 7.6 KiB | 0.498 ms · 24.2 KiB | 2.636 ms · 378.7 KiB | 10.518 ms · 1,507 KiB |
| **`single_set_init_k4`** (`k=4`) | 1.826 ms · 25.7 KiB | 13.123 ms · 32.0 KiB | 122.431 ms · 57.8 KiB | 1,888.6 ms · 500.0 KiB | **INFEASIBLE** (`>35 s`¹) |
| **`single_set_init_const_rate`** (`k=q/4`) | 1.895 ms · 25.7 KiB | 50.121 ms · 36.9 KiB | 2,810.6 ms · 181.3 KiB | **INFEASIBLE** (`>35 s`²) | **INFEASIBLE** (skipped) |
| **`two_set_init_k2`** (full basis intersect) | 1.496 ms · 57.0 KiB | 16.536 ms · 111.9 KiB | 310.120 ms · 563.0 KiB | 6,703.4 ms · 17,498 KiB `§` | **INFEASIBLE** (`>60 s`³) |
| **`two_set_gm_const_rate`** (`k=q/8`, GM only) | 0.558 ms · 4.5 KiB | 20.033 ms · 8.8 KiB | 1,340.4 ms · 73.5 KiB | **INFEASIBLE** (`>35 s`⁴) | **INFEASIBLE** (skipped) |
| **`two_set_iter_decode_w3`** (`w=3`, per-word) | 2.311 ms · 10.1 KiB | 3.387 ms · 13.0 KiB | 9.567 ms · 15.4 KiB | 33.134 ms · 22.6 KiB | **INFEASIBLE** (init `>35 s`³) |
| **`two_set_global_decode_w3`** (`w=3`, per-word) | 1.392 ms · 10.4 KiB | 11.664 ms · 39.5 KiB | 184.512 ms · 356.4 KiB | 5,001.4 ms · 41,687 KiB `§` | **INFEASIBLE** (init+dec `>60 s`⁵) |
| **`expander_metrics`** (`n/2` nodes, `n` edges) | 0.417 ms · 40.9 KiB | 0.411 ms · 83.6 KiB | 1.627 ms · 400.8 KiB | 12.915 ms · 3,146 KiB | 442.119 ms · 37,196 KiB |
| **`minimum_distance_py_k2`** (`k=2`) | 0.176 ms · 19.7 KiB | 0.907 ms · 20.8 KiB | 8.166 ms · 28.9 KiB | 840.664 ms · 424.4 KiB | **INFEASIBLE** (init `>35 s`³) |

*Infeasibility reasons*:
- ¹ Extrapolated `~28–35 s` per call × warmup+repeats (`>100 s`) from `build_algebra_basis` + `k=4` Horner evaluations over degree-`4095` `Polynomial_ZZ_pEX`.
- ² At `k = 256, n = 1024`, `generator_matrix()` performs `262,144` degree-`<=1023` NTL Horner evaluations (`~160 s/call` extrapolated from $n^{2.90}$).
- ³ `TamoBargCodeTwoSets.__init__` builds `1,024` algebra powers of degree up to `4095` in `Polynomial_ZZ_pEX` and runs Zassenhaus `left_kernel_matrix()` on a `4096 × 4096` `Matrix_gf2e_dense` over `GF(2^{12})` (`~140–250 s` extrapolated).
- ⁴ `k = 128, n = 1024` GM evaluation extrapolates to `~85 s/call` ($n^{3.03}$).
- ⁵ `q=4096` global Gao `lagrange_polynomial` extrapolates to `~110–140 s/word` and `~660 MiB` Python heap for the $4096^2/2$ `divided_difference` table.
- `§` Measured in targeted `q=1024` runs (`n=3` timed calls after 1 warm-up) with `60 s` budget.

### 5.2 Empirical Log-Log Exponents vs. Theoretical Complexity

| Workload | Pts (`n`) | Time Slope (all `q`) | $R^2$ | Time Slope (`q >= 64`) | $R^2$ | Peak Mem Slope | Theoretical Complexity | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `subgroups_cosets` | 5 | **1.03** | 0.997 | **1.07** | 0.998 | 1.17 | $O(q \log q)$ | Matches theory; scales cleanly to `q=4096` (`10.5 ms`). |
| `single_set_init_k4` | 4 | **1.66** | 0.994 | **1.79** | 0.997 | 0.68 | $O(n^2)$ | Approaching $n^2$ (`122.4 → 1888.6 ms` = ×15.4 = $4^{1.97}$) via `build_algebra_basis` + `k=4` degree-$n$ evals. |
| `single_set_init_const_rate` | 3 | **2.63** | 0.996 | **2.90** | 1.000 | 0.70 | $O(n^3)$ | Exact cubic $O(k n^2) = O(n^3)$ from nested `[[poly(alpha) ...]]`. |
| `two_set_init_k2` | 4 | **2.03** | 0.997 | **2.17** | 1.000 | 1.36 | $O(n^2 \text{–} n^3)$ | Dominated by `build_algebra_basis` ($O(n^2)$ NTL poly mul) + `padded_list` ($O(n^2)$) + M4RIE `left_kernel_matrix` ($O(n^\omega)$ + $2^m$ cliff at $q=1024$). |
| `two_set_gm_const_rate` | 3 | **2.81** | 0.998 | **3.03** | 1.000 | 1.00 | $O(k n^2) = O(n^3)$ | Exact cubic blowup ($4^{3.03} = 66.8\times$ per 4× $q$). |
| `two_set_iter_decode_w3` | 4 | **0.65** | 0.954 | **0.82** | 0.997 | 0.19 | $O(n)$ | Near-linear (`256 → 1024` = ×3.46 = $4^{0.90}$) because `w=3` triggers ~3 local Gao decodes ($O(1)$) + $n/2$ coset syndrome checks ($O(n)$). |
| `two_set_global_decode_w3` | 4 | **1.97** | 0.991 | **2.19** | 0.997 | **1.95** | $O(n^2 \text{–} n^{2.5})$ | `256 → 1024` is ×27.1 ($4^{2.38}$) time & ×117 memory: `lagrange_polynomial` $O(n^2)$ NTL poly updates + $O(n^2)$ `divided_difference` table. |
| `expander_metrics` | 5 | **1.25** | 0.874 | **1.66** | 0.957 | 1.24 | $O(n^3)$ | Constant overhead at `q<=64`; from `q=1024 → 4096` jumps ×34.2 ($4^{2.55}$) as dense `np.linalg.eigvalsh` on $2048 \times 2048$ matrix takes over (`37.2 MiB`). |
| `minimum_distance_py_k2` | 4 | **1.99** | 0.940 | **2.46** | 0.959 | 0.69 | $O(q^{k-1} n) = O(q^2)$ | Matches $O(q^2)$ up to `q=256` (`0.91 → 8.17 ms` = ×9.0), then suffers a **×103 jump** at `q=1024` (`840.7 ms`) due to M4RIE's `GF(2^8) → GF(2^{10})` Newton-John table cliff (`v * M`: `0.038 → 0.284 µs/col`). |

---

## 6. Global Top-10 Performance Hotspots (`file:line`)

Ranked by cross-workload severity (measured self-time share × ubiquity across construction, decoding, distance, and simulation). Every location is directly verified against `cProfile`, `py-spy --native`, and installed source code.

| Rank | Location (`file:line`) | Category | Measured Cost (Peak / Typical Share) | Root Cause & Exact Mechanism | Expected Optimization Leverage |
|---:|---|---|---|---|---|
| **1** | [`sage/rings/polynomial/polynomial_ring.py:2361`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/polynomial/polynomial_ring.py#L2361) (`lagrange_polynomial`) + `:2272` (`divided_difference`) | Sage polynomial (NTL) | **76.0%** of `GlobalDecoder` (`177 ms/call` at `q=256`; `~3.8 s` & `41.7 MiB` at `q=1024`), **66.1%** of `ErasureDecoder`, **64.4%** of `TwoStepDecoder` (`w=12`), **39.7%** of `GlobalEEDecoder` | Pure-Python $O(n^2)$ Newton divided-difference interpolation building NTL `Polynomial_ZZ_pEX` incrementally on every decode (`O(n^2)` poly muls + `O(n^2)` memory table). | **10–100×** via barycentric evaluation (for erasures), syndrome-based Berlekamp-Massey/Forney (no $n$-point interpolation!), or subproduct-tree / NTT interpolation over `GF(2^m)`. |
| **2** | [`sage/structure/sequence.py:297`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/structure/sequence.py#L297) (`Sequence` → `LazyImport(BooleanMonomialMonoid)` at `:81`) + [`sage/modules/free_module_element.pyx:573`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/modules/free_module_element.pyx#L573) (`numpy.set_printoptions`) | Sage framework overhead (+ `posix.stat` I/O) | **62.8%** of `IterativeDecoder` (`57.3%` `importlib`+`stat`, `5.5%` `set_printoptions`), **59.0%** of `IterativeEEDecoder`, **38.0%** of `e2e_simulation`, **~25%** of `ErrorDecoder` | Every `vector(self._F, [...])` call in coset loops (`tamo_barg_two_sets.py:556,580,674,709,723`, `tamo_barg.py:468`) re-imports missing `sage.rings.polynomial.pbori.pbori` (6,550 `fstatat64` scans per 50 decodes) AND calls `numpy.set_printoptions(legacy="1.25")`. | **2.5–3× immediate speedup** in Sage by replacing `vector(F, lst)` with cached `V(lst)` (`VectorSpace(F, m)`) or computing syndromes via flat NumPy/C arrays; **>50×** without Sage. |
| **3** | [`src/tamo_barg_lrc/tamo_barg.py:334`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg.py#L334) & [`src/tamo_barg_lrc/tamo_barg_two_sets.py:505`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L505) (`generator_matrix` nested `<listcomp>`) | Sage polynomial (NTL) via Python loop | **90.2%** of `single_set_construct` (`2.75 s` at `q=256, k=64`), **46.9%** of `two_set_construct` (`390 ms` at `q=256, k=16`); **$O(n^3)$** scaling | Evaluates every basis polynomial independently at all $n$ points via `poly(alpha)` (`Polynomial_ZZ_pEX.__call__`, GMP/NTL Horner) instead of evaluating the $s$ algebra basis powers `g(x)^j` once (or matrix-multiplying coefficient matrix by Vandermonde / additive FFT). | **50–500×**: `g(alpha)` is constant on each coset (only `partition_size` unique values!) — single-set GM becomes $O(k n)$ table lookup; two-set GM via `coeff_mat @ Vandermonde` or additive FFT drops $O(k n^2) \to O(n^2 \log n)$. |
| **4** | [`sage/coding/grs_code.py:1505`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1505) (`_polynomial_vanishing_at_alphas`, loop at `:1529–1530`) | Sage framework / Sage polynomial (NTL) | **35.1%** of `GlobalEEDecoder` (`155 ms/call` at `q=256`), **3.4%** of `IterativeEEDecoder` (`w=16`) | `GRSErrorErasureDecoder` creates a fresh punctured GRS code per call, defeating `@cached_method`, and multiplies `G = G * (x - alpha_i)` in a naive $O(n^2)$ linear loop (`Polynomial_ZZ_pEX.__mul__`). | **>50×**: Precompute full vanishing polynomial $A(x)$ once (it is simply $x^q - x$ when $n = q$!) and divide by the tiny degree-$e$ erasure locator $\prod_{i\in E}(x - \alpha_i)$, or use Forney/BM erasure-error decoding. |
| **5** | [`src/tamo_barg_lrc/polynomials.py:76`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L76) (`build_algebra_basis`, loop at `:104–108`) | Sage polynomial (NTL) via Python loop | **20.4%** of `two_set_construct` (`150 ms/call` at `q=256`), **2.5%** of `single_set_construct` (`72 ms/call`) | Computes `[1, g, g^2, ..., g^{s-1}]` by repeated dense `Polynomial_ZZ_pEX` multiplication (`curr = curr * good_poly`) over NTL arbitrary-precision `ZZ_pE` (`O(s^2 \cdot |H|)` NTL muls). | **10–100×**: Over `GF(2^m)` with additive subgroup $H$, `g(x)` is a sparse linearized polynomial ($\le m+1$ non-zero terms), or compute directly in evaluation domain (no monomial expansion needed!). |
| **6** | [`sage/coding/grs_code.py:1014`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/grs_code.py#L1014) (`GRSEvaluationPolynomialEncoder.encode` `<listcomp>`) | Sage framework / Sage polynomial (NTL) | **18.4%** of `GlobalDecoder` (`43 ms/call`), **9.6%** of `GlobalEEDecoder` (`43 ms/call`), **8.6%** of `ErrorDecoder`, **3.9%** of `e2e_simulation` | After Gao/BW finds message polynomial `p(x)`, re-encodes codeword via `[col_mults[i] * p(alphas[i]) for i in range(n)]` — $n$ NTL `Polynomial_ZZ_pEX.__call__` Horner evaluations + another `vector()` call! | **20–100×** via syndrome/Forney error-value evaluation (only modifies the $\le t$ error positions instead of re-evaluating all $n$ coordinates!). |
| **7** | [`sage/rings/polynomial/polynomial_ring.py:336`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/polynomial/polynomial_ring.py#L336) (`_element_constructor_`) + [`sage/rings/finite_rings/finite_field_givaro.py:229`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/rings/finite_rings/finite_field_givaro.py#L229) (`_element_constructor_`) | Sage polynomial & field arithmetic | **25.9%** of `two_set_construct` (`120 ms` poly + `68 ms` field across `130k` coercions/3 calls), **15.0%** of `ErasureDecoder`, **13.6%** of `ErrorDecoder` | Heavy Python-level type dispatch & element coercion during `padded_list()` (`polynomials.py:200,206`), `poly_ring(coeffs)` (`polynomials.py:228`), `lagrange_polynomial`, and `base_field(x)` conversions. | **>50×** by representing `GF(2^m)` elements as unboxed `uint16` / `uint8` integers (log/exp tables or PCLMULQDQ/SIMD) instead of boxed Sage Python objects. |
| **8** | [`src/tamo_barg_lrc/tamo_barg_two_sets.py:380`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/tamo_barg_two_sets.py#L380) (`minimum_distance_py`, lines `:406–408`) + [`sage/modules/free_module.py:2166,6514`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/modules/free_module.py#L2166) | Sage matrix/LA + Python loop | **99.5%** of `minimum_distance_py` (`61.6 ms` at `GF(64) k=3`; `840.7 ms` at `GF(1024) k=2`) | Iterates `VectorSpace(field, tail_dim)` constructing $q^{k-1}$ Sage vectors in Python (`38.3%`), computes `lead_row + tail_vec * sub_gen` hitting M4RIE's $2^m$ cliff (`57.8%`), instead of Gray-code single-row XOR/add updates in contiguous `uint8`/`uint16` arrays. | **50–500×** via Gray-code incremental codeword updates (`cw ^= delta * row`) in C/Numba/Cython over `uint8`/`uint16` arrays (or Brouwer-Zimmermann algorithm). |
| **9** | [`src/tamo_barg_lrc/polynomials.py:254`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/polynomials.py#L254) (`evaluate_message_polynomial` `<genexpr>`) + [`sage/coding/encoder.py:119`](file:///home/antigravity/.venv-lrc/lib/python3.11/site-packages/sage/coding/encoder.py#L119) (`encode`) | Sage polynomial / framework | **72.8%** (`enc_poly`, `0.83 ms` two-set / `5.97 ms` single-set) & **16.5%** (`encode`, `0.14 ms` two-set / `0.62 ms` single-set) | `evaluate_message_polynomial` runs `sum(base_ring(m[i]) * basis[i], zero)` over NTL `Polynomial_ZZ_pEX` objects; `encode()` wraps `m * G` in Sage `Encoder` validation & `FreeModuleElement` allocation. | **20–100×** via dense `GF(2^m)` matrix-vector product over `uint8`/`uint16` arrays (and polynomial coefficient matrix multiply for `enc_poly`). |
| **10** | [`src/tamo_barg_lrc/expander.py:308`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L308) (`save_bipartite_graph_plot` → `matplotlib/bezier.py:295`, `PIL`) & [`expander.py:219,236`](file:///home/antigravity/amirmulla/optimal_lrc/src/tamo_barg_lrc/expander.py#L219) (`compute_expansion_metrics` → `networkx` + `np.linalg.eigvalsh`) | I/O & NetworkX | **16.8%** I/O + **2.9%** NetworkX of `e2e_simulation` (`~300 ms` fixed overhead per sim run); `expander_metrics` jumps to **`442 ms` & `37.2 MiB`** at `q=4096` | Unconditional Matplotlib Bezier edge rendering & PNG encoding on every simulation run (plus **319 ms** import cost!), and dense `O(|V|^3)` `np.linalg.eigvalsh` on full adjacency & Laplacian matrices (twice!) in `compute_expansion_metrics`. | **5–20×** by lazy-importing Matplotlib & making plotting optional, and replacing dense `eigvalsh` with sparse Lanczos (`scipy.sparse.linalg.eigsh` / power iteration for top-2 + Fiedler value) or direct bipartite incidence arrays. |

---

## 7. Synthesis & Architectural Implications for SageMath Replacement

1. **Why SageMath hurts performance here is structural, not incidental**:
   - Even for small binary extension fields `GF(2^8)` where every field element fits in a single byte (`uint8`) and multiplication is 2 table lookups (`exp[(log[a] + log[b]) % 255]`, ~1 ns), Sage routes polynomials through **NTL arbitrary-precision `ZZ_pE` over GMP limbs**, vectors through **boxed Python object arrays** (`FreeModuleElement_generic_dense`), and vector creation through **`sequence.py:297`'s uncached failing `LazyImport` + `numpy.set_printoptions`**.
   - Every single top-8 computational hotspot operates strictly over `GF(q)` (`q <= 4096`) using four elementary primitives: finite-field arithmetic (`GF(2^m)` / `GF(p^m)`), univariate polynomial ops, dense `GF(q)` linear algebra (RREF / kernel / matvec), and Reed-Solomon error/erasure decoding (Berlekamp-Massey + Chien + Forney).
2. **Immediate quick wins even before full replacement** (for downstream engineering reference; no source modified in this audit):
   - Fixing `vector(F, lst)` → cached `V(lst)` in decoders recovers **~60%** of iterative decoding time immediately.
   - Replacing nested `[[poly(alpha) ...]]` in `generator_matrix()` with evaluation-domain algebra/Vandermonde matmul cuts `GF(256)` single-set construction from `2,835 ms` to `< 15 ms` and unlocks `q = 1024–4096`.
   - Replacing `GRSGaoDecoder` / `GRSErrorErasureDecoder` with a syndrome-based Berlekamp-Massey + Forney decoder eliminates `lagrange_polynomial` (`76%`) and `_polynomial_vanishing_at_alphas` (`35%`), cutting `GF(256)` global decode from `223–406 ms` to `< 1 ms` (`>200×` speedup).
   - Lazy-importing `matplotlib` in `expander.py` and making `simulation/__init__.py` lazy saves `~320–485 ms` on every CLI invocation.
