# B4 — `python-flint` (+ Custom Decoders) Evaluation Report

**Candidate:** `python-flint` `0.9.0` (bundling FLINT `3.6.0`, GMP `6.3.0`, MPFR `4.2.2`) + custom L2/L3/L4 algorithms targeting [B2's `ff_backend` protocol](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/B2_sage_audit.md).  
**Testbed:** Linux x86_64 (`Debian 12`), 2 vCPU, Python `3.11.2`. Isolated prototype venv `/tmp/venv-B4` (`python-flint==0.9.0`, `numpy==2.4.6`); baseline `/home/antigravity/.venv-lrc` (`passagemath 10.8.14`, untouched; `pytest -q` → `34 passed`).  
**Methodology:** Fixed seeds (`42`), warm-up $\ge 3$ ($\ge 1$ for $>0.5\text{ s}$), repeats $\ge 20$ ($3\text{–}5$ for $>0.5\text{ s}$), `time.perf_counter` reporting **median [p10, p90]** and `tracemalloc` peak Python heap; scaling sweep $q \in \{16, 64, 256, 1024, 4096\}$ plus prime (`GF(13)`, `GF(257)`) and odd extension (`GF(81)`, `GF(1369)`) checks.  
**Artifacts:** [flint_backend.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/flint_backend.py), [verify_bit_exact_sage_gen.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/verify_bit_exact_sage_gen.py), [verify_bit_exact_flint_check.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/verify_bit_exact_flint_check.py), [bench_sage.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/bench_sage.py), [bench_flint.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/bench_flint.py), [bench_extra_flint.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/bench_extra_flint.py), [compare_benchmarks.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/compare_benchmarks.py).

---

## Executive Summary

1. **Bifurcated performance profile — extraordinary on L3/L4 polynomials & RS decoding, severe regression on L2 extension-field matrices:**
   - **Polynomials & RS Gao/EE decoding (L3 + L4):** `fq_default_poly` backed by FLINT 3.6.0 C kernels (`fq_zech_poly` / `fq_nmod_poly` with Kronecker substitution) + our subproduct-tree interpolation/evaluation and Half-GCD partial XGCD delivers **42×–472× speedups** over Sage `GRSGaoDecoder` (`1.53 ms` vs `721.9 ms` at `GF(256)`, `54.0 ms` vs `15,495 ms` at `GF(1024)`), unlocks **`GF(4096)` `[4096, 2048]` Gao decoding in `333.8 ms`** (where Sage is $>100\text{ s}$ infeasible), and cuts peak Python memory by **~190×** (`219 KiB` vs `~42 MiB` at `q=1024`).
   - **Critical API gap — `fq_default_mat` does NOT exist in `python-flint 0.9.0`:** Although the bundled `libflint-6839011d.so.24.0.0` exports `fq_default_mat_rref`, `fq_default_mat_nullspace`, and `fq_default_mat_mul`, `python-flint 0.9.0` only wraps `nmod_mat` ($\mathbb{F}_p$) and `fmpz_mod_mat` ($\mathbb{Z}/N\mathbb{Z}$). Extension-field linear algebra must fall back to Python loops over boxed `fq_default` objects (or $m\times$-inflated `nmod_mat` restriction of scalars), making `GF(256) 128×256` RREF **798× slower** than Sage M4RIE (`2,187 ms` vs `2.74 ms`; restriction-of-scalars `nmod_mat`: `618 ms`, still **225× slower**) and `intersect_encoding_bases` **3.3×–6.1× slower** at `q=256`.
   - **Boxed scalar overhead:** `python-flint` has no vectorized/array API and no small-field element cache; individual `fq_default` scalar ops cost **~350 ns/op** (`125×` slower per element than C bulk `fq_default_poly` addition at `2.8 ns/coeff`, and **10× slower** than Sage's cached Givaro scalars for $q \le 256$).
2. **100% bit-exact parity:** FLINT 3.6.0 uses the exact same Conway polynomials as Sage across all tested extension fields (`GF(16)..GF(65536)`, `GF(9)`, `GF(81)`, `GF(121)`, `GF(1369)`). Our prototype passed **1,253 / 1,253** seeded test cases bit-for-bit against Sage (syndrome, `intersect_encoding_bases`, Gao Euclid & Half-GCD, Berlekamp–Welch, Error-Erasure, and subspace enumeration).
3. **Compact footprint & clean install:** `9.7 MiB` wheel (`25.7 MiB` installed, `51.2 ms` import time vs Sage's `2.1 GiB` / `608.5 ms`), PyPI `abi3` wheels across Linux/macOS/Windows for Python `3.10–3.14`. Dual-licensed `MIT` (bindings) + `LGPL-3.0-or-later` (bundled FLINT/GMP/MPFR) — legally compatible with MIT-licensed `tamo-barg-lrc` source distribution (far cleaner than GPL-2.0+ `passagemath`), though with standard LGPL-3.0 relinking caveats if statically bundled.
4. **Verdict:** **Adopt Partially** (optional L3 polynomial & L4 high-degree RS Gao/EE engine paired with a vectorized NumPy/Cython L1–L2 matrix backend) — **Reject as sole full-stack replacement** until upstream wraps `fq_default_mat`. Weighted score: **3.33 / 5.00 (66.5%)**.

---

## 1. Fit Assessment Against B2's `ff_backend` Protocol

| Capability (`ff_backend` Layer) | `python-flint 0.9.0` Native Support | Exact Behavior / Version Gaps Observed | Fit |
|---|---|---|---|
| **L1: $\mathbb{F}_{2^m}, \mathbb{F}_p, \mathbb{F}_{p^m}$ ($q \le 2^{16}+$)** | `fq_default_ctx(p, m)`, `fq_default`, `nmod` | Verified across `GF(2), GF(13), GF(16)..GF(65536), GF(9), GF(81), GF(121), GF(1369)`. **100% Conway-polynomial match with Sage** for all $m \ge 2$. `DEFAULT` uses `FQ_ZECH` (`fq_type=1`) for $q \le 256$ (and `GF(1369)`), switching to `FQ_NMOD` (`fq_type=2`) for `GF(1024)..GF(65536)`; explicit `fq_type='FQ_ZECH'` works up to `GF(2^{16})`. **Quirks:** for $m=1$ (`NMOD`, `fq_type=4`), `ctx.gen()` returns **`0`** (not `1`!), `ctx.modulus()` is `x`, and `primitive_element()` is absent (must check/search). `to_list()` returns `list[fmpz]` (needs Horner at $p$ for `to_int()`, or cached dict). | **Strong (with thin adapter)** |
| **L2: Prime-field matrices ($\mathbb{F}_p$)** | `nmod_mat` (`rref`, `nullspace`, `rank`, `solve`, `inv`, `*`, `transpose`), `fmpz_mod_mat` (no `nullspace`) | Full C FLINT linear algebra over $\mathbb{F}_p$; handles `0`-row matrices cleanly (`nmod_mat(0, m, [], p)`). | **Strong ($\mathbb{F}_p$ only)** |
| **L2: Extension-field matrices ($\mathbb{F}_{p^m}, m \ge 2$)** | **NOT EXPOSED** (`fq_default_mat` absent!) | Verified via `dir(flint)` and `pkgutil.iter_modules(flint.types)`: zero matrix types exist for `fq_default`, `fq_nmod`, `fq_zech`, or `fq`, **even though** `nm -D` confirms `libflint-6839011d.so.24.0.0` exports `fq_default_mat_mul`, `fq_default_mat_rref`, and `fq_default_mat_nullspace`. Forces pure-Python RREF/matvec loops or $m\times$-inflated `nmod_mat` restriction of scalars. | **Fatal Gap** |
| **L3: Univariate polynomials $\mathbb{F}_q[x]$** | `fq_default_poly_ctx`, `fq_default_poly` (51 methods), `nmod_poly` (40), `fmpz_mod_poly` (60) | Native C `+`, `-`, `*`, `divmod`, `%`, `exact_division`, `gcd`, `xgcd`, `derivative`, `left_shift`, `right_shift`, `truncate`, `mul_low`, `mul_mod`, `compose`, `compose_mod`, `factor`, `roots`, scalar `evaluate(x)` / `__call__(x)`. `deg(0) == -1`. **Missing wrappers:** `multipoint_evaluate` (only on `fmpz_mod_poly`, NOT `fq_default_poly`, despite `fq_nmod_poly_evaluate_fq_nmod_vec_fast` in `libflint.so`!), `interpolate`, `product_roots` (`fq_nmod_poly_product_roots` is in `libflint.so`!), and partial/half-GCD stop threshold (`xgcd` always runs to full `gcd == 1`). | **Good (fast C core; needs ~140 LOC tree/HGCD)** |
| **L4: RS Decoders (Gao / BW / EE) & Code Hierarchy** | **None** (computer algebra library, not coding theory) | Custom Gao (subproduct-tree interp/eval + Half-GCD) maps exceptionally well onto `fq_default_poly`. Berlekamp–Welch maps cleanly to `nmod_mat.nullspace()` on $\mathbb{F}_p$, but hits the `fq_default_mat` gap on $\mathbb{F}_{p^m}$. | **Partial (Gao/EE great; BW weak on $\mathbb{F}_{2^m}$)** |
| **L5: Subspace enumeration & misc** | `nmod_mat` + `fmpz.factor()` | No built-in `subspaces(dim)`, but trivial to enumerate RREF pivot matrices via `nmod_mat` or pure Python bitmasks (~30 LOC). | **Adequate** |
| **Vectorization / Batching** | **None** (no NumPy buffer protocol / array ufuncs) | Every `fq_default` is a heap-allocated Cython object (~350 ns/op). Only native C bulk container over $\mathbb{F}_{p^m}$ is `fq_default_poly` (`1000`-coeff add is `0.0028 ms` = `2.8 ns/coeff`, **125.4×** faster than `1000` boxed `fq_default` adds at `0.350 ms`; pack `0.030 ms`, unpack `0.132 ms`). | **Poor** |

---

## 2. Kernel Micro-Benchmarks (`python-flint 0.9.0` vs Sage `passagemath 10.8.14`)

All timings are **median [p10, p90] in ms** on identical seeded inputs (`seed=42`). Speedup $= \text{Sage median} / \text{FLINT median}$ (`>1×` = FLINT faster; `<1×` = FLINT slower).

### 2.1 Finite-Field Scalar & Array Ops (`N = 1,000` ops, ms)

| Field (`fq_type`) | `mul_1k` Sage | `mul_1k` FLINT (Ratio) | `inv_1k` Sage | `inv_1k` FLINT (Ratio) | `pow_1k` Sage | `pow_1k` FLINT (Ratio) |
|---|---:|---:|---:|---:|---:|---:|
| `GF(13)` (`NMOD`) | 0.037 [0.036, 0.057] | 0.353 [0.347, 0.373] (**0.11×**) | 0.037 [0.036, 0.037] | 0.146 [0.146, 0.156] (**0.25×**) | 0.030 [0.029, 0.030] | 0.166 [0.164, 0.180] (**0.18×**) |
| `GF(16)` (`FQ_ZECH`) | 0.036 [0.035, 0.037] | 0.358 [0.348, 0.369] (**0.10×**) | 0.035 [0.035, 0.038] | 0.134 [0.132, 0.143] (**0.27×**) | 0.041 [0.041, 0.043] | 0.191 [0.190, 0.203] (**0.22×**) |
| `GF(64)` (`FQ_ZECH`) | 0.034 [0.033, 0.035] | 0.356 [0.349, 0.369] (**0.09×**) | 0.043 [0.035, 0.046] | 0.133 [0.132, 0.140] (**0.32×**) | 0.050 [0.042, 0.054] | 0.200 [0.198, 0.211] (**0.25×**) |
| `GF(81)` (`FQ_ZECH`) | 0.033 [0.032, 0.034] | 0.355 [0.348, 0.366] (**0.09×**) | 0.037 [0.037, 0.038] | 0.134 [0.133, 0.144] (**0.28×**) | 0.035 [0.034, 0.035] | 0.208 [0.206, 0.219] (**0.17×**) |
| `GF(256)` (`FQ_ZECH`) | 0.034 [0.033, 0.036] | 0.353 [0.341, 0.361] (**0.10×**) | 0.036 [0.036, 0.036] | 0.136 [0.132, 0.159] (**0.26×**) | 0.043 [0.042, 0.051] | 0.218 [0.216, 0.231] (**0.19×**) |
| `GF(1024)` (`FQ_NMOD` def.) | 0.093 [0.093, 0.105] | 0.779 [0.769, 0.786] (**0.12×**) | 0.108 [0.105, 0.116] | 1.237 [1.211, 1.262] (**0.09×**) | 0.112 [0.109, 0.120] | 3.360 [3.317, 3.426] (**0.03×**) |
| `GF(1024)` (`FQ_ZECH` opt.) | 0.093 [0.093, 0.105] | 0.359 [0.347, 0.363] (**0.26×**) | 0.108 [0.105, 0.116] | 0.139 [0.138, 0.151] (**0.78×**) | 0.112 [0.109, 0.120] | 0.242 [0.237, 0.256] (**0.46×**) |
| `GF(4096)` (`FQ_NMOD` def.) | 0.104 [0.103, 0.117] | 0.817 [0.798, 0.827] (**0.13×**) | 0.110 [0.103, 0.117] | 1.516 [1.469, 1.533] (**0.07×**) | 0.116 [0.112, 0.134] | 4.783 [4.699, 4.931] (**0.02×**) |
| `GF(4096)` (`FQ_ZECH` opt.) | 0.104 [0.103, 0.117] | 0.355 [0.333, 0.373] (**0.29×**) | 0.110 [0.103, 0.117] | 0.139 [0.136, 0.143] (**0.79×**) | 0.116 [0.112, 0.134] | 0.253 [0.244, 0.270] (**0.46×**) |

**Takeaway:** Sage's `FiniteField_givaro` pre-allocates an element cache for `order < 500` (`~34 ns/op` pointer return) and drops to `~100 ns/op` at `q=1024, 4096`. `python-flint` heap-allocates a fresh `fq_default` Cython object on every scalar op (`~350 ns/op` for `FQ_ZECH`, jumping to `0.8–4.8 µs/op` under default `FQ_NMOD` at `q >= 1024` due to polynomial XGCD inversion / binary exponentiation). Forcing `FQ_ZECH` speeds up scalar `inv`/`pow` by **9×–19×**, but slows `fq_default_poly` mul/gcd by **1.4×–1.8×** (`19.4 ms` vs `11.1 ms` at `GF(4096)`) because `FQ_NMOD` unlocks FLINT's Kronecker-substitution fast polynomial multiplication.

### 2.2 Univariate Polynomial Kernels (Degree $n - 1$, $n = q$, ms)

| Op | `GF(13)` | `GF(16)` | `GF(64)` | `GF(81)` | `GF(256)` | `GF(1024)` | `GF(4096)` | Slope ($\mathbb{F}_{2^m}$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`poly_mul` Sage** | 0.0004 | 0.139 [0.133, 0.150] | 0.239 [0.236, 0.249] | 0.179 [0.175, 0.197] | 1.455 [1.385, 1.535] | 15.60 [8.01, 17.79] | 66.65 [43.97, 96.01] | $n^{1.19}$ |
| **`poly_mul` FLINT** | 0.0004 (**1.0×**) | 0.0012 (**112×**) | 0.017 (**14.4×**) | 0.027 (**6.7×**) | 0.358 (**4.1×**) | 1.560 (**10.0×**) | 11.44 (**5.8×**) | $n^{1.64}$ |
| **`poly_gcd` Sage** | 0.001 | 0.182 [0.179, 0.196] | 8.897 [4.78, 10.83] | 4.740 [4.70, 9.78] | 154.6 [130.8, 219.0] | 796.9 [728.1, 815.7] | 5,435 (1-shot) | $n^{1.81}$ |
| **`poly_gcd` FLINT** | 0.001 (**1.2×**) | 0.0023 (**78.8×**) | 0.022 (**411×**) | 0.032 (**147×**) | 1.107 (**140×**) | 45.79 (**17.4×**) | 308.3 (**17.6×**) | $n^{2.25}$ |
| **`poly_eval` Sage** | 0.006 | 0.412 [0.406, 0.439] | 13.11 [6.92, 15.15] | 7.127 [6.82, 15.05] | 254.2 [196.0, 288.7] | 7,517 [7,414, 7,517] | INFEASIBLE (`>15 s`) | $n^{2.34}$ |
| **`poly_eval` FLINT (Horner)** | 0.007 (**0.88×**) | 0.009 (**43.8×**) | 0.051 (**256×**) | 0.069 (**104×**) | 0.425 (**598×**) | 289.2 (**26.0×**) | 5,441 (`n/a`) | $n^{2.54}$ |
| **`poly_eval` FLINT (Tree)** | 0.007 (**0.85×**) | 0.010 (**43.1×**) | 0.077 (**171×**) | 0.117 (**61.1×**) | 0.742 (**342×**) | **17.82 (422×)** | **106.0** (`>140×`) | **$n^{1.74}$** |
| **`poly_interp` Sage** | 0.038 | 0.680 [0.624, 0.694] | 17.00 [8.63, 19.06] | 8.315 [8.02, 16.48] | 261.7 [220.9, 296.3] | 6,481 [6,102, 6,481] | INFEASIBLE (`>100 s`) | $n^{2.18}$ |
| **`poly_interp` FLINT (Cold)** | 0.103 (**0.37×**) | 0.154 (**4.4×**) | 0.174 (**97.7×**) | 0.232 (**35.8×**) | 1.210 (**216×**) | 16.91 (**383×**) | **102.6** (`>1000×`) | $n^{1.27}$ |
| **`poly_interp` FLINT (Precomp)** | 0.015 (**2.5×**) | 0.019 (**36.4×**) | 0.088 (**193×**) | 0.116 (**71.6×**) | **0.726 (361×)** | **10.69 (607×)** | **66.15** (`>1500×`) | **$n^{1.53}$** |

**Takeaway:** Replacing Sage's NTL `Polynomial_ZZ_pEX` (GMP limbs + Python `lagrange_polynomial`) with `fq_default_poly` + subproduct trees yields **two to three orders of magnitude** speedup across every polynomial kernel on $\mathbb{F}_{2^m}$ and odd $\mathbb{F}_{p^m}$ (`GF(81)`). Even on `GF(4096)`, full 4096-point interpolation + multipoint evaluation completes in `66–106 ms`.

### 2.3 Matrix RREF, Left Kernel & `intersect_encoding_bases` (ms)

*(Sage matrices copied per call via `M.__copy__()` to bypass Sage's cached RREF.)*

| Workload (`k × n` or LRC Config) | Field Backend (Sage / FLINT) | RREF Sage | RREF FLINT (Ratio) | Left Kernel Sage | Left Kernel FLINT (Ratio) |
|---|---|---:|---:|---:|---:|
| `GF13_32x64` | LinBox / `nmod_mat` (C) | 0.042 [0.040, 0.057] | 0.043 [0.043, 0.043] (**0.97×**) | 0.069 [0.067, 0.080] | 0.019 [0.019, 0.019] (**3.69×**) |
| `GF13_128x256` | LinBox / `nmod_mat` (C) | 0.545 [0.530, 0.561] | 1.495 [1.447, 1.505] (**0.36×**) | 0.741 [0.715, 0.764] | 0.991 [0.965, 0.994] (**0.75×**) |
| `GF257_128x256` | LinBox / `nmod_mat` (C) | 42.02 [20.58, 45.54] | 1.568 [1.521, 1.582] (**26.8×**) | 11.89 [9.75, 12.01] | 1.005 [0.995, 1.828] (**11.8×**) |
| `GF16_8x16` | M4RIE / Python `fq_default` | 0.026 [0.024, 0.036] | 0.501 [0.482, 0.543] (**0.05×**) | 0.066 [0.062, 0.081] | 0.393 [0.381, 0.407] (**0.17×**) |
| `GF64_32x64` | M4RIE / Python `fq_default` | 0.064 [0.059, 0.077] | 33.79 [33.38, 35.14] (**0.002×**) | 0.115 [0.109, 0.130] | 23.76 [23.49, 24.12] (**0.005×**) |
| `GF81_40x81` | Generic Cython / Python `fq_default` | 4.909 [4.421, 9.991] | 67.96 [67.24, 69.58] (**0.07×**) | 3.353 [3.278, 3.441] | 45.93 [45.58, 46.83] (**0.07×**) |
| `GF256_128x256` | M4RIE / Python `fq_default` | **2.741** [1.007, 3.099] | **2,187** [2,170, 2,194] (**0.0013×**) | **0.873** [0.829, 2.997] | **1,485** [1,474, 1,487] (**0.0006×**) |
| `GF1024_64x128` | M4RIE / Python `fq_default` | 3.450 [1.396, 3.525] | 494.7 [491.5, 497.4] (**0.007×**) | 0.989 [0.797, 2.924] | 336.3 [335.2, 340.2] (**0.003×**) |
| `GF4096_64x128` | M4RIE / Python `fq_default` | 8.358 [7.442, 13.09] | 514.4 [512.5, 514.6] (**0.016×**) | 4.436 [4.293, 5.999] | 346.0 [344.8, 348.4] (**0.013×**) |

- **Restriction-of-scalars probe (`nmod_mat` over $\mathbb{F}_2$):** Embedding $\mathbb{F}_{2^m} \hookrightarrow M_{m \times m}(\mathbb{F}_2)$ avoids Python loops (`GF16_8x16` $\to$ `32×64`: `0.061 ms` RREF / `0.042 ms` nullspace; `GF64_32x64` $\to$ `192×384`: `5.65 ms` / `4.55 ms`; `GF256_128x256` $\to$ `1024×2048`: `618.1 ms` / `424.2 ms`), which is **3.5×–8.2× faster** than Python `fq_default` RREF, yet still **88×–225× slower** than Sage's M4RIE (`0.064 ms` / `2.74 ms`) due to the $O(m^3)$ blowup!
- **End-to-end `intersect_encoding_bases` (two-set LRC):**

| Config | Sage Median [p10, p90] (ms) | FLINT Median [p10, p90] (ms) | Speedup |
|---|---:|---:|---:|
| `GF16_c4_r2` | 0.707 [0.685, 0.744] | **0.352** [0.346, 0.370] | **2.01×** |
| `GF64_c8_r6` | 15.79 [15.32, 17.54] | 27.70 [27.31, 29.58] | **0.57×** |
| `GF256_c4_r2` | 151.5 [150.7, 152.3] | 506.7 [504.7, 508.5] | **0.30×** |
| `GF256_c16_r14` | 333.2 [330.5, 335.9] | 2,026 [2,015, 2,030] | **0.16×** |

### 2.4 Matrix–Vector Syndrome & Subspace Enumeration (ms)

| Syndrome Config (`[n, k]`) | Sage `H*vector(F,lst)` (repo) | Sage `H*V(lst)` (cached) | FLINT `matvec` (vs repo / vs `V`) | FLINT `deg(R)<k` (vs repo / vs `V`) |
|---|---:|---:|---:|---:|
| `GF13 [12, 8]` (`nmod_mat`) | 0.0531 | 0.0035 | **0.0060** (**8.80×** / 0.57×) | 0.0143 (3.71× / 0.24×) |
| `GF16 [4, 2]` (local) | 0.0576 | 0.0065 | **0.0050** (**11.6×** / **1.31×**) | **0.0047** (**12.3×** / **1.38×**) |
| `GF64 [8, 4]` (local) | 0.0603 | 0.0093 | 0.0241 (**2.50×** / 0.39×) | **0.0097** (**6.22×** / 0.96×) |
| `GF256 [16, 14]` (local $d=3$) | 0.0622 | 0.0130 | 0.0232 (**2.69×** / 0.56×) | **0.0202** (**3.08×** / 0.64×) |
| `GF256 [16, 8]` (local $d=9$) | 0.0674 | 0.0165 | 0.0930 (0.73× / 0.18×) | **0.0187** (**3.60×** / 0.88×) |
| `GF1024 [32, 16]` (local) | 0.1347 | 0.0800 | 0.6465 (0.21× / 0.12×) | 0.1632 (0.83× / 0.49×) |
| `GF4096 [64, 32]` (local) | 0.4351 | 0.3649 | 2.6427 (0.16× / 0.14×) | 0.4506 (0.97× / 0.81×) |
| `GF256 [256, 128]` (global) | 1.5069 | 1.3201 | 25.123 (0.06× / 0.05×) | **0.7319** (**2.06×** / **1.80×**) |

- **Key algorithmic insight:** Using FLINT's subproduct-tree interpolation (`deg(R) < k`) instead of an $O(n(n-k))$ Python `matvec` cuts global `GF(256)` syndrome checking by **34×** (`25.1 ms → 0.732 ms`) and beats Sage's cached `H * V(lst)` by **1.80×** — AND yields `R(x)` for free if decoding is needed!
- **Subspace enumeration (`find_additive_subgroups`, dim 2):**

| Config (# subspaces) | Sage Median [p10, p90] (ms) | FLINT (`nmod_mat` RREF) (ms) | Speedup |
|---|---:|---:|---:|
| `GF(16)`, dim 2 (35) | 37.67 [23.72, 56.09] | **1.481** [1.442, 1.486] | **25.4×** |
| `GF(64)`, dim 2 (651) | 792.5 [778.7, 808.5] | **31.38** [30.83, 33.43] | **25.3×** |
| `GF(81)`, dim 2 (130) | 153.9 [105.4, 221.6] | **11.72** [11.50, 11.94] | **13.1×** |
| `GF(256)`, dim 2 (10,795) | ~31,800 (A1 extrap.) | **613.3** [588.1, 613.4] | **~52×** |

*(Note: A1 showed pure-Python integer-bitmask enumeration reaches `3.71 ms` at `GF(64)` — 8.5× faster still than `nmod_mat` + boxed `fq_default` mapping.)*

### 2.5 Complete RS Decoding — Length-$n$ Gao, Error-Erasure & Length-$c$ Berlekamp–Welch (ms)

#### Length-$n$ RS Gao Decoding ($n = q$, $k = n/2$, $w = t = \lfloor(n-k)/2\rfloor$ errors)

| Config (`[n, k]`, $t$) | Sage `GRSGaoDecoder` (ms / Peak KiB) | FLINT Gao Euclid (ms) | FLINT Gao Half-GCD (ms / Peak KiB) | Speedup (HGCD vs Sage) |
|---|---:|---:|---:|---:|
| `GF(13) [12, 6]` ($t=3$) | 0.180 [0.173, 0.195] / 6.1 | 0.028 [0.027, 0.031] | **0.028** [0.027, 0.028] / 2.4 | **6.5×** |
| `GF(16) [16, 8]` ($t=4$) | 1.572 [1.542, 1.731] / 5.5 | 0.037 [0.036, 0.038] | **0.037** [0.036, 0.038] / 2.9 | **42.4×** |
| `GF(64) [64, 32]` ($t=16$) | 33.12 [21.62, 46.16] / 29.7 | 0.169 [0.166, 0.182] | **0.177** [0.174, 0.190] / 10.6 | **186.6×** |
| `GF(81) [81, 41]` ($t=20$) | 21.69 [20.40, 43.38] / 45.0 | 0.277 [0.272, 0.287] | **0.279** [0.275, 0.290] / 18.2 | **77.8×** |
| `GF(256) [256, 128]` ($t=64$) | 721.9 [535.8, 731.2] / 339.0 | 1.566 [1.545, 1.579] | **1.528** [1.436, 1.565] / 55.1 | **472.5×** |
| `GF(1024) [1024, 512]` ($t=256$) | 15,495 [10,676, 15,495] / ~42 MiB* | 196.3 [195.0, 198.6] | **53.96** [53.33, 54.49] / 218.8 | **287.2×** |
| `GF(4096) [4096, 2048]` ($t=1024$) | INFEASIBLE (`>100 s`, ~660 MiB*) | 3,226 [3,220, 3,226] | **333.8** [332.7, 343.2] / 865.6 | **>300×** |
| **Empirical $\mathbb{F}_{2^m}$ slope** | **$n^{2.21}$** | **$n^{2.15}$** | **$n^{1.73}$** ($q \ge 256$: $n^{1.94} \to n^{1.32}$ at $1024 \to 4096$) | — |

*\*Sage `q=1024`/`4096` peak memory from B1 profiling (`41.7 MiB` measured at `q=1024`).*  
**Half-GCD payoff:** At `q=1024`, Half-GCD beats Euclidean partial XGCD by **3.6×** (`54.0 ms` vs `196.3 ms`); at `q=4096`, by **9.7×** (`333.8 ms` vs `3,226 ms`, with the XGCD phase itself dropping `21.3×` from `3,217 ms → 151 ms`!).  
**RS Error-Erasure (FLINT HGCD):** `GF(256) [16, 8]` ($w=2, e=2$): **`0.061 ms`**; `GF(256) [256, 16]` ($w=3, e=2$): **`3.632 ms`** vs B1 Sage `GlobalEEDecoder` `405.6 ms` (**~112× faster**); `GF(1024) [1024, 512]` ($w=100, e=50$): **`73.56 ms`** vs A5 Sage `16,321 ms` (**~222× faster**).

#### Length-$c$ (Local Coset) Berlekamp–Welch Decoding

| Config (`[c, r]`, $w$) | Matrix Backend (FLINT) | Sage `GRSBerlekampWelch` (ms) | FLINT BW (ms) | Speedup |
|---|---|---:|---:|---:|
| `GF(13) [12, 8]` ($w=1$) | `nmod_mat.nullspace()` (C) | 0.611 [0.577, 0.636] | **0.134** [0.133, 0.152] | **4.55×** |
| `GF(13) [12, 6]` ($w=3$) | `nmod_mat.nullspace()` (C) | 0.588 [0.553, 0.619] | **0.128** [0.125, 0.131] | **4.59×** |
| `GF(16) [4, 2]` ($w=1$) | Python `fq_default` RREF | 0.800 [0.752, 0.862] | **0.054** [0.052, 0.056] | **14.95×** |
| `GF(64) [8, 4]` ($w=2$) | Python `fq_default` RREF | 1.084 [1.055, 1.148] | **0.255** [0.250, 0.272] | **4.26×** |
| `GF(256) [16, 14]` ($w=1$) | Python `fq_default` RREF | 2.168 [2.038, 3.595] | **1.485** [1.439, 1.511] | **1.46×** |
| `GF(256) [16, 8]` ($w=4$) | Python `fq_default` RREF | 1.907 [1.835, 2.076] | **1.595** [1.559, 1.632] | **1.20×** |
| `GF(1024) [32, 16]` ($w=8$) | Python `fq_default` RREF | 4.893 [4.695, 5.047] | 19.32 [18.94, 20.73] | **0.25×** |
| `GF(4096) [64, 32]` ($w=16$) | Python `fq_default` RREF | 17.40 [16.93, 20.58] | 146.5 [144.3, 155.2] | **0.12×** |

*(Note: B2 proved `GRSGaoDecoder` and `GRSBerlekampWelchDecoder` are 100% behavior-equivalent; replacing local Berlekamp–Welch with FLINT Gao drops `GF(256) [16, 8]` from `1.60 ms → 0.04 ms` and `GF(4096) [64, 32]` from `146.5 ms → ~0.4 ms`!)*

---

## 3. Prototype & Bit-Exact Verification vs Sage

Implemented in [flint_backend.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/flint_backend.py) (~520 LOC):
- **`FlintFiniteField`**: `fq_default_ctx` wrapper handling prime ($m=1$) and extension ($m \ge 2$) fields, deterministic base-$p$ `to_int()`/`from_int()` matching Sage's `.to_integer()`, and primitive element search.
- **(a) Syndrome computation** (`FlintRSCodec.parity_check_matrix`, `syndrome`, `has_zero_syndrome`): dual-GRS Lagrange weights $\eta_j = 1/A'(\alpha_j)$ via formal derivative + subproduct-tree evaluation, safe `0 × n` full-rate guard, plus $O(M(n)\log n)$ `deg(R) < k` polynomial check.
- **(b) `intersect_encoding_bases_flint`**: exact Zassenhaus left-kernel + reverse-column RREF yielding strictly ascending degree-minimal polynomials.
- **(c) RS Decoders (`FlintRSCodec`)**: `decode_gao` (both Euclidean partial-XGCD and $O(M(n)\log n)$ Half-GCD with precomputed subproduct-tree interpolation & multipoint evaluation), `decode_berlekamp_welch`, and `decode_errors_erasures` (punctured Gao + re-evaluation, handling $e=0$, $e=d-1$, $e \ge d$).
- **(d) Subspace enumeration** (`find_additive_subgroups_flint`): RREF pivot enumeration via `nmod_mat`.

**Verification ([verify_bit_exact_sage_gen.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/verify_bit_exact_sage_gen.py) → [verify_bit_exact_flint_check.py](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B4/verify_bit_exact_flint_check.py), `seed=42`):**

| Prototype Component | Fields & Edge Cases Covered | Cases Tested | Bit-Exact Match |
|---|---|---:|---:|
| Syndrome (`H * v` & `is_zero`) | `GF(13, 16, 64, 81, 256, 1369)`; valid codewords, 1–3 errors, random, full-rate $k=n$ | 400 | **400 / 400 (100%)** |
| `intersect_encoding_bases` | Structured two-set LRC (`add/add`, `add/mult`, `mult/mult`) + random bases over `GF(13, 16, 64, 81, 256)` | 150 | **150 / 150 (100%)** |
| RS Gao (Euclid & Half-GCD) | `GF(13, 16, 64, 81, 256, 1024)`; $w=0$, $1 \le w \le t$, beyond-radius failures & misdecodes, $d=2$, $k=n$ | 450 | **450 / 450 (100%)** |
| RS Berlekamp–Welch | All $n \le 64$ RS decode cases above (`nmod_mat` on `GF(13)`, `fq_default` RREF on $\mathbb{F}_{p^m}$) | 430 | **430 / 430 (100%)** |
| RS Error-Erasure | `GF(13, 16, 64, 81, 256)`; $2w+e \le d-1$, $e=d-1$, $e \ge d$, beyond-radius | 250 | **250 / 250 (100%)** |
| Subspace enumeration | `GF(16)` ($s=4$), `GF(64)` ($s=4$), `GF(81)` ($s=9$) | 3 | **3 / 3 (100%)** |
| **Total** | | **1,253** *(1,683 checks w/ BW)* | **100.0%** |

---

## 4. Gap List & Engineering Effort to Fill

| Missing Feature in `python-flint 0.9.0` | Impact on `tamo-barg-lrc` | Workaround / Effort (LOC) |
|---|---|---|
| **1. `fq_default_mat` (matrices over $\mathbb{F}_{p^m}, m \ge 2$)** | **Blocker** for L2: `intersect_encoding_bases`, `encode`, `unencode`, `minimum_distance_py`, `H * v`, Berlekamp–Welch. Python fallback is 200×–1,690× slower than M4RIE. | Pure-Python RREF: ~120 LOC (slow). Fast fix requires **either** pairing with a NumPy/Cython LUT L2 matrix engine (~250 LOC) **or** contributing Cython bindings for `fq_default_mat` upstream (~500 LOC Cython). |
| **2. `fq_default_poly` multipoint eval / interpolation / `product_roots`** | Needed for generator matrix, erasure interpolation, and Gao decoding. | Subproduct-tree routines over `fq_default_poly`: **~100 LOC** (already prototyped & verified; fast $O(M(n)\log n)$!). |
| **3. Partial / Half-GCD stopping at $\deg < \lfloor(n+k)/2\rfloor$** | `fq_default_poly.xgcd()` always runs to `gcd == 1`. | Custom `_hgcd` + `partial_xgcd_hgcd` over `fq_default_poly`: **~90 LOC** (already prototyped & verified; `3.6×–9.7×` faster than Euclid at `q=1024, 4096`). |
| **4. Vectorized / unboxed array API (`GF(2^m)` arrays)** | High boxing cost (`~350 ns/op`) hurts `minimum_distance_py`, batched `encode`, local coset syndromes, and `subgroups` loops. | Cannot be fixed inside `python-flint` without leaving `fq_default` for NumPy integer LUT arrays (~150 LOC). |
| **5. `primitive_element()`, `to_int()`, prime-field `gen() == 0` quirk** | Required by `subgroups.py` (`shift_additive_subgroup`, `list_multiplicative_subgroups`, `field_element_to_int`). | Thin `FlintFiniteField` wrapper: **~90 LOC** (already prototyped). |
| **6. L4 RS decoders & `AbstractLinearCode`/`Encoder`/`Decoder`** | Replaces Sage coding hierarchy + `GRSGao`/`GRSBW`/`GRSErrorErasure`. | `FlintRSCodec` (~150 LOC, done) + lightweight code/encoder/decoder base classes (~150 LOC). |
| **7. Subspace enumeration (`VectorSpace.subspaces`) & `divisors`** | Used in `subgroups.py`. | Bitmask / `nmod_mat` RREF generator + trial-division `divisors`: **~45 LOC**. |
| **Total custom glue (if standalone)** | | **~750–900 LOC** (still slow on L2 unless paired with a non-FLINT L2 matrix engine!) |

---

## 5. Non-Functional Evaluation

| Criterion | Verified Evidence (`python-flint 0.9.0`) | Comparison vs Sage (`passagemath 10.8.14`) |
|---|---|---|
| **License & MIT compatibility** | `License-Expression: MIT AND LGPL-3.0-or-later`. Python bindings: **MIT**; bundled shared libraries (`libflint 3.6.0`, `libgmp 6.3.0`, `libmpfr 4.2.2`): **LGPL-3.0-or-later**. **MIT-compatible for source/pip distribution**: `tamo-barg-lrc` remains MIT-licensed while dynamically linking `python-flint` as an external wheel (LGPL-3.0 §4 permits proprietary/MIT apps using dynamically linked LGPL libs). *Caveat:* static bundling or redistribution of modified wheels must satisfy LGPL-3.0 relinking/source-offer rules (stricter than pure MIT/BSD like NumPy/`galois`). | Massive improvement over `passagemath` (17 wheels **GPL-2.0-or-later** + **GPLv3** `memory_allocator`), which creates a direct copyleft conflict with MIT. |
| **Install footprint (`du -sh`)** | Wheel download: **9.7 MiB** (`libflint` 14.5 MB, `libmpfr` 0.83 MB, `libgmp` 0.74 MB). Installed package: **25.7 MiB** (`9.5M flint` + `16M python_flint.libs`). Total `/tmp/venv-B4` (with `numpy`): **83 MiB** (~28 MiB without NumPy). | **~75× smaller** than `.venv-lrc` (`2.1 GiB`, 26+ Sage wheels, 24,339+ files). |
| **Import time (12 subprocesses)** | `import flint`: **51.20 ms** [50.55, 53.22] (`pass`: `10.78 ms`). `import flint, numpy`: `124.9 ms`. | **11.9× faster** than `import tamo_barg_lrc` (`608.5 ms`) and **22× faster** than `TamoBargCodeTwoSets` (`1,141 ms`). |
| **Wheel availability & build reqs** | Prebuilt `abi3` wheels on PyPI (FLINT/GMP/MPFR bundled — zero system deps!) for **Linux** (`manylinux` x86_64 & aarch64), **macOS** (`arm64`; note `x86_64` CPython CI wheels retired after `0.8.0`), **Windows** (`x86_64`, `arm64`), **Pyodide**, plus `conda-forge`. Source builds require FLINT $\ge 3.0$, C compiler, Cython `3.1–3.2`, Meson/Ninja. | Far better cross-platform coverage (including Windows x86_64/arm64 and Pyodide!). |
| **Python versions** | CPython **3.10–3.14** + **3.14t** (free-threaded) + PyPy 3.11 (`Requires-Python: >=3.10`). | Matches `pyproject.toml`'s `>=3.10` (unlike `passagemath`, which requires `>=3.11,<3.14`). |
| **Pure-Python vs compiled** | Compiled Cython (`abi3`) wrapping C FLINT 3.6.0. Zero runtime dependency on NumPy or Numba (`numpy` coupling = **none**). | Self-contained single wheel. |
| **Maintenance health** | Very healthy (`flintlib/python-flint`, led by Oscar Benjamin & Fredrik Johansson): `0.6.0` (Feb 2024) $\to$ `0.7.0` (Mar 2025) $\to$ `0.8.0` (Aug 2025) $\to$ **`0.9.0` (Jul 2026)**, `0.10.0` in active dev (commits within 3 weeks, 40 open issues, 13 PRs, PEP 561 type stubs). | Active, focused, much smaller blast radius than modular Sage. |

---

## 6. Score Matrix

| Criterion | Weight | Score (0–5) | Weighted | Rationale |
|---|---:|---:|---:|---|
| **Performance** | 25% | **3.0** | 0.750 | 5/5 on L3 polynomials & L4 RS Gao/EE (`42×–607×` faster, `GF(4096)` in `334 ms`), but 1/5 on L2 extension-field matrices (`798×` slower than M4RIE) & L1 boxed scalars (`10×` slower than Givaro). |
| **Feature coverage** | 20% | **2.5** | 0.500 | Full $\mathbb{F}_{p^m}$ field & polynomial coverage + `nmod_mat`, but missing `fq_default_mat`, multipoint eval/interp, partial XGCD, vectorization, and all L4 coding primitives. |
| **Correctness risk** *(inv.)* | 20% | **4.0** | 0.800 | Battle-tested FLINT 3.6.0 C math + exact Conway parity (1,253/1,253 bit-exact vs Sage); -1.0 for custom HGCD/tree/RREF maintenance & `m=1` `gen()==0` quirk. |
| **Install simplicity** | 10% | **4.5** | 0.450 | Single `9.7 MiB` wheel (`1.2 s` install), bundled FLINT/GMP/MPFR, `abi3` wheels for Py 3.10–3.14 across Linux/macOS arm64/Windows; -0.5 for macOS Intel drop & heavy source build. |
| **Maintainability** | 10% | **4.0** | 0.400 | Active releases (`0.9.0` Jul 2026), clean typed API; -1.0 because missing `fq_default_mat` forces ~750+ LOC of local wrappers. |
| **License** | 5% | **3.5** | 0.175 | MIT bindings + LGPL-3.0-or-later bundled C libs: legally compatible with MIT dynamic dependency (beats GPL Sage), but less frictionless than pure MIT/BSD for static redistribution. |
| **Migration effort** *(inv.)* | 10% | **2.5** | 0.250 | Moderate-high (~750–900 LOC custom L2/L3/L4 glue; requires pairing with a separate L2 matrix engine to avoid regressions). |
| **Weighted Total** | **100%** | — | **3.33 / 5.00** *(66.5%)* | |

---

## 7. Verdict

### **ADOPT PARTIALLY** *(optional L3 polynomial & L4 high-degree RS Gao/EE layer only)* — **REJECT as standalone full-stack Sage replacement**

**Reasons:**
1. **Why NOT standalone full adoption:** The absence of `fq_default_mat` and vectorized array ops in `python-flint 0.9.0` is a deal-breaker for L2 linear algebra over $\mathbb{F}_{2^m}$. Replacing Sage with `python-flint` alone would regress `intersect_encoding_bases` by **3.3×–6.1×** at `q=256` (and far worse at `q=1024`), regress matrix RREF/kernel by **225×–1,690×** vs M4RIE, and cripple `minimum_distance_py` and batched encoding/syndromes.
2. **Where `python-flint` shines (partial adoption):** If `tamo-barg-lrc` adopts a clean `ff_backend` split — e.g., a fast unboxed NumPy/Cython/LUT engine for **L1/L2** ($\mathbb{F}_{2^m}$ vectors, matrices, RREF, local syndromes, `minimum_distance_py`) — then `python-flint` is an **elite drop-in accelerator for L3 polynomials and L4 global RS Gao / Error-Erasure decoding** at large block lengths (`n = 256, 1024, 4096+`) and odd-characteristic extensions (`GF(81)`, `GF(1369)`): **287×–607× faster** than Sage with **~190× lower peak memory**, **100% Conway-polynomial parity**, `51 ms` import time, and `25.7 MiB` disk footprint.
3. **Trigger to re-evaluate for full adoption:** Upstream `python-flint` wrapping FLINT's already-bundled `fq_default_mat` (`fq_default_mat_rref`, `fq_default_mat_nullspace`, `fq_default_mat_mul`, `fq_default_mat_solve`) and `fq_nmod_poly_evaluate_fq_nmod_vec_fast` would immediately close the L2 gap and raise `python-flint`'s score above `4.2 / 5.0`.
