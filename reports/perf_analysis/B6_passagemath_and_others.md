# B6 — Slimmed `passagemath`, Ecosystem Candidates, Hybrid Architecture & Native-Extension Evaluation

**Scope:** Evaluation of (1) slimming modular `passagemath` to the minimal and optimal wheel sets, (2) full SageMath via conda-forge/system, (3) SymPy, (4) Python finite-field & coding-theory libraries (`pyfinite`/`ffield`, `reedsolo`, `komm`, `unireedsolomon`, `pyldpc`, `zfec`, `bchlib`, `gf256`, `scikit-dsp-comm`, `finitefield`), (5) hybrid `ff_backend` plugin architecture (with a verified working prototype), and (6) Rust (`PyO3`) / C++ (`pybind11`/`nanobind`) native-extension effort–benefit estimates.

**Environment & Methodology:**
- **Host:** Linux x86_64 (Debian 12), `Intel(R) Xeon(R) Platinum 8481C`, `nproc = 2`, Python `3.11.2`, `uv 0.13.8`.
- **Isolation:** Baseline `/home/antigravity/.venv-lrc` (`passagemath 10.8.14`, `2,082 MB`) untouched; all candidate installations run in isolated `/tmp/venv-B6-*` environments. Baseline `pytest -q` re-verified `34 passed` at completion.
- **Benchmark protocol:** Fixed seeds (`random.seed(42)`, `np.random.seed(42)`, `set_random_seed(42)`), warm-up $\ge 3$ calls, repeats $\ge 20$ ($\ge 5$ for $>0.8\text{ s}$ calls), reporting `median [p10, p90]` via `time.perf_counter()` across $q \in \{16, 64, 256, 1024, 4096\}$. Note: sibling swarm agents shared the 2 vCPUs during parts of the wall-clock import and `q >= 1024` runs; contention-sensitive ratios are cross-checked against single-core baselines from [B1_profiling.md](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/B1_profiling.md) and [B2_sage_audit.md](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/B2_sage_audit.md).
- **Artifacts & prototypes:** All scripts, PyPI metadata dumps, benchmark JSONs, and bit-exact verification suites live in [`reports/perf_analysis/scratch/B6/`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/).

---

## Executive Summary

1. **Surprise `passagemath` Discovery — Smaller *and* 2×–15× Faster With Zero Code Changes:**
   - There is **no** `passagemath-coding` wheel on PyPI (`HTTP 404`); `sage.coding` (`GeneralizedReedSolomonCode`, `channel`, `AbstractLinearCode`) ships inside **`passagemath-modules`**.
   - The **absolute minimal** wheel set passing all 34 tests is **`passagemath-modules[gf]`** (`passagemath-modules` + `passagemath-pari`, 5 `passagemath-*` wheels, **`1,025 MB` vs `2,082 MB`**, **-50.8% disk**). `passagemath-pari` cannot be dropped because `sage/rings/polynomial/polynomial_ring.py:3628` unconditionally imports `sage.libs.pari` inside `irreducible_element()` before consulting `conway-polynomials`.
   - However, bare `modules+pari` drops M4RIE (`left_kernel_64x256` regresses **144×**, `0.36 ms → 51.2 ms`). The **sweet-spot wheel set** is **`passagemath-modules[m4rie]` + `passagemath-brial`** (7 `passagemath-*` wheels, **`1,089 MB`**, **-47.7% disk**, dropping `groups`, `gap`, `flint`, `ntl`, `linbox`, `iml`, `combinat`, `repl`, Jupyter):
     - Dropping `passagemath-ntl` (and `flint`) escapes Sage's catastrophic `Polynomial_ZZ_pEX` (NTL arbitrary-precision GMP) default over $\mathbb{F}_{2^m}$, accelerating `GF(256)` `two_set_init` by **5.92×** (`306.1 ms → 51.7 ms`), `GlobalDecoder` by **7.8×–15.8×** (`215.3 ms → 13.6–27.7 ms`), `poly_gcd` by **32.7×**, and `poly_eval_256` by **14.3×**.
     - Adding `passagemath-brial` (+48 MB) eliminates the B1 `sequence.py:297` `LazyImport(BooleanMonomialMonoid)` `fstatat64` storm, speeding up `H * vector(F, ...)` by **3.25×** (`55.5 µs → 17.1 µs`) and `IterativeDecoder` by **1.93×** (`9.25 ms → 4.80 ms`).
   - **Remaining blockers for `passagemath`:** All `passagemath-*` wheels are **GPL-2.0-or-later** (MIT conflict), require Python `>=3.11,<3.15` (breaking `pyproject.toml`'s `>=3.10`), still occupy **>1.0 GiB**, and **lack Windows (`win_amd64`) wheels** for `passagemath-pari` and `passagemath-brial`.
2. **All Other Non-B3/B4/B5 Python Libraries Fail Fit or Performance — Reject Quickly:**
   - **SymPy (`1.14.0`, BSD):** Native `GF(q)` is **prime fields only** (`GF(16)` silently builds $\mathbb{Z}/16\mathbb{Z}$ with zero divisors `2 * 8 == 0`!). Hand-rolled $\mathbb{F}_{2^m}$ via `sympy.polys.galoistools` is **830×–3,276× slower** than Sage Givaro, and `DomainMatrix.rref()` on `64×256` over `GF(257)` takes **`1,236.5 ms`** (**>2,500× slower** than M4RIE).
   - **`pyfinite` (`1.9.1`, MIT):** Binary-only, erasure-only RS, **40×–225× cliff** above `GF(256)`, sdist-only, and **silently pollutes the working directory with `ffield.lut.<m>` files**.
   - **`reedsolo` (`1.7.0`, Public Domain) & `unireedsolomon` (`1.0.6`, MIT):** Binary-only, cyclic BCH-view ($n \le 2^m - 1$, cannot include $0$ or arbitrary coset evaluation points), global non-thread-safe tables; `unireedsolomon` also crashes with **`IndexError` in `ff.py:228`** on `GF(64)`/`GF(1024)`.
   - **`komm` (`0.37.0`, GPL-3.0-only):** Hard license conflict, binary-only, bit-expanded cyclic RS only, **126×–1,455× slower** than Sage Givaro.
   - **`pyldpc` (`0.7.9`, MIT):** Abandoned (Feb 2020), fails PEP 517 isolated install, binary LDPC only.
3. **Recommended Architecture — Hybrid `ff_backend` Plugin (`4.53 / 5.00`, Adopt Partially / As Migration Bridge):**
   - Decouple the codebase via B2's `ff_backend` `Protocol` and `importlib.metadata.entry_points(group="tamo_barg_lrc.backends")`. Ship a pure-MIT default backend (from B3/B4/B5 or our LUT/NumPy codec) and expose slimmed `passagemath-modules[m4rie]` + `passagemath-brial` as an **opt-in `[sage]` extra** for differential verification and legacy Sage users.
   - Our scratch prototype ([`hybrid_prototype.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/hybrid_prototype.py)) — using plain Python + NumPy LUT tables and syndrome-based Sugiyama + Forney decoding with exact $\alpha = 0$ handling — passed **4,540 / 4,540 bit-exact differential tests (0 mismatches)** vs Sage across `GF(16)`, `GF(64)`, `GF(256)`, `GF(13)`, and `GF(81)` while running **100× faster** on `GF(256)` length-256 RS decode (`222.86 ms → 2.23 ms`) and **24.5× faster** on local length-8 RS decode (`1.151 ms → 0.047 ms`).

---

## 1. Candidate 1 — Modular `passagemath` Slimming Analysis

### 1.1 Wheel Discovery & Iterative Dropping Results

PyPI queries ([`pypi_probe.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/pypi_probe.py) $\to$ [`pypi_metadata.json`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/pypi_metadata.json)) confirmed:
- **`passagemath-coding`**, **`passagemath-coding-theory`**, and **`passagemath-gf2x`** do **not** exist on PyPI (`HTTP 404`).
- All of `sage.coding.*` (`AbstractLinearCode`, `Encoder`, `Decoder`, `GeneralizedReedSolomonCode`, `random_error_vector`) is bundled directly inside **`passagemath-modules`**, which also compiles Cython Givaro (`FiniteField_givaro`) directly into the wheel.
- `passagemath-modules` defines built-in extras: `[gf]` (`= [pari]`), `[m4ri]` (`passagemath-m4ri-m4rie`), `[m4rie]` (`= [m4ri, pari]`), `[flint]`, `[ntl]`, `[linbox]`, `[standard]`.

Testing progressively slimmer isolated venvs against the full 34-test suite (`pytest -q`):

| Configuration | Explicit `passagemath` Spec | `passagemath-*` Wheels | Total Pkgs | Venv Size (`du -sm`) | Delta vs `.venv-lrc` | Install Time (`uv` cached) | 34 Tests Pass? | `GF(256)` Backends (Field / Poly / Matrix) | `GF(65536)` Works? |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **Baseline `.venv-lrc`** | `modules, flint, pari, groups` (+ transitives) | 17 | 76 | `2,082 MB` | baseline | ~25–40 s (cold) | **34/34** | Givaro / **NTL `ZZ_pEX`** / M4RIE | Yes (`ntl_gf2e`) |
| **Bare `modules` only** | `passagemath-modules` | 4 | 27 | `719 MB` | -65.5% | `6.25 s` | **0/34 FAIL**¹ | N/A (`GF(p^m)` crashes) | No |
| **Minimal (`[gf]`)** | `passagemath-modules[gf]` (`modules + pari`) | **5** | 29 | **`1,025 MB`** | **-50.8%** | `8.56 s` | **34/34** | Givaro / **`generic_dense`** / `generic_dense` | Yes (`pari_ffelt`) |
| **Minimal + `brial`** | `passagemath-modules[gf]`, `passagemath-brial` | 6 | 30 | `1,073 MB` | -48.5% | `5.31 s`² | **34/34** | Givaro / `generic_dense` / `generic_dense` | Yes (`pari_ffelt`) |
| **Sweet-spot (`[m4rie]+brial`)** | `passagemath-modules[m4rie]`, `passagemath-brial` | **7** | 31 | **`1,089 MB`** | **-47.7%** | `5.67 s`² | **34/34** | Givaro / **`generic_dense`** / **M4RIE** | Yes (`pari_ffelt`) |

- ¹ **Why `passagemath-pari` is strictly irreducible:** In `sage/rings/polynomial/polynomial_ring.py:3628`, `PolynomialRing_dense_mod_p.irreducible_element()` executes `from sage.libs.pari import pari` unconditionally on line 1, *before* checking `exists_conway_polynomial(p, n)`. Even though `conway-polynomials` suffices for all fields $q \le 4096$, omitting `passagemath-pari` raises `ModuleNotFoundError: No module named 'sage.libs.pari'` on every `GF(p^m)` call.
- ² Measured with warm `uv` wheel cache on Linux (`passagemath-brial` download `13.6 MiB` took `1.13 s`). Cold network download for the 7-wheel sweet-spot set is ~`180–210 MiB` compressed wheels (`~15–30 s` on broadband).

**Wheels successfully eliminated without breaking any test:**
`passagemath-groups`, `passagemath-gap`, `passagemath-gap-pkg-{smallgrp,primgrp,factint}-data`, `passagemath-repl` (+ 18 Jupyter/IPython/widget wheels), `passagemath-flint`, `passagemath-ntl`, `passagemath-linbox`, `passagemath-iml`, `passagemath-combinat` (**10 `passagemath-*` wheels + ~25 transitive packages dropped, saving `993 MB`**).

### 1.2 Wheel Availability Across OS & Python Versions (PyPI Verified `10.8.14`)

| Wheel | `Requires-Python` | Wheel Python Tags | Linux `x86_64` / `aarch64` (`manylinux`+`musllinux`) | macOS `arm64` / `x86_64` | Windows `win_amd64` | Compressed Wheel Size | License |
|---|---|---|---|---|---|---:|---|
| `passagemath-modules` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **Yes** | `18.8–82.0 MB` | GPL-2.0-or-later |
| `passagemath-categories` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **Yes** | `10.2–50.8 MB` | GPL-2.0-or-later |
| `passagemath-objects` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **Yes** | `4.0–20.6 MB` | GPL-2.0-or-later |
| `passagemath-m4ri-m4rie` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **Yes** | `1.2–4.6 MB` | GPL-2.0-or-later |
| `passagemath-environment` | `>=3.11, <3.15` | `py3` | Yes (pure) | Yes (pure) | **Yes** | `0.14 MB` | GPL-2.0-or-later |
| `passagemath-pari` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **NO** | `28.7–74.5 MB` | GPL-2.0-or-later |
| `passagemath-brial` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **NO** | `2.2–14.2 MB` | GPL-2.0-or-later |
| `passagemath-flint` / `ntl` / `linbox` / `gap` | `>=3.11, <3.15` | `cp311–cp314` | Yes | Yes | **NO** | `4–88 MB` each | GPL-2.0-or-later |

**Platform & version blockers:**
1. **No Python 3.10 wheels** (conflicts with `pyproject.toml`'s `requires-python = ">=3.10"`).
2. **No native Windows wheels** for `passagemath-pari` (required!) or `passagemath-brial`, so Windows users still require WSL2 even with slimmed `passagemath`.
3. **Copyleft license (`GPL-2.0-or-later` + `GPLv3` `memory_allocator`)** persists across every wheel.

### 1.3 Performance Comparison: Full `.venv-lrc` vs Slimmed `passagemath` Variants

Measured via [`bench_sage_kernels.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/bench_sage_kernels.py) (`sage_full.json`, `sage_minimal.json`, `sage_min_brial.json`, `sage_min_m4ri_brial.json`, `sage_clean_rref_imports.json`):

#### End-to-End Workloads & Import Time (`GF(256)`, `[256, 16, r=[2,2], d=[3,3]]`)

| Metric | Full `.venv-lrc` (17 wheels, `2.08 GB`) | Minimal `modules+pari` (5 wheels, `1.03 GB`) | Minimal + `brial` (6 wheels, `1.07 GB`) | **Sweet-spot `[m4rie]+brial`** (7 wheels, `1.09 GB`) | Speedup (`[m4rie]+brial` vs Full) |
|---|---:|---:|---:|---:|---:|
| `import tamo_barg_lrc` (idle / loaded) | `596 ms`¹ / `1,331 ms` | **`525.9 ms`** / `1,119 ms` | `880 ms` (loaded) | `~550 ms`¹ / `1,129 ms` | **~1.10–1.18×** (drops `sage.all__sagemath_flint` `107 ms`) |
| `two_set_init_gf256` | `306.1 [301.8, 310.2]` ms | `143.9 [143.2, 146.3]` ms | `294.9` ms (no M4RIE, loaded) | **`51.7 [50.9, 54.2]` ms** | **5.92× faster** |
| `two_set_iter_decode_w3_gf256` | `9.25 [9.00, 10.91]` ms | `7.95 [7.83, 8.00]` ms | `5.45 [4.91, 7.05]` ms | **`4.80 [4.68, 6.83]` ms** | **1.93× faster** |
| `two_set_global_decode_w3_gf256` | `215.35 [215.1, 236.2]` ms | **`13.60 [13.5, 14.2]` ms** | `28.28` ms (loaded) | **`27.68 [26.0, 29.4]` ms** (loaded; `~14 ms` idle) | **7.78×–15.8× faster** |

*(¹ Idle `full` import `596 ms` from B1; `min` idle measured `525.9 ms` (`[508.5, 572.7]`); `brial`/`m4ri_brial` runs overlapped sibling swarm CPU load, inflating wall times by ~1.8–2× on long ops.)*

#### Kernel Scaling Sweep ($q \in \{16, 64, 256, 1024, 4096\}$, Median ms)

| Kernel | `q=16` Full / Min / **Opt** | `q=64` Full / Min / **Opt** | `q=256` Full / Min / **Opt** | `q=1024` Full / Min / **Opt** | `q=4096` Full / Min / **Opt** | Key Mechanism |
|---|---|---|---|---|---|---|
| `gf_mul_1k` | `0.034 / 0.035 / 0.034` | `0.038 / 0.032 / 0.033` | `0.034 / 0.034 / 0.036` | `0.101 / 0.100 / 0.099` | `0.091 / 0.098 / 0.100` | Identical Givaro C++ Zech LUT in all 3 |
| `gf_inv_1k` | `0.017 / 0.018 / 0.017` | `0.019 / 0.018 / 0.017` | `0.018 / 0.017 / 0.019` | `0.080 / 0.077 / 0.080` | `0.074 / 0.079 / 0.080` | Identical Givaro LUT (`elem_cache` cliff at $q \ge 500$) |
| `poly_mul` ($\deg = q/4$) | `0.013 / 0.002 / 0.002` | `0.297 / 0.011 / 0.011` | `0.344 / 0.103 / 0.097` | `2.05 / 2.83 / 6.02` | `10.61 / 26.84 / 55.42` | No-NTL wins **3.5×–28×** for $\deg \le 64$; NTL Karatsuba wins `2.5×` at $\deg = 1024$ |
| `poly_gcd` ($\deg = q/4$) | `0.026 / 0.010 / 0.009` | `0.412 / 0.036 / 0.036` | **`8.33 / 0.275 / 0.255`** | `171.8 / 19.3 / 19.0` | `898.9 / 221.1 / 497.5` | **32.7× faster** at $q=256$ without NTL marshaling |
| `poly_eval` ($\min(q,1024)$ pts) | `0.279 / 0.022 / 0.020` | `2.76 / 0.124 / 0.124` | **`45.46 / 3.23 / 3.17`** | `939.5 / 78.3 / 82.1` | `5,256 / 163.5 / 341.5` | **14.3×–32.2× faster** without Givaro$\leftrightarrow$NTL `ZZ_pE` conversion |
| `poly_interp` ($n=\min(q,256)$) | `0.630 / 0.075 / 0.073` | `8.73 / 0.669 / 0.660` | **`168.6 / 15.7 / 15.6`** | `245.8 / 36.3 / 38.4` | `338.1 / 18.4 / 39.5` | **10.8×–18.4× faster** without NTL `Polynomial_ZZ_pEX` |
| `rref` uncached ($k \times n$, $n=\min(q,256)$) | `0.013 / 0.014 / 0.011` | `0.033 / 0.651 / 0.030` | **`0.491 / 69.7 / 0.506`** | `6.86 / 232.9 / 4.92` | `19.65 / 233.4 / 19.31` | Bare Min regresses **142×**; **Opt restores 100% M4RIE speed** |
| `left_kernel` ($k \times n$) | `0.054 / 0.038 / 0.049` | `0.073 / 0.576 / 0.067` | **`0.356 / 51.2 / 0.374`** | `0.866 / 128.2 / 1.61` | `2.39 / 76.8 / 4.42` | Bare Min regresses **144×**; **Opt restores M4RIE** |
| `syndrome_2x4_vec` (`vector()`) | `0.055 / 0.056 / 0.017` | `0.053 / 0.056 / 0.017` | **`0.056 / 0.056 / 0.017`** | `0.058 / 0.056 / 0.018` | `0.061 / 0.058 / 0.018` | **`brial` cures `pbori` `fstatat64` storm (3.25× faster)** |
| `syndrome_2x4_Vcall` (`V(lst)`) | `0.006 / 0.006 / 0.006` | `0.006 / 0.007 / 0.006` | `0.006 / 0.006 / 0.007` | `0.007 / 0.008 / 0.007` | `0.007 / 0.008 / 0.007` | Code-level fix bypasses `sequence.py` (9× vs `vector()`) |
| `rs_gao` ($n=\min(q,256), k=n/4$) | `1.27 / 0.304 / 0.219` | `12.81 / 3.18 / 3.09` | **`222.9 / 22.6 / 23.2`** | `408.1 / 110.5 / 193.8` | `522.6 / 77.5 / 181.6` | **9.6×–9.9× faster** at $q=256$ (still $O(n^2)$ Python Gao) |
| `rs_bw` ($n=8, k=4$, 2 err) | `0.900 / 1.81 / 0.468` | `1.073 / 0.955 / 0.461` | **`1.151 / 0.811 / 0.468`** | `1.247 / 0.733 / 0.453` | `1.376 / 0.764 / 0.480` | **2.46×–2.87× faster** (M4RIE kernel + no `pbori` + no NTL) |

*(Opt = `passagemath-modules[m4rie]` + `passagemath-brial`.)*

### 1.4 Expressing Slimmed `passagemath` as an Optional Extra in `pyproject.toml`

```toml
[project.optional-dependencies]
# Recommended slimmed SageMath reference/compatibility extra (~1.09 GiB, 2x-15x faster than full stack).
# Note: GPL-2.0-or-later; requires Python >=3.11,<3.15; Linux & macOS wheels only (WSL2 on Windows).
sage = [
    "passagemath-modules[m4rie]>=10.8,<11",
    "passagemath-brial>=10.8,<11",
]
# Bare minimum (~1.03 GiB, omits M4RIE and brial — NOT recommended due to 144x RREF regression):
sage-minimal = [
    "passagemath-modules[gf]>=10.8,<11",
]
```

*(Optionally, once `tamo-barg-lrc` avoids calling `vector(F, ...)` directly in favor of `V(lst)`, `passagemath-brial` can also be dropped, saving another 48 MB.)*

---

## 2. Candidate 2 — Full SageMath (conda-forge / System Package)

| Dimension | Assessment |
|---|---|
| **Install footprint** | `3.5–5.5 GiB` unpacked (`conda-forge::sage` or distro `sagemath`), hundreds of native C/C++/Fortran libraries (GAP, Singular, Maxima, PARI, FLINT, NTL, LinBox, PPL, R/GSL, Jupyter). |
| **CI cost** | GitHub Actions cold environment setup takes **3–6 minutes** (`mamba`/`pixi`) and blows standard cache quotas; Docker images exceed `2–4 GiB`. |
| **User friction** | Forces `sage -python` or a dedicated Conda environment; cannot be `pip install tamo-barg-lrc`'d into a standard Python/uv/Poetry project; **no native Windows support** (WSL2 required). |
| **Performance** | Identical to baseline `.venv-lrc` — still defaults to NTL `Polynomial_ZZ_pEX` over $\mathbb{F}_{2^m}$ (so `GlobalDecoder` is **15× slower** than slimmed `passagemath-modules[m4rie]` unless code explicitly passes `implementation="generic"`!). Only advantage: ships `libgap` so `code.weight_distribution()` / `code.minimum_distance()` use GAP instead of the pure-Python fallback. |
| **License** | **GPL-3.0-or-later** (strict copyleft, incompatible with distributing a permissive MIT bundle). |
| **Verdict** | **Reject as primary dependency**; keep compatibility only for users who already run inside `sage -python`. |

---

## 3. Candidate 3 — SymPy (`1.14.0`, BSD)

### 3.1 Capability Fit & Critical Correctness Trap

Empirical probing ([`sympy_triage.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/sympy_triage.py)) exposed **four disqualifying architectural gaps**:
1. **No first-class $\mathbb{F}_{p^m}$ (`m >= 2`) domain — and a dangerous silent bug:** Calling `sympy.polys.domains.GF(16)` or `Poly(..., modulus=16)` does **not** raise an error and does **not** construct $\mathbb{F}_{2^4}$! It silently constructs $\mathbb{Z}/16\mathbb{Z}$ (`is_Field == False`, `GF(16)(2) * GF(16)(8) == 0 mod 16`, `1 / GF(16)(2)` raises `NotInvertible: zero divisor`). Furthermore, `GF(13).algebraic_field(...)` raises `DomainError: Cannot create algebraic field over GF(13)`.
2. **Only `GF(p)` is native:** Prime fields $\mathbb{F}_p$ work via `GF(p, symmetric=False)` + `DomainMatrix` (RREF, nullspace, matmul) + `sympy.polys.galoistools`.
3. **Extension fields $\mathbb{F}_{p^m}$ require hand-rolling on `galoistools`:** Representing elements of $\mathbb{F}_{2^m}$ as dense `ZZ` lists modulo a Conway polynomial via `gf_mul`/`gf_rem`/`gf_gcdex` works mathematically, but forfeits `DomainMatrix`, `Poly`, and vectorization.
4. **Zero coding-theory primitives:** No `RSCodec`, no Gao/Berlekamp–Welch/ErrorErasure decoders, no subspace enumeration, no NumPy array acceleration.

### 3.2 SymPy Kernel Micro-Benchmarks vs SageMath

From [`bench_sympy.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/bench_sympy.py) (`sympy_bench.json`; import time `292.7 [289.2, 297.7]` ms, venv `106 MB` with NumPy / `~40 MB` standalone):

| Kernel | Field | SymPy Median `[p10, p90]` (ms) | Sage Full Baseline (ms) | Ratio (Sage / SymPy) |
|---|---|---:|---:|---:|
| `gf_mul_1k` | `GF(13)` (native) | `0.405 [0.392, 0.422]` | `~0.034` | **0.08× (12× slower)** |
| `gf_inv_1k` | `GF(13)` (native) | `1.794 [1.772, 1.815]` | `~0.018` | **0.01× (100× slower)** |
| `rref_64x256` | `GF(257)` (`DomainMatrix`) | **`1,236.5 [1,118.9, 1,414.9]`** | `0.491` (`GF(256)` M4RIE) | **0.0004× (2,518× slower)** |
| `left_kernel_64x256` | `GF(257)` (`DomainMatrix`) | **`260.0 [221.9, 267.8]`** | `0.356` (`GF(256)` M4RIE) | **0.0014× (730× slower)** |
| `syndrome_2x4` | `GF(257)` (`DomainMatrix`) | `0.0145 [0.0134, 0.0181]` | `0.0040` (prevec) / `0.0555` (`vector()`) | `0.28×` / `3.8×` |
| `gf_mul_1k` | `GF(16)` (`galoistools`) | `6.08 [5.94, 12.06]` | `0.034` | **0.0056× (179× slower)** |
| `gf_mul_1k` | `GF(256)` (`galoistools`) | **`28.30 [14.40, 29.78]`** | `0.034` | **0.0012× (832× slower)** |
| `gf_inv_1k` | `GF(256)` (`galoistools`) | **`58.98 [52.05, 97.05]`** | `0.018` | **0.0003× (3,277× slower)** |
| `gf_pow_1k` | `GF(256)` (`galoistools`) | **`107.70 [77.45, 156.28]`** | `0.039` | **0.0004× (2,762× slower)** |
| `poly_mul` ($\deg 64$) | `GF(256)` (`galoistools`) | **`65.15 [64.67, 109.59]`** | `0.344` (full) / `0.097` (opt) | **0.0053× (189×–672× slower)** |
| `gf_inv_1k` | `GF(4096)` (`galoistools`) | **`207.05 [204.27, 209.94]`** | `0.074` | **0.0004× (2,798× slower)** |

### 3.3 SymPy Prime-Field Prototype Verification

Despite being unusable for $\mathbb{F}_{2^m}$ performance, we verified a clean SymPy `GF(13)` prototype ([`sympy_prototype_verify.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/sympy_prototype_verify.py)) implementing (a) `DomainMatrix` syndrome, (b) Zassenhaus `intersect_encoding_bases`, and (c) `galoistools` RS Gao decoding against **1,050 seeded SageMath reference cases** (`400` syndrome + `300` basis intersection + `350` RS Gao including beyond-radius): **1,050 / 1,050 passed (0 mismatches)**.

**Verdict for SymPy:** **Reject.** Lacks native $\mathbb{F}_{p^m}$ support (`GF(16)` trap) and is **2–3 orders of magnitude slower** than Sage on every finite-field and matrix kernel.

---

## 4. Candidate 4 — Other Python Coding & Finite-Field Libraries (Triage & Benchmarks)

### 4.1 Capability Triage Table (All Verified via PyPI & Isolated Install)

| Library (Version) | License (MIT-compat?) | Wheel / Build | Last Release | `GF(2^m)` | `GF(p)` / Odd `GF(p^m)` | Poly Ring | Matrices (RREF/Ker) | Arbitrary-Eval RS (Gao/BW/EE) | Disqualifying Evidence |
|---|---|---|---|---|---|---|---|---|---|
| **`pyfinite`** `1.9.1` (`ffield`) | MIT (Yes) | sdist only (+ `msgpack-python`) | Nov 2022 | Yes (LUT cliff $>2^8$) | **No** | **No** | Basic (`GenericMatrix`, no kernel) | **No** (erasure-only RS) | Binary-only; no error decoding; **40×–225× slowdown at $q \ge 1024$**; **silently writes `ffield.lut.<m>` files to CWD!** |
| **`ffield`** (standalone) | N/A | HTTP 404 | N/A | — | — | — | — | — | Does not exist on PyPI; it is a submodule of `pyfinite`. |
| **`reedsolo`** `1.7.0` | Public Domain / MIT-0 (Yes) | Pure `py3` wheel | Jan 2023 | Yes (`c_exp` up to 12+) | **No** | Basic `gf_poly_*` | **No** | **No** (cyclic BCH-view $n \le q-1$ only) | Binary-only; cannot include $0 \in \mathbb{F}_q$ or arbitrary coset eval points; **global mutable LUTs (not thread-safe, 1 field per process)**. |
| **`komm`** `0.37.0` | **GPL-3.0-only (NO)** | Pure `py3` wheel (`numpy`, `tqdm`) | Oct 2026 | `FiniteBifield(m)` | **No** | `BinaryPolynomial` ($\mathbb{F}_2[x]$ only) | **No** | **No** (bit-expanded cyclic RS only) | **Hard GPL-3.0 license violation**; binary-only; `ReedSolomonCode(4, 5)` has `n=60, k=44` (bit-level cyclic); **126×–1,455× slower** than Sage Givaro. |
| **`unireedsolomon`** `1.0.6` | MIT (Yes) | `cp310 win_amd64` only; sdist else | Apr 2023 | Yes (buggy!) | **No** | Basic `Polynomial` | **No** | **No** (cyclic BCH-view only) | Binary cyclic-only; **verified upstream crash `IndexError` in `ff.py:228`** when calling `GF2int.inverse()` after switching to `GF(64)` or `GF(1024)`! |
| **`pyldpc`** `0.7.9` | MIT (Yes) | **Build FAILED** (sdist) | Feb 2020 | **No** ($\mathbb{F}_2$ only) | **No** | **No** | Binary only | **No** (LDPC only) | Abandoned (2020); fails PEP 517 isolated install (`setup.py` imports undeclared `numpy`); zero RS or $\mathbb{F}_q$ support. |
| **`zfec`** `1.6.0.0` | **GPL-2+ (NO)** | Binary wheels | Nov 2024 | `GF(2^8)` only | **No** | **No** | **No** | **No** (erasure-only) | Hard GPL-2+ conflict; fixed `GF(256)` erasure coding only. |
| **`bchlib`** `2.1.3` | **GPLv2 (NO)** | Binary wheels | Sep 2024 | Binary BCH only | **No** | **No** | **No** | **No** | Hard GPLv2 conflict; Linux kernel binary BCH wrapper only. |
| **`scikit-dsp-comm`** `2.1.2` | BSD (Yes) | Pure `py3` wheel | Apr 2026 | **No** | **No** | **No** | **No** | **No** | Signal processing & convolutional/Viterbi comm sims; no $\mathbb{F}_q$ or RS. |
| **`gf256`** `0.2.0` / **`finitefield`** `0.4.2` | BSD / Unknown | sdist only | 2016 / 2017 | `GF(256)` / basic | No / basic | No / basic | **No** | **No** | Unmaintained 9–10 years, zero wheels, incomplete toy packages. |
| **`creedsolo`** | N/A | HTTP 404 | N/A | — | — | — | — | — | Not on PyPI (`reedsolo` ships optional Cython source `creedsolo.pyx` inside its sdist). |

### 4.2 Micro-Benchmarks for Viable Subset (`pyfinite`, `reedsolo`, `komm`, `unireedsolomon`)

Measured in isolated `/tmp/venv-B6-eco` (`66 MB` combined) via [`bench_eco.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/bench_eco.py) (`eco_bench.json`):

| Candidate (Import Time) | Kernel | `q=16` | `q=64` | `q=256` | `q=1024` | `q=4096` | Ratio vs Sage Full (`q=256`) |
|---|---|---:|---:|---:|---:|---:|---|
| **`pyfinite`** (`48.3 ms`) | `gf_mul_1k` (ms) | `0.120` | `0.119` | `0.114` | **`4.424`** | **`6.031`** | `0.30×` (3.3× slower; **66× slower** at $q=4096$) |
| | `gf_inv_1k` (ms) | `0.127` | `0.124` | `0.137` | **`23.975`** | **`31.173`** | `0.13×` (7.8× slower; **420× slower** at $q=4096$) |
| | `syndrome_2x4` (ms) | `0.0024` | `0.0019` | `0.0018` | `0.0178` | `0.0208` | `2.2×` vs Sage prevec (pure Python LUT for $q\le 256$) |
| **`reedsolo`** (`23.0 ms`) | `gf_mul_1k` (ms) | `0.105` | `0.103` | `0.108` | `0.140` | `0.130` | `0.32×` (3.1× slower than Givaro) |
| | `poly_mul` ($\deg q/4$) | `0.003` | `0.023` | `0.333` | `15.721` | `230.591` | `1.03×` at $q=256$; **21.7× slower** at $q=4096$ |
| | `cyclic_rs_dec` ($n=\min(q-1,255)$) | `0.102` | `0.797` | `22.866` | `31.690` | `31.872` | `9.7×` faster than Sage Gao (but **cyclic-only**, incompatible!) |
| **`komm`** (`345.9 ms`) | `gf_mul_1k` (ms) | `1.756` | `2.017` | `2.437` | `4.558` | `4.780` | `0.014×` (**71× slower**; up to **126×** under load) |
| | `gf_inv_1k` (ms) | `8.031` | `11.137` | `27.360` | `30.600` | `18.614` | `0.0006×` (**1,555× slower**) |
| **`unireedsolomon`** (`13.9 ms`) | `gf_mul_1k` (ms) | `0.208` | `0.208` | `0.223` | `0.231` | N/A | `0.15×` (6.5× slower) |
| | `gf_inv_1k` (ms) | `0.132` | **CRASH**¹ | `0.126` | **CRASH**¹ | N/A | ¹ `IndexError: array index out of range` (`ff.py:228`) |

**Verdict for all §4 libraries:** **Reject unconditionally.** None supports odd primes/prime powers (`GF(13)`, `GF(81)`, `GF(1369)`), none supports arbitrary-evaluation-point GRS decoding (including $0 \in \mathbb{F}_q$ and cosets), and none provides finite-field RREF/kernel/subspaces.

---

## 5. Candidate 5 — Hybrid Strategy & `ff_backend` Plugin Architecture

### 5.1 Architectural Design

Instead of a risky big-bang rewrite or remaining locked to a 2.1 GiB GPL stack, `tamo-barg-lrc` can adopt B2's `ff_backend` `Protocol` as a pluggable backend boundary:

1. **Discovery via `importlib.metadata.entry_points`:**
   ```toml
   [project.entry-points."tamo_barg_lrc.backends"]
   numpy = "tamo_barg_lrc.backends.numpy_lut:NumpyLUTBackend"   # default MIT, zero-Sage
   sage  = "tamo_barg_lrc.backends.sage:SageBackend"            # optional [sage] extra
   # Future/optional plugins: galois (B3), flint (B4), cython/numba (B5), rust
   ```
2. **Two-tier Hybrid Deployment Options:**
   - **Option H1 (Transition / Reference Hybrid):** Keep slimmed `passagemath-modules[m4rie]` + `passagemath-brial` (`[sage]` extra, `1.09 GiB`) strictly as (i) CI differential test oracle guaranteeing bit-exact mathematical parity, and (ii) optional fallback for exotic/large fields ($q > 2^{16}$) or users needing Sage `AbstractLinearCode` objects. Route all decode hot loops (syndrome gating, local/global RS error & error-erasure decoding, `intersect_encoding_bases`) to the fast native/LUT backend.
   - **Option H2 (Pure-MIT Runtime + `[sage]` Dev/Validation Extra):** Implement the full `ff_backend` `Protocol` in MIT-licensed Python/NumPy (+ optional B3 `galois` / B4 `python-flint` / B5 Numba-Cython accelerator), so default `pip install tamo-barg-lrc` is **100% MIT, ~65–120 MB, cross-platform (Linux/macOS/Windows, Python 3.10–3.14), with `<100 ms` import**, while `pip install tamo-barg-lrc[sage]` enables differential CI testing against SageMath.

### 5.2 Working Prototype & Bit-Exact Verification (4,540 Random Cases, 0 Mismatches)

We implemented [`hybrid_prototype.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/hybrid_prototype.py) (~520 LOC pure Python + NumPy, zero external coding deps) supporting **arbitrary $\mathbb{F}_{p^m}$** (binary, odd prime, and odd prime powers with Sage-identical `.to_integer()` encoding):
- **(a) Syndrome computation (`FastFiniteField.matvec`):** Vectorized `log`/`exp` LUT + `np.bitwise_xor.reduce` for $\mathbb{F}_{2^m}$, exact `int64` BLAS matmul mod $p$ for $\mathbb{F}_p$, and digit-wise vectorized add for odd $\mathbb{F}_{p^m}$.
- **(b) `intersect_encoding_bases_fast`:** Zassenhaus stacked-matrix `left_kernel_matrix_gf` + reverse-column `rref_gf` producing bit-exact canonical degree-minimal polynomials.
- **(c) `FastRSCodec` (`decode_errors` + `decode_errors_erasures`):** Syndrome-based Sugiyama (partial XGCD) + Chien + Forney on **arbitrary distinct evaluation points** (including $\alpha_j = 0$, handled exactly via $S_0 = e_0 \eta_0 + \sum_{j \ne 0} e_j \eta_j$ and shifting $S_1 \dots S_{d-2}$ when $\alpha_0 = 0$ is erased!). Completely eliminates $O(n^2)$ Lagrange interpolation, $O(n^2)$ vanishing polynomials, and per-call punctured GRS code construction.

**Differential Verification Results** ([`verify_and_bench_hybrid.py`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/verify_and_bench_hybrid.py) $\to$ [`hybrid_verify_bench.json`](file:///home/antigravity/amirmulla/optimal_lrc/reports/perf_analysis/scratch/B6/hybrid_verify_bench.json), seeded `42`):

| Primitive Verified vs SageMath `.venv-lrc` | Fields Tested | Random Cases | Edge Cases Covered | Mismatches |
|---|---|---:|---|---:|
| **(a) Syndrome (`H @ v`)** | `GF(16), GF(64), GF(256), GF(13), GF(81)` | **1,200** | Arbitrary rectangular $(1..12) \times (2..32)$ | **0** |
| **(b) `intersect_encoding_bases`** | `GF(16), GF(64), GF(256), GF(13), GF(81)` | **1,100** | Empty, partial, full overlap; exact canonical degree order | **0** |
| **(c1) RS Error Decoder** (vs `GRSGaoDecoder`) | `GF(16), GF(64), GF(256), GF(13), GF(81)` | **1,120** | 0 err, $1..t$ err (incl. $\alpha=0$), **224 beyond-radius** (100% same misdecode vs fail), **2 Sage `ValueError` leaks cleanly converted to `DecodingError`**, $n=k$ | **0** |
| **(c2) RS Error-Erasure** (vs `GRSErrorErasureDecoder`) | `GF(16), GF(64), GF(256), GF(13), GF(81)` | **1,120** | $e=0$, $e=d-1$, $e \ge d$, $2\tau+e \le d-1$, beyond radius, erasing $\alpha=0$, arbitrary garbage at erased positions, $n=k$ | **0** |
| **Total Verified** | All 3 field families ($\mathbb{F}_{2^m}, \mathbb{F}_p, \mathbb{F}_{p^m}$) | **4,540** | — | **0** |

### 5.3 Hybrid Prototype Kernel Benchmarks vs SageMath

Even as uncompiled pure Python + NumPy (before any Numba/Cython/FLINT acceleration from B3–B5):

| Kernel | `q=16` Hybrid / Speedup | `q=64` Hybrid / Speedup | `q=256` Hybrid / Speedup | `q=1024` Hybrid / Speedup | `q=4096` Hybrid / Speedup |
|---|---:|---:|---:|---:|---:|
| `gf_mul_1k` (vec NumPy) | `0.022 ms` (`1.5×`) | `0.023 ms` (`1.7×`) | `0.021 ms` (**`1.6×`**) | `0.023 ms` (**`4.3×`**) | `0.023 ms` (**`4.0×`**) |
| `syndrome_2x4` (vs `vector()`) | `0.012 ms` (`4.6×`) | `0.012 ms` (`4.5×`) | `0.012 ms` (**`4.7×`**) | `0.012 ms` (**`5.0×`**) | `0.012 ms` (**`5.1×`**) |
| `syndrome_(n-k)xn` ($n\le 256$) | `0.016 ms` (`0.9×`) | `0.078 ms` (`1.4×`) | `2.06 ms` (`0.94×`) | `1.04 ms` (**`42.8×`**) | `1.05 ms` (**`31.1×`**) |
| **`rs_decode` ($n=\min(q,256), k=n/4$)** | `0.067 ms` (**`19.1×`**) | `0.222 ms` (**`57.7×`**) | **`2.233 ms` (`99.8×`!)** | **`4.353 ms` (`93.8×`!)** | **`4.419 ms` (`118.3×`!)** |
| **`rs_decode` local `[4,2]` (vs BW)** | `0.032 ms` (**`22.6×`**) | `0.033 ms` (**`23.1×`**) | **`0.033 ms` (`24.7×`!)** | `0.032 ms` (**`27.8×`**) | `0.033 ms` (**`26.4×`**) |
| **`rs_decode` local `[8,4]` (vs BW)** | `0.045 ms` (**`19.9×`**) | `0.048 ms` (**`22.3×`**) | **`0.047 ms` (`24.5×`!)** | `0.047 ms` (**`26.5×`**) | `0.045 ms` (**`30.6×`**) |
| `rref` ($k \times n$, pure Python loop) | `0.106 ms` (`0.13×`) | `2.11 ms` (`0.02×`) | `70.8 ms` (`0.007×`) | `59.1 ms` (`0.12×`) | `63.0 ms` (`0.31×`) |

*(Speedups relative to Sage `.venv-lrc` baseline. Note the single weakness of pure-Python/NumPy in this prototype: dense RREF/kernel over `64×256` takes `70.8 ms` / `221 ms` because of the Python pivot loop — which is precisely why pairing `ff_backend` with a compiled/JIT RREF kernel from B3 `galois`, B4 `python-flint`, B5 Numba/Cython, or keeping M4RIE in hybrid mode achieves best-of-both-worlds speed across **every** layer!)*

### 5.4 Gap List & Effort Estimate (LOC) to Complete Pure-MIT `ff_backend`

| Missing Component (Beyond `hybrid_prototype.py`) | Effort (LOC) | Notes |
|---|---:|---|
| Conway polynomial table ($q \le 2^{16}$, odd $p^m$ up to $1369+$) + primitive-element finder | ~120 | Or reuse MIT `conway-polynomials` wheel (`0.10`, 0.2 MB, MIT) |
| `enumerate_subspaces(p, m, d)` bitmask RREF generator (replaces `VectorSpace.subspaces`) | ~45 | Proven **365×–938× faster** than Sage in A1 |
| `FFPoly` wrapper + balanced product tree `vanishing_poly` + direct Newton `interpolate` | ~180 | Replaces `PolynomialRing`, `prod`, `lagrange_polynomial` |
| Native `LinearCode` / `Encoder` / `Decoder` lightweight base classes + `divisors` | ~150 | Replaces `AbstractLinearCode`, `Encoder`, `Decoder`, `cartesian_product` |
| `SageBackend` adapter implementing `FFBackend` protocol (for `[sage]` extra & CI differential tests) | ~220 | Thin wrapper around Sage `GF`, `matrix`, `PolynomialRing`, `GeneralizedReedSolomonCode` |
| Fast compiled/JIT RREF & `left_kernel` kernel (delegate to B3/B4/B5 or 80-LOC Cython/Numba) | ~80–150 | Closes the one `rref`/`left_kernel` gap vs M4RIE |
| **Total migration glue + backend implementation** | **~800–900 LOC** | Incremental migration: hot decoders can be swapped first! |

---

## 6. Candidate 6 — Other Languages (Rust / C++ Effort–Benefit Estimate Only)

No prototype built per B6 instructions; effort and risk estimated against B2's `ff_backend` protocol:

| Dimension | **Rust (`PyO3` + `maturin`)** | **C++ (`nanobind` / `pybind11` + FLINT/NTL or custom AVX2)** |
|---|---|---|
| **Existing crates / libs** | `reed-solomon-erasure` & `reed-solomon-simd` are **erasure-only over `GF(2^8)` / `GF(2^16)`** (useless for general error+erasure GRS over $\mathbb{F}_{p^m}$!). Must write custom GF/poly/RREF/Sugiyama-Forney crate. | **FLINT** (`fq_nmod`, `fq_nmod_poly`, `fq_nmod_mat`) already covers L1–L3 at C speed (covered directly by B4 `python-flint` without maintaining custom C++!); NTL/Givaro add heavy build deps. Custom header-only C++20 LUT/PCLMUL covers L1–L4. |
| **Expected speedup vs Sage** | **100×–2,000×** (SIMD `PSHUFB`/`PCLMULQDQ` for `GF(2^m)` GEMM/RREF/syndrome, `< 50 µs` length-256 RS decode, GIL-free `rayon` Monte Carlo simulations). | **100×–2,000×** (comparable to Rust; AVX2/PCLMUL or direct FLINT C calls). |
| **Engineering effort** | **~2,000–3,000 LOC** Rust + PyO3 bindings + `maturin` CI wheel matrix (Linux `manylinux`/`musllinux`, macOS universal2, Windows `win_amd64`). | **~2,000–3,000 LOC** C++ + `nanobind` + `scikit-build-core`/CMake + `cibuildwheel` CI (linking GMP/FLINT on Windows is notoriously painful). |
| **Maintenance & onboarding** | High contributor friction for a Python/coding-theory research repo; two toolchains (`cargo` + `uv`); ABI3 wheels simplify Python minor-version upgrades. | Highest build/ABI friction (compiler differences MSVC/Clang/GCC, CMake, native dependency vendoring). |
| **License** | **MIT / Apache-2.0** (100% compatible). | Custom C++: **MIT**; bundling FLINT/GMP: **LGPL-3.0-or-later** (compatible dynamically/weakly, requires license notice). |
| **When justified?** | Only if simulations scale to **$q \ge 4096$–$65536$** with millions of Monte Carlo trials/sec where B3/B4/B5 Python+C/Numba hits Python loop ceilings. | Rarely justified over **B4 `python-flint`** (which already exposes FLINT's C library with prebuilt wheels) or **Rust/PyO3** (far cleaner packaging via `maturin`). |
| **Verdict** | **Defer / Reject for v0.3** (overkill while B3/B4/B5 + Hybrid deliver 25×–500× at 1/4 the effort). | **Reject** (strictly dominated by `python-flint` for C algebra or Rust/PyO3 for custom native code). |

---

## 7. Non-Functional Comparison (B6 Candidates)

| Candidate | License | MIT-Compat? | Venv Footprint (`du -sh`) | Wheel OS Coverage (Lin / Mac / Win) | Python Versions | Compiled vs Pure Py | Maintenance Health | NumPy / Numba Coupling |
|---|---|---|---|---|---|---|---|---|
| **Slimmed `passagemath` (`[m4rie]+brial`)** | GPL-2.0-or-later (+ GPLv3 `memory_allocator`) | **No** | `1,089 MB` | Lin / Mac / **No Win** | `3.11–3.14` (breaks `3.10`) | Compiled C/C++/Cython | Active (`10.8.14`, Nov 2026) | Light (`numpy` optional) |
| **Minimal `passagemath` (`[gf]`)** | GPL-2.0-or-later | **No** | `1,025 MB` | Lin / Mac / **No Win** | `3.11–3.14` | Compiled C/C++/Cython | Active (`10.8.14`) | None |
| **Full SageMath (conda/system)** | GPL-3.0-or-later | **No** | `3.5–5.5 GB` | Lin / Mac / **WSL2 only** | Bundled | Compiled | Active | Bundled |
| **SymPy (`1.14.0`)** | BSD | **Yes** | `~40 MB` (`106 MB` w/ NumPy) | All (`py3-none-any`) | `>=3.9` | Pure Python | Very healthy (Apr 2025) | None |
| **`pyfinite` (`1.9.1`)** | MIT | **Yes** | `<2 MB` | sdist only | Unspecified | Pure Python (+ `msgpack`) | Stale (Nov 2022) | None |
| **`reedsolo` (`1.7.0`)** | Public Domain / MIT-0 | **Yes** | `<1 MB` | All (`py3-none-any`) | Unspecified | Pure Python | Quiet (Jan 2023) | None |
| **`komm` (`0.37.0`)** | **GPL-3.0-only** | **No** | `~64 MB` (w/ NumPy) | All (`py3-none-any`) | `>=3.11` | Pure Python + NumPy | Active (Oct 2026) | `numpy>=2.1.1`, `tqdm` |
| **`unireedsolomon` (`1.0.6`)** | MIT | **Yes** | `<1 MB` | `win_amd64 cp310` only; sdist else | `>=3.7` | Pure Python | Quiet (Apr 2023) | None |
| **`pyldpc` (`0.7.9`)** | MIT | **Yes** | N/A (build fails) | sdist only (broken PEP 517) | Unspecified | Pure Python | Abandoned (Feb 2020) | Broken `setup.py` `numpy` |
| **Hybrid (`ff_backend` + MIT LUT/NumPy + opt `[sage]`)** | **MIT** (core); GPL opt-in `[sage]` | **Yes** (core) | `~65 MB` core (`+1.09 GB` if `[sage]`) | **All** (core); Lin/Mac for `[sage]` | **`>=3.10`** (core) | Pure Python + NumPy (pluggable) | First-party | `numpy` only (no Numba required) |
| **Rust (`PyO3` + `maturin`)** | MIT / Apache-2.0 | **Yes** | `~2–5 MB` wheel | All (via `maturin` CI) | `>=3.10` (`abi3`) | Compiled Rust | First-party (dual-lang) | Optional (`numpy` buffer) |
| **C++ (`nanobind` + custom/FLINT)** | MIT (or LGPL-3.0) | **Yes** | `~5–25 MB` wheel | Lin / Mac / Hard Win | `>=3.10` | Compiled C++ | First-party (high friction) | Optional |

---

## 8. Weighted Score Matrix & Master Decision Table (All Candidates)

Scoring scale: **0–5** (higher is always better; `Correctness Risk` and `Migration Effort` are inverted so `5 = lowest risk / lowest effort`).  
Weights per `alternatives_template.md`: **Performance 25% · Coverage 20% · Correctness 20% · Install 10% · Maintainability 10% · License 5% · Effort 10%**.

| Candidate (Source Report) | Perf (25) | Cover (20) | Correct (20) | Install (10) | Maint (10) | Lic (5) | Effort (10) | **Weighted Total (/5.00)** | Footprint | Headline Speedup vs Sage (`GF(256)` Global RS / Init) | **Verdict** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **Hybrid `ff_backend` + Fast Decode + opt `[sage]`** (B6) | 4.5 | 5.0 | 5.0 | 4.0 | 4.5 | 5.0 | 3.0 | **4.53** | `~65 MB` core (`+1.09 GB` `[sage]`) | **`100×` RS (`2.23 ms`)** / pairs w/ B3–B5 for RREF | **Adopt architecture** (plugin boundary + `[sage]` CI oracle; pair w/ B3/B4/B5 fastest RREF/LA engine) |
| **Rust (`PyO3` + `maturin`)** (B6, est.) | 5.0 | 4.5 | 3.5 | 3.5 | 3.0 | 5.0 | 1.5 | **3.90** | `~5 MB` wheel | `~100–2,000×` (est.) | **Defer** (revisit only if $q \ge 4096$ simulations saturate Python/Numba/FLINT) |
| **Slimmed `passagemath` (`[m4rie]+brial`)** (B6) | 2.5 | 5.0 | 5.0 | 2.0 | 3.5 | 1.0 | 5.0 | **3.73** | `1,089 MB` (`-48%`) | **`7.8–15.8×` Global RS** / **`5.9×` Init** (0 LOC!) | **Adopt immediately as short-term stopgap & optional `[sage]` extra** |
| **C++ (`nanobind` + custom/FLINT)** (B6, est.) | 5.0 | 4.5 | 3.0 | 2.5 | 2.5 | 4.5 | 1.5 | **3.63** | `~10–25 MB` | `~100–2,000×` (est.) | **Reject** (dominated by `python-flint` or Rust/PyO3) |
| **Bare Minimal `passagemath` (`[gf]`)** (B6) | 1.5 | 5.0 | 5.0 | 2.0 | 3.5 | 1.0 | 5.0 | **3.48** | `1,025 MB` (`-51%`) | `15.8×` Global RS / `2.1×` Init (**144× RREF regression!**) | **Reject** (strictly dominated by `[m4rie]+brial` for `+64 MB`) |
| **Full SageMath (conda/system)** (B6) | 1.5 | 5.0 | 5.0 | 0.5 | 4.0 | 1.0 | 5.0 | **3.38** | `3.5–5.5 GB` | `1.0×` (baseline) | **Reject** as primary dep (keep passive compatibility only) |
| **`reedsolo` `1.7.0`** (B6) | 2.5 | 1.0 | 2.0 | 5.0 | 2.5 | 5.0 | 1.0 | **2.33** | `<1 MB` | `9.7×` cyclic RS (incompatible!) | **Reject** (binary cyclic-only, global non-thread-safe tables) |
| **SymPy `1.14.0`** (B6) | 0.5 | 1.5 | 2.0 | 4.5 | 5.0 | 5.0 | 1.0 | **2.13** | `~40 MB` | **189×–3,277× slower** on `GF(2^m)` | **Reject** (no native `GF(p^m)` domain; `GF(16)` trap; far too slow) |
| **`komm` `0.37.0`** (B6) | 0.5 | 1.0 | 3.5 | 4.5 | 4.5 | 0.5 | 1.0 | **2.05** | `~64 MB` | **71×–1,555× slower** | **Reject** (GPL-3.0-only, binary-only, bit-level cyclic RS) |
| **`pyfinite` `1.9.1`** (B6) | 1.5 | 1.0 | 2.5 | 2.5 | 1.5 | 5.0 | 1.0 | **1.83** | `<2 MB` | `40×–225×` cliff $>2^8$ | **Reject** (binary-only, erasure-only, pollutes CWD with `ffield.lut.*`) |
| **`unireedsolomon` `1.0.6`** (B6) | 2.0 | 1.0 | 1.0 | 3.0 | 2.0 | 5.0 | 1.0 | **1.75** | `<1 MB` | Crashes on `GF(64)`/`GF(1024)` | **Reject** (cyclic-only + verified `IndexError` bug in `ff.py:228`) |
| **`pyldpc` `0.7.9`** (B6) | 0.0 | 0.0 | 1.0 | 0.5 | 0.5 | 5.0 | 0.0 | **0.55** | N/A | N/A | **Reject** (abandoned, broken PEP 517 build, LDPC-only) |
| **`galois`** (`B3_galois.md`) | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *5.0 (MIT)* | *TBD (C1)* | ***TBD (C1)*** | *TBD (C1)* | *TBD (C1)* | *Placeholder — `B3_galois.md` not yet generated at B6 completion* |
| **`python-flint`** (`B4_flint.md`) | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *4.5 (MIT+LGPL)* | *TBD (C1)* | ***TBD (C1)*** | *TBD (C1)* | *TBD (C1)* | *Placeholder — `B4_flint.md` not yet generated at B6 completion* |
| **Custom NumPy / Numba / Cython** (`B5_custom.md`) | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *TBD (C1)* | *5.0 (MIT)* | *TBD (C1)* | ***TBD (C1)*** | *TBD (C1)* | *TBD (C1)* | *Placeholder — `B5_custom.md` not yet generated at B6 completion* |

---

## 9. Concrete Recommendations for C1 Synthesis

1. **Immediate Zero-Risk Win (Today, 0 LOC):** Update installation docs and add `[project.optional-dependencies] sage = ["passagemath-modules[m4rie]>=10.8,<11", "passagemath-brial>=10.8,<11"]`. This immediately cuts venv disk usage by **~1.0 GiB (-48%)** while speeding up `GF(256)` two-set init by **5.9×**, iterative decoding by **1.9×**, and global Gao decoding by **7.8×–15.8×** (all 34 tests pass unchanged).
2. **Target Architecture (v0.3+):** Adopt the **`ff_backend` plugin architecture** (`importlib.metadata.entry_points`) using the best pure-MIT / permissive backend from **B3–B5** (combining fast compiled/JIT RREF with our verified `FastRSCodec` Sugiyama+Forney decoder for **~100×+** decoding speedups, `<100 MB` install, Windows + Python 3.10–3.14 wheels, and clean MIT licensing), while retaining slimmed `passagemath` strictly as an optional `[sage]` extra for CI differential testing.
3. **Do NOT invest further time in SymPy, `pyfinite`, `reedsolo`, `komm`, `unireedsolomon`, `pyldpc`, `zfec`, `bchlib`, or custom C++:** All are decisively disqualified on field support, arbitrary-evaluation-point RS semantics, license compatibility, bugs, or 2–3 orders-of-magnitude slowdowns.
