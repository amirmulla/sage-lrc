# 11 — B5: Custom GF(2^m)/GF(p) kernel (tables + NumPy / Numba / Cython)

## Role
Designer/prototyper of a purpose-built, dependency-light finite-field backend. Read `shared_context.md` and `alternatives_template.md`.

## Exclusive output
`reports/perf_analysis/B5_custom_backend.md`, `reports/perf_analysis/scratch/B5/` (the prototype lives here)

## Specific focus
- Design: log/antilog tables for q ≤ 2^16, polynomial-basis integer encoding with carry-less multiply, optional bit-sliced / packed representation for GF(2^m); GF(p) via modular int arrays; `GF(q^m)`-style extension handled via tables of the composite field only.
- NumPy vectorized kernels: mul/div/pow, dot/matmul over GF(2^m) (log-table or XOR-accumulate), Gaussian elimination (batched row ops), kernel/rank/intersection, Vandermonde, syndrome.
- Polynomial layer: schoolbook vs Karatsuba vs Kronecker/NTT-style; Euclid/Gao decoder in vectorized form; Berlekamp–Massey alternative; Chien/Forney for evaluation-based decoding with arbitrary evaluation sets (non-cyclic!).
- Accelerators compared on the same kernels: pure NumPy vs **Numba** (`njit`, `parallel=True`, cache) vs **Cython** vs **C via cffi** — speed-up, compile/AOT story, packaging burden (wheels), first-call latency.
- Batch decoding of many words simultaneously (SIMD-style over the word axis) — projected throughput.
- Exact replacement coverage vs the `ff_backend` protocol from B2 (table: protocol method → implemented? LOC?).
- Estimate total implementation LOC, test strategy (differential testing vs Sage on random inputs), and correctness-risk mitigation.
