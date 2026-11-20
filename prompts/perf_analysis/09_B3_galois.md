# 09 — B3: `galois` + NumPy

## Role
Evaluator of the `galois` library (https://github.com/mhostetter/galois). Read `shared_context.md` and `alternatives_template.md` (follow it exactly).

## Exclusive output
`reports/perf_analysis/B3_galois.md`, `reports/perf_analysis/scratch/B3/`

## Specific focus
- `galois.GF(2**m)` arrays, JIT (numba) compile time vs steady-state, lookup-table vs calculation modes, `GF.Vandermonde`, `np.linalg` over GF (`row_reduce`, `null_space`, `rank`), `galois.Poly` (interpolation via `lagrange_poly`, gcd, evaluation), `galois.ReedSolomon` (BCH/RS decode — does it support generalized RS with arbitrary evaluation points and column multipliers? errors+erasures?), `galois.FieldArray` subfields/`is_primitive`.
- Large q (2^12) memory for tables; first-call JIT latency impact on test-suite & CLI.
- Subspace enumeration: not provided — implement in numpy and benchmark vs Sage.
- Custom GRS Gao/BW decoder built on `galois.Poly` and its speed vs Sage's.
- numba/numpy version constraints with Python 3.11/3.12/3.13.
