# 10 — B4: `python-flint` (+ custom decoders)

## Role
Evaluator of FLINT bindings. Read `shared_context.md` and `alternatives_template.md`.

## Exclusive output
`reports/perf_analysis/B4_python_flint.md`, `reports/perf_analysis/scratch/B4/`

## Specific focus
- `flint.fq_default`, `fq_default_poly`, `fq_default_mat`, `nmod_poly`, `nmod_mat`, `fmpz_mod_poly`: GF(p^m) construction, element ops, polynomial gcd/xgcd/half-gcd/interpolation/multipoint evaluation (fast algorithms!), matrix rref/nullspace/rank over GF(q).
- Whether `fq_default_*` supports all needed operations in the installed version (check API; record version gaps).
- Implement Gao decoding with fast half-gcd (`xgcd` / `gcd` internals) and measure vs Sage Gao at n up to 4096; Berlekamp–Welch via `fq_default_mat.solve`/`nullspace`.
- Subspace enumeration: implement via mat ops; compare.
- Python-level overhead per element op (boxed objects) vs vectorized APIs; batch strategy.
- Wheel availability, build requirements (FLINT bundled?), license (LGPL-3 compat. with MIT distribution — analyse), PyPI size.
