# 12 — B6: Slimmed passagemath + other candidates

## Role
Evaluator of (a) staying on Sage in the lightest possible form and (b) the remaining ecosystem candidates. Read `shared_context.md` and `alternatives_template.md`.

## Exclusive output
`reports/perf_analysis/B6_passagemath_and_others.md`, `reports/perf_analysis/scratch/B6/`

## Candidates
1. **Modular passagemath**: minimal wheel set needed to run the whole project (iteratively drop wheels: pari, ntl, gap, groups, combinat, repl, linbox, flint...). Report minimal set, `du -sh`, import time, install time, wheel availability across OS/Python versions, whether `sage.coding` (`GeneralizedReedSolomonCode`, channels) is shipped in a wheel (check `passagemath-*` names on PyPI; record exact findings), and how to express it as optional extra in `pyproject.toml`.
2. **Full SageMath** via conda-forge/system: pros/cons, CI cost, user friction.
3. **SymPy** (`GF`, `galoistools`, `Poly(modulus=)`): GF(p) only? GF(2^m) support? speed.
4. **pyfinite**, **galois-free pure-Python** `ffield`, **`reedsolo`** (RS over GF(2^8) only?), **`komm`**, **`unireedsolomon`**, **`pyldpc`/`sage-free coding libs`**: capability triage table; reject quickly with evidence when unsuitable; benchmark only the viable ones.
5. **Hybrid strategies**: keep Sage for rare/offline paths (subspace enumeration, construction) and use a fast backend for decode hot loops; optional-backend plugin architecture (`ff_backend` entry point) with Sage as the reference implementation and differential tests.
6. **Other languages**: Rust (PyO3 + `reed-solomon-erasure`/custom), C++ (pybind11, NTL, FLINT) — effort/benefit estimate only, no prototype required.

Output a decision table across ALL candidates (including B3–B5 headline numbers, referenced by file) once those reports exist; otherwise leave placeholders and let C1 fill them.
