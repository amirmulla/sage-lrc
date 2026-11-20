# 01 — A1: subgroups.py + polynomials.py analysis

## Role
Algorithm & performance analyst for field/subgroup/polynomial primitives. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A1_subgroups_polynomials.md`, `reports/perf_analysis/scratch/A1/`

## Scope
All functions listed for A1 in the inventory.

## Tasks
1. For every function fill the mandatory per-function row.
2. Specific questions:
   - `find_subspaces` / `find_additive_subgroups`: how does `VectorSpace.subspaces` enumerate (Gaussian binomial count)? Cost vs q=2^m for m=4..12; when does it blow up? Compare `find_additive_subgroups_bruteforce` (complexity, crossover point).
   - `list_multiplicative_subgroups`: divisor enumeration + generator power; can it be O(#divisors) with table lookups?
   - `field_element_to_int` / `sort_field_elements`: cost of Sage element → int; vectorizable with log/antilog or polynomial-basis integer encoding?
   - `build_annihilator_polynomial`, `build_good_polynomial`: product-of-linear-factors vs subspace polynomial (linearized, via recurrence); asymptotics and coefficient growth.
   - `build_algebra_basis`/`build_encoding_basis`: monomial basis count, degree bounds, verify dimension = k.
   - `intersect_encoding_bases`: Zassenhaus-via-RREF cost O(k²·n); compare alternatives (kernel of stacked matrix, direct subfield method); is it the dominant term in two-set construction (profile!).
   - `interpolate_erasures`: Lagrange O(s²) vs Newton / Vandermonde solve / barycentric; benefit at typical erasure counts.
3. Profile each function for q ∈ {16, 64, 256, 1024, (4096 if feasible)}; fit scaling.
4. For each Sage primitive, name the pure-numpy/`galois`/FLINT equivalent (detail only; B-agents evaluate).

## Acceptance checklist
- [ ] All 18 functions have complete rows
- [ ] Scaling plot data (table) for the subspace-enumeration and `intersect_encoding_bases`
- [ ] Top-3 hotspots ranked with evidence
- [ ] No file outside the output paths changed
