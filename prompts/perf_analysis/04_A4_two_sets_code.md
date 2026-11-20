# 04 — A4: two-set code, encoder, helpers analysis

## Role
Analyst for `TamoBargCodeTwoSets`, its encoder and helpers. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A4_two_sets_code_encoder.md`, `reports/perf_analysis/scratch/A4/`

## Tasks
1. Mandatory per-function row for `_is_int_like`, `_int_log_base`, `_safe_parity_check_matrix`, `TamoBargCodeTwoSets.__init__/design_distance/minimum_distance_py`, `TamoBargTwoSetsVectorEncoder.*`.
2. Specific questions:
   - `__init__` (~350 ms on GF(256)): profile phase by phase (subgroups, bases, `intersect_encoding_bases`, generator matrix). Which phase dominates and why? What can be lazy?
   - `design_distance` vs true `minimum_distance_py`: formula, assumptions, tightness.
   - `minimum_distance_py`: projective enumeration complexity O(q^k/(q-1)·n); early termination, Gray-code/incremental update, numpy vectorization, bit-packing for GF(2^m); max feasible (q,k).
   - `_safe_parity_check_matrix`: kernel computation backend (Linbox?) cost, fallbacks, cache.
   - Encoder: message→polynomial→codeword path; batching many messages as a single matrix product.
   - Generator-matrix construction: O(k·n) evaluations — vectorizable with log tables?
3. Scaling: q ∈ {16, 64, 256, 1024} for init; k ∈ {2,3,4} for min distance.

## Acceptance checklist
- [ ] All rows complete
- [ ] Phase-by-phase init profile table
- [ ] `minimum_distance_py` speed-up proposal with estimated factor
- [ ] No out-of-scope changes
