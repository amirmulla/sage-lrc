# 05 — A5: two-set decoders analysis

## Role
Decoder algorithm & convergence analyst. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A5_two_sets_decoders.md`, `reports/perf_analysis/scratch/A5/`

## Tasks
1. Mandatory per-function row for `_partition_is_syndrome_clean` and `decode_to_code` of all six decoders: Iterative, IterativeEE, Global, GlobalEE, TwoStep, TwoStepEE (12 rows) plus a comparison table.
2. Specific questions:
   - Local decoding per coset: Berlekamp–Welch vs Gao vs ErrorErasure — algorithm, complexity in r and #errors, constants in Sage.
   - Iterative: iteration count distribution vs error weight; convergence/stall criteria; worst-case; expander-based guarantee vs observed.
   - Global/TwoStep: length-n Gao decode `O(n log² n)` theory vs Sage actual (polynomial arithmetic backend: FLINT/NTL/pure?). Measure at n = 16…4096. Gain from fast-multiplication/half-gcd vs quadratic Euclid.
   - TwoStep: fraction of words where first stage already finishes; cost saved by syndrome gating; opportunities to skip the global decode.
   - EE variants: erasure handling cost (`interpolate_erasures`, locator polynomials).
   - Syndrome check cost (matrix-vector over GF(q)): batched numpy with log tables vs Sage `vector * matrix`.
   - Batch decoding: how much of per-word cost is fixed Python/Sage overhead (profile) — estimate achievable throughput with vectorized/batched decoding.
3. Correctness: decoding radius achieved vs theoretical (empirical success-rate curve for error weights 0..t+2 on GF(64) and GF(256), ≥ 200 words each, seeded).
4. Cost model: ms/word = a + b·(#iterations) + c·(global decode); fit constants.

## Acceptance checklist
- [ ] 12 complete rows + comparison table
- [ ] Success-rate-vs-error-weight table
- [ ] Cost-model fit
- [ ] Ranked list of decoder speed-ups with estimated factors
- [ ] No out-of-scope changes
