# 03 — A3: tamo_barg.py (single-set code) analysis

## Role
Coding-theory algorithm analyst for the single-partition Tamo–Barg LRC. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A3_tamo_barg.md`, `reports/perf_analysis/scratch/A3/`

## Tasks
1. Mandatory per-function row for every method of `TamoBargCode`, `TamoBargVectorEncoder`, `TamoBargErasureDecoder`, `TamoBargErrorDecoder`.
2. Specific questions:
   - `TamoBargCode.__init__`: what is precomputed vs lazily cached (`cached_method`)? Cost split.
   - `minimum_distance`: closed form vs search; validity conditions.
   - `TamoBargVectorEncoder.enc_poly` / `generator_matrix`: polynomial evaluation cost (Horner vs per-point), matrix build O(k·n·deg); can generator matrix be built by vectorized evaluation / Vandermonde-like structure?
   - Encoding per message: O(k·n) matrix-vector vs FFT-style additive FFT on subgroup structure.
   - `TamoBargErasureDecoder.decode_to_code`: Lagrange per coset, complexity O(#erased·r²); batching across cosets.
   - `TamoBargErrorDecoder`: local GRS decoder construction cost, BerlekampWelch cost per coset O(r³) vs O(r²) variants (Gao/Euclid, Peterson), parity-check precomputation; syndrome gating effectiveness.
   - Memory of precomputed per-coset decoders (n/r objects) and whether they can share one decoder when cosets are shifts of the same subgroup (affine-equivalent local codes).
3. Benchmark per-word latency and construction across q ∈ {16, 64, 256, 1024} and several (r, k).
4. Quantify Sage-framework overhead (Decoder/Encoder registration, input-space validation, vector creation) vs actual math using `cProfile`.

## Acceptance checklist
- [ ] Every method has a row
- [ ] Framework-overhead vs math-time split (%)
- [ ] Shared-local-decoder opportunity evaluated with numbers
- [ ] No out-of-scope changes
