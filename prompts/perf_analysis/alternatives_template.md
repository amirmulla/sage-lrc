# Common template for alternative-evaluation agents (09–12)

Inputs: `reports/perf_analysis/A*.md`, `B1_profiling.md`, `B2_sage_audit.md` (read them first; use B2's `ff_backend` protocol as the target).

For the candidate(s) assigned, deliver in your report:
1. **Fit**: field support (GF(2^m), GF(p^m), sizes up to 2^12+), polynomial ring, matrices (kernel/RREF/rank/intersection), subspace enumeration, RS decoding (Gao/BW/ErrorErasure), vectorization/batching.
2. **Kernel micro-benchmarks** (shared methodology; same parameters as Sage baseline, q ∈ {16, 64, 256, 1024, 4096}): GF mul/inv/pow (scalar & array), polynomial multiply/gcd/interpolate/evaluate, RREF/kernel for k×n matrices, matrix-vector syndrome, subspace enumeration, a complete length-n RS Gao decode, and a length-r Berlekamp–Welch decode. Report speed-up ratio vs Sage (median, p10/p90).
3. **Prototype**: implement in `scratch/<ID>/` a replacement for at least: (a) syndrome computation, (b) `intersect_encoding_bases`, (c) one RS decoder, and verify bit-exact equality vs Sage outputs on ≥ 1000 random cases (seeded).
4. **Gap list**: features the candidate lacks and the effort to fill (LOC estimate).
5. **Non-functional**: license (must be MIT-compatible), install footprint (`du -sh` of venv), wheel availability for Linux/macOS/Windows, pure-Python vs compiled, Python versions, maintenance health (last release, issue activity), numpy/numba coupling.
6. **Score matrix** (0–5): performance, feature coverage, correctness risk (inverted), install simplicity, maintainability, license, migration effort (inverted). Provide weighted total (weights: perf 25, coverage 20, correctness 20, install 10, maintain 10, license 5, effort 10).
7. **Verdict**: adopt fully / adopt partially (list which layers) / reject, with reasons.

Rules: separate venv per agent; never touch `.venv-lrc`; if install fails record it and mark results "unverified".
