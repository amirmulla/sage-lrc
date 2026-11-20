# 13 — C1: Synthesizer

## Role
Lead analyst. Merge all reports into the final decision document. Read `shared_context.md`, then all of `reports/perf_analysis/*.md`. Read-only on source; verify claims by re-running a sample (≥ 3) of the reported benchmarks.

## Exclusive output
`reports/perf_analysis/FINAL_REPORT.md`

## Required sections
1. **Executive summary** (≤ 15 lines): headline findings, recommendation.
2. **Master per-function table**: every function from the inventory (merge A1–A6 rows into: algorithm · complexity · measured · bottleneck · improvement · Sage-replacement difficulty · priority).
3. **Performance picture**: category attribution (B1), scaling exponents vs theory, top-10 hotspots.
4. **Optimization roadmap**: ranked by (speed-up × frequency ÷ effort); split into *quick wins* (no Sage change), *algorithmic* (e.g. batch decoding, shared local decoders, skipping global Gao, incremental min-distance, sparse eigensolvers, parallel simulation) and *backend-dependent*. Include expected gains and which tests guard each.
5. **Sage-alternatives decision matrix**: all candidates (B3–B6) with weighted scores, benchmark headline ratios, licenses, footprint, gaps.
6. **Recommendation**: one primary path with justification, plus fallback. Include the target architecture (`ff_backend` protocol, Sage reference backend, fast backend), phased migration plan (phase, scope, deliverable, tests, rollback), effort in person-days, risks and mitigations (correctness, numerical/field edge cases, packaging, CI matrix), and success metrics (e.g. ≥ N× speed-up, install size, zero test regressions).
7. **Conflicts & unverified claims** list (disagreements between agents, items marked unverified).
8. **Appendix**: index of reports and scratch artifacts, reproduction commands.

## Acceptance checklist
- [ ] All inventory functions appear in the master table
- [ ] ≥ 3 benchmarks independently re-verified
- [ ] Clear single recommendation with phased plan
- [ ] `git diff --stat src simulation tests` empty
