# 08 — B2: Sage dependency surface audit

## Role
Dependency auditor. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/B2_sage_audit.md`, `reports/perf_analysis/scratch/B2/`

## Tasks
1. Exhaustively list every Sage import and every Sage-originated call/method in `src/`, `simulation/`, `tests/` with file:line (use `ast` + grep; script in scratch). Include implicit use (operators on field elements, `.parent()`, `.list()`, `.polynomial()`, `.nth_root`, etc.).
2. Classify each usage into layers:
   - **L1 field arithmetic** (GF(q), q = p^m, element ops, conversions)
   - **L2 linear algebra** (matrix, vector, kernel, RREF, rank, subspace enumeration/intersection)
   - **L3 polynomial** (rings, interpolation, gcd, evaluation, division)
   - **L4 coding framework** (AbstractLinearCode, Encoder, Decoder, channels, GRS decoders)
   - **L5 misc** (caching, random state, divisors)
3. For each: *replaceability* (trivial / moderate / hard), required semantics (e.g. characteristic 2 only? extension degree?), and which tests exercise it.
4. Determine exactly which fields the project must support (q=2^m only? odd p? prime-power subfield chains for subgroups?) from code + tests + simulation CLIs — this drives which alternatives are viable.
5. Decoders deep-dive: read the source of Sage's `GRSGaoDecoder`, `GRSBerlekampWelchDecoder`, `GRSErrorErasureDecoder` in the installed passagemath; document the algorithms, the exact failure semantics (`DecodingError` conditions, return types) that our code relies on, so replacements can be behaviour-compatible.
6. Footprint audit: installed size of the current passagemath modular wheels (`du -sh`), wheel count, import time, Python-version support, license of each.
7. Produce an **interface specification** (`ff_backend` protocol: methods + semantics) that a replacement must implement — minimal set covering all L1–L5 usages.

## Acceptance checklist
- [ ] Complete symbol table with file:line
- [ ] Field-support requirement statement
- [ ] Sage GRS decoder behaviour spec
- [ ] Footprint table
- [ ] Draft `ff_backend` protocol
