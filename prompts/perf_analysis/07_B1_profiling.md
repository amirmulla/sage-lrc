# 07 — B1: Whole-program profiling & scaling

## Role
End-to-end performance engineer. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/B1_profiling.md`, `reports/perf_analysis/scratch/B1/`

## Tasks
1. Build a reproducible benchmark harness `scratch/B1/bench.py` (seeded, methodology from shared context) covering: construction (single-set, two-set), encode, every decoder, `minimum_distance_py`, one end-to-end simulation run.
2. `cProfile` + `pstats` (top 30 cumulative and tottime) for each workload; flame data via `py-spy` if installable (`uv pip install py-spy` in a scratch venv; record failure otherwise).
3. `tracemalloc` peak/steady-state memory per workload; import time (`python -X importtime`) — report cost of `import sage.all__sagemath_modules` and per-module lazy imports.
4. Scaling sweep q ∈ {16, 64, 256, 1024, 4096}: time and memory; log-log fit exponent; mark infeasible points with reason.
5. Attribute time to categories: **Sage field arithmetic**, **Sage polynomial**, **Sage matrix/linear algebra**, **Sage framework overhead (Encoder/Decoder/validation)**, **pure Python loops**, **NetworkX**, **I/O**. Provide a global pie-style table (% per category per workload).
6. Rank the top 10 hotspots globally with file:line.

## Acceptance checklist
- [ ] Harness runs end-to-end with one command; documented
- [ ] Category-attribution table for ≥ 6 workloads
- [ ] Scaling table + exponents
- [ ] Top-10 hotspot list
- [ ] No source changes
