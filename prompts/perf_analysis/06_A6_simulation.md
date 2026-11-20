# 06 — A6: simulation framework analysis

## Role
Monte-Carlo/IO performance analyst. Read `shared_context.md`. Read-only on source.

## Exclusive output
`reports/perf_analysis/A6_simulation.md`, `reports/perf_analysis/scratch/A6/`

## Tasks
1. Mandatory per-function row for every function in `simulation/runner.py` and `simulation/postprocess.py`, plus one row per CLI script in `simulation/*.py`.
2. Specific questions:
   - Per-trial overhead breakdown: error sampling (`sample_fixed_weight_error`), message generation, encode, channel, decode, comparison, logging. Profile % each.
   - `build_decoder` / `invoke_decoder`: per-trial reconstruction vs reuse.
   - `compute_code_weight_distribution`: enumeration complexity O(q^k·n); vectorization or sampling alternatives.
   - CSV writing: flush frequency, per-row open/close, pandas/csv comparison.
   - Parallelism: embarrassingly parallel trials — evaluate `multiprocessing` / `concurrent.futures` with seed-spawning (`SeedSequence`), pickle-ability of Sage objects, per-process warm-up cost, expected speed-up on `nproc` cores; measure a prototype in scratch only.
   - Statistical efficiency: confidence intervals, early stopping, importance sampling for rare failure events.
   - `process_error_distribution_csv`: streaming vs load-all; memory.
3. Provide a throughput table (trials/s) for each simulation type at GF(64) and GF(256).

## Acceptance checklist
- [ ] All functions/CLIs have rows
- [ ] Per-trial overhead breakdown table
- [ ] Parallel prototype speed-up measurement
- [ ] No out-of-scope changes
